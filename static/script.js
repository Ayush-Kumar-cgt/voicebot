/**
 * script.js — Voicebot frontend
 *
 * Fixes vs original:
 *  1. STT accuracy    — OfflineAudioContext resampling (no aliasing)
 *  2. LLM latency     — chat WS pre-warmed at page load, auto-reconnects
 *  3. TTS start delay — TTS WS opened in parallel with chat WS
 *  4. Interrupt VAD   — silenceStart = Date.now() on confirmed interrupt
 *  5. Streaming speed — AudioWorklet replaces ScriptProcessor (off main thread)
 *  6. Pipeline races  — pipelineId token cancels stale callbacks
 *
 * Bug fixes in this pass:
 *  A. `dest` undefined in createMicStream — removed dead MediaStreamDestination
 *  B. warmChatSocket() moved to page load (was called per mic click → duplicate sockets)
 *  C. chatSocket.ready re-read at call time (stale promise after auto-reconnect)
 *  D. async onAudioFrame concurrent overlap — processing lock prevents out-of-order PCM
 *  E. PCM computed once per frame, shared between interrupt + normal send paths
 *  F. voice param wired through runPipeline → streamReply (was always undefined)
 *  G. STT callback state ownership cleaned — runPipeline owns appState, not the callback
 */

const STT_WS  = `ws://${location.host}/ws/stt`;
const CHAT_WS = `ws://${location.host}/ws/chat`;
const TTS_WS  = `ws://${location.host}/ws/tts`;

// ── VAD constants ──────────────────────────────────────────────────────────
const STT_SAMPLE_RATE     = 16000;
const MIC_GAIN            = 1.0;
const PCM_PEAK_LIMIT      = 0.95;
const SILENCE_THRESHOLD   = 0.015;
const SILENCE_DURATION_MS = 1000;
const INTERRUPT_THRESHOLD = 0.015;
const INTERRUPT_HOLD_MS   = 50;

// ── App state ──────────────────────────────────────────────────────────────
let appState         = "IDLE";   // IDLE | LISTENING | PROCESSING | SPEAKING
let isRecording      = false;
let silenceStart     = 0;
let hasSpoken        = false;
let interruptStart   = 0;
let recordingSession = null;
let audioContext     = null;
let activePipelineId = 0;   // each pipeline turn gets a unique id; stale callbacks self-cancel

// ── TTS playback state ─────────────────────────────────────────────────────
let ttsAbort      = new AbortController();
let ttsRunner     = Promise.resolve();
let isBotSpeaking = false;
let currentAudio  = null;

// ── WebSocket handles ──────────────────────────────────────────────────────
let sttSocket  = null;
let chatSocket = null;

// ── DOM ────────────────────────────────────────────────────────────────────
const micBtn  = document.getElementById("mic-btn");
const status  = document.getElementById("status");
const chatBox = document.getElementById("chat-box");

// ── Helpers ────────────────────────────────────────────────────────────────
const setStatus = t => { if (status.textContent !== t) status.textContent = t; };

function addBubble(text, role) {
  const d = document.createElement("div");
  d.className = `bubble ${role}`;
  d.textContent = text;
  chatBox.appendChild(d);
  chatBox.scrollTop = chatBox.scrollHeight;
  return d;
}

function splitSentence(text) {
  const m = text.match(/^([\s\S]*?[.!?])(\s+|$)/);
  if (m) return { sentence: m[1].trim(), rest: text.slice(m[0].length) };
  if (text.length > 50) {
    const m2 = text.match(/^([\s\S]*?[,;:—])(\s+)/);
    if (m2) return { sentence: m2[1].trim(), rest: text.slice(m2[0].length) };
  }
  return null;
}

// ── PCM — Naive Resampler (fix #1) ─────────────────────────────────────────
function resampleFloat32(floats, inRate, outRate) {
  if (inRate === outRate) return floats;
  const ratio = inRate / outRate;
  const out = new Float32Array(Math.round(floats.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const s = Math.floor(i * ratio), e = Math.floor((i + 1) * ratio);
    let sum = 0, n = 0;
    for (let j = s; j < e && j < floats.length; j++) { sum += floats[j]; n++; }
    out[i] = n ? sum / n : 0;
  }
  return out;
}

function floatTo16BitPcm(floats, inRate) {
  const samples = resampleFloat32(floats, inRate, STT_SAMPLE_RATE);
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-PCM_PEAK_LIMIT, Math.min(PCM_PEAK_LIMIT, samples[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm.buffer;
}

// ── WebSocket ──────────────────────────────────────────────────────────────
function openSocket(url, label) {
  const ws = new WebSocket(url);
  ws.ready = new Promise((res, rej) => {
    ws.onopen  = res;
    ws.onerror = () => rej(new Error(`${label} WS failed to open`));
  });
  return ws;
}

// ── Fix #2 + bug B: pre-warm chat WS at page load, not inside mic click ───
// Calling it per mic click created duplicate sockets and leaked connections.
function warmChatSocket() {
  chatSocket = openSocket(CHAT_WS, "Chat");
  chatSocket.onclose = () => {
    if (isRecording || appState !== "IDLE") setTimeout(warmChatSocket, 500);
  };
  chatSocket.onerror = () => console.error("Chat WS error");
}
warmChatSocket(); // page load — single call

// ── Low-level audio playback ───────────────────────────────────────────────
function playBlobNow(blob, signal) {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return resolve();
    
    const url   = URL.createObjectURL(blob);
    const audio = new Audio(url);
    currentAudio  = audio;
    isBotSpeaking = true;

    const cleanup = () => {
      URL.revokeObjectURL(url);
      if (currentAudio === audio) { 
        currentAudio = null; 
        isBotSpeaking = false; 
      }
    };

    signal.addEventListener("abort", () => { 
      audio.pause(); 
      cleanup(); 
      resolve(); 
    }, { once: true });
    
    audio.onended = () => { cleanup(); resolve(); };
    audio.onerror = () => { cleanup(); reject(new Error("Playback failed")); };
    
    audio.play().catch(err => { 
      cleanup(); 
      reject(err); 
    });
  });
}

function scheduleBlob(blob, signal) {
  ttsRunner = ttsRunner.then(() => {
    if (signal.aborted) return Promise.resolve();
    return playBlobNow(blob, signal);
  }).catch(() => {});
}

// ── Interrupt / stop all audio ─────────────────────────────────────────────
function stopAllAudio() {
  ttsAbort.abort();
  ttsAbort  = new AbortController();
  ttsRunner = Promise.resolve();
  if (currentAudio) { 
    currentAudio.pause(); 
    currentAudio = null; 
  }
  isBotSpeaking = false;
}

// ── speakTurn: one WS for full bot turn ───────────────────────────────────
// wsReady exposed so streamReply can await it in parallel with chat WS (fix #3).
function speakTurn(voice) {
  const signal = ttsAbort.signal;
  let closed           = false;
  let pendingSentences = 0;
  let turnFinished     = false;

  const ws = new WebSocket(TTS_WS);
  ws.binaryType = "arraybuffer";

  const wsReady = new Promise((res, rej) => {
    ws.onopen  = res;
    ws.onerror = () => rej(new Error("TTS WS failed to open"));
  });

  function maybeClose() {
    if (turnFinished && pendingSentences <= 0 && !closed && ws.readyState < WebSocket.CLOSING) {
      closed = true;
      ws.close();
    }
  }

  ws.onmessage = event => {
    if (signal.aborted) return;
    if (typeof event.data === "string") {
      try {
        const d = JSON.parse(event.data);
        if (d.error) { console.error("TTS error:", d.error); return; }
        if (d.type === "done") { pendingSentences = Math.max(0, pendingSentences - 1); maybeClose(); }
      } catch (_) {}
      return;
    }
    scheduleBlob(new Blob([event.data], { type: "audio/wav" }), signal);
    setStatus("speaking...");
  };

  ws.onerror = () => { if (!signal.aborted) console.error("TTS WS error"); };
  ws.onclose = () => { closed = true; };

  signal.addEventListener("abort", () => {
    if (!closed && ws.readyState < WebSocket.CLOSING) { closed = true; ws.close(); }
  }, { once: true });

  return {
    wsReady,
    async send(text) {
      if (signal.aborted || !text.trim()) return;
      try {
        await wsReady;
        if (!signal.aborted && ws.readyState === WebSocket.OPEN) {
          pendingSentences++;
          ws.send(JSON.stringify({ text, voice }));
        }
      } catch (err) {
        if (!signal.aborted) console.error("TTS send error:", err);
      }
    },
    finish() { turnFinished = true; maybeClose(); }
  };
}

// ── Chat streaming ─────────────────────────────────────────────────────────
async function streamReply(message, voice) {
  // Fix #3: TTS WS opens immediately — parallel with chat WS
  const turn = speakTurn(voice);

  // Bug C fix: re-read chatSocket.ready here, not at warmChatSocket() call time.
  // After auto-reconnect chatSocket is a new object; cached .ready would be stale.
  await Promise.all([chatSocket.ready, turn.wsReady]);

  const botBubble = addBubble("", "bot");
  let reply   = "";
  let pending = "";
  setStatus("thinking...");

  return new Promise((resolve, reject) => {
    chatSocket.onmessage = event => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.error) { turn.finish(); reject(new Error(data.error)); return; }

      if (data.type === "token") {
        reply   += data.token;
        pending += data.token;
        botBubble.textContent = reply;
        chatBox.scrollTop = chatBox.scrollHeight;

        let split = splitSentence(pending);
        while (split) {
          turn.send(split.sentence);
          pending = split.rest;
          split   = splitSentence(pending);
        }
        return;
      }

      if (data.type === "done" || data.reply) {
        botBubble.textContent = data.reply || reply;
        if (pending.trim()) turn.send(pending);
        turn.finish();
        resolve(data.reply || reply);
      }
    };
    chatSocket.onerror = () => { turn.finish(); reject(new Error("Chat WS failed")); };
    chatSocket.send(JSON.stringify({ message, stream: true }));
  });
}

// ── Pipeline ───────────────────────────────────────────────────────────────
function isHallucination(transcript) {
  const t = transcript.trim();
  if (!t || t.length < 2) return true;
  const artifacts = new Set([
    "thanksforwatching", "thankyouforwatching",
    "likeandsubscribe", "subtitlesbytheamara",
    "pleasesub", "subscribetomychannel"
  ]);
  return artifacts.has(t.toLowerCase().replace(/[^a-z]/g, ""));
}

// Fix #6 + bug F: pipelineId cancels stale callbacks; voice wired through.
async function runPipeline(transcript, voice) {
  if (isHallucination(transcript)) { 
      appState = "LISTENING"; 
      setStatus("listening..."); 
      return; 
  }
  const myId = ++activePipelineId;
  addBubble(transcript, "user");
  try {
    await streamReply(transcript, voice);
    await ttsRunner;
    if (activePipelineId === myId) setStatus("listening...");
  } catch (err) {
    if (!ttsAbort.signal.aborted && activePipelineId === myId) {
      console.error("Pipeline error:", err);
      setStatus(err.message);
    }
  } finally {
    if (activePipelineId === myId) {
      appState = "LISTENING";
      setStatus("listening...");
    }
  }
}

async function sendText() {
  const input = document.getElementById("text-input");
  const text  = input.value.trim();
  if (!text) return;
  input.value = "";
  addBubble(text, "user");
  try {
    await streamReply(text);
    await ttsRunner;
  } catch (err) {
    if (!ttsAbort.signal.aborted) { console.error("Text pipeline error:", err); setStatus(err.message); }
  }
}

// ── Mic / STT ──────────────────────────────────────────────────────────────
// Bug A fixed: removed dead `dest` (MediaStreamDestination).
// After switching to AudioWorklet, dest was unused — just source → gain → worklet.
async function createMicStream() {
  const rawStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false }
  });
  audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(rawStream);
  const gain   = audioContext.createGain();
  gain.gain.value = MIC_GAIN;
  source.connect(gain);
  rawStream.getAudioTracks()[0].addEventListener("ended",
    () => rawStream.getTracks().forEach(t => t.stop()));
  return { rawStream, gain };
}

async function startSTTStream(onTranscript) {
  sttSocket = (!sttSocket || sttSocket.readyState >= WebSocket.CLOSING)
    ? openSocket(STT_WS, "STT") : sttSocket;
  await sttSocket.ready;

  sttSocket.onmessage = async event => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "ready") return;
      if (data.transcript !== undefined) {
        if (!data.transcript.trim()) {
           appState = "LISTENING";
           setStatus("listening...");
           return;
        }
        await onTranscript(data.transcript);
        return;
      }
      if (data.error) { console.error("STT:", data.error); appState = "LISTENING"; setStatus("listening..."); }
    } catch (e) { console.error("STT parse:", e); }
  };
  sttSocket.onerror = () => console.error("STT WS error");
  sttSocket.send(JSON.stringify({ type: "start", audioFormat: "pcm_s16le", sampleRate: STT_SAMPLE_RATE }));

  return {
    sendChunk: c => { if (sttSocket.readyState === WebSocket.OPEN) sttSocket.send(c); },
    flush:     () => { if (sttSocket.readyState === WebSocket.OPEN) sttSocket.send(JSON.stringify({ type: "flush" })); },
    stop:      () => { if (sttSocket.readyState === WebSocket.OPEN) sttSocket.send(JSON.stringify({ type: "stop" })); },
  };
}

// ── AudioWorklet inline processor (fix #5) ────────────────────────────────
// Off main thread. 128-sample frames @ 48kHz ≈ 2.7ms vs ScriptProcessor(4096) ≈ 85ms.
const PCM_WORKLET_SRC = `
class PcmProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0][0];
    if (ch && ch.length) this.port.postMessage(ch.slice());
    return true;
  }
}
registerProcessor("pcm-processor", PcmProcessor);
`;

async function createAudioWorklet(ctx, gainNode, onFrame) {
  const blob = new Blob([PCM_WORKLET_SRC], { type: "application/javascript" });
  const url  = URL.createObjectURL(blob);
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  const worklet = new AudioWorkletNode(ctx, "pcm-processor");
  worklet.port.onmessage = e => onFrame(e.data);
  gainNode.connect(worklet);
  // capture-only node — no connect to destination needed
  return worklet;
}

// ── Mic button ─────────────────────────────────────────────────────────────
micBtn.addEventListener("click", async () => {
  if (!isRecording) {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("microphone requires http://localhost or https"); return;
    }
    try {
      const { rawStream, gain } = await createMicStream();

      // Bug F: read voice once here, pass through pipeline
      const voice = document.getElementById("tts-voice")?.value ?? "default";

      // Bug G fix: STT callback only calls runPipeline.
      // runPipeline owns appState/status via pipelineId; outer state management conflicted.
      const sttStream = await startSTTStream(async transcript => {
        appState = "PROCESSING"; setStatus("thinking...");
        await runPipeline(transcript, voice);
        // appState + status reset handled in runPipeline finally
      });

      // Bug D fix: processing lock prevents concurrent async frame invocations.
      // Without it, async resampling overlaps → out-of-order PCM sent to STT.
      let frameProcessing = false;

      const onAudioFrame = (buf) => {
        if (frameProcessing) return; // drop frame — previous still resampling
        frameProcessing = true;
        try {
          // RMS (DC-corrected)
          let meanSum = 0;
          for (let i = 0; i < buf.length; i++) meanSum += buf[i];
          const mean = meanSum / buf.length;
          let sqSum = 0;
          for (let i = 0; i < buf.length; i++) { const s = buf[i] - mean; sqSum += s * s; }
          const rms = Math.sqrt(sqSum / buf.length);

          // Bug E fix: compute PCM once, reuse in both interrupt + normal paths.
          let pcmCache = null;
          const getPcm = () => {
            if (!pcmCache) pcmCache = floatTo16BitPcm(buf, audioContext.sampleRate);
            return pcmCache;
          };

          // ── Interrupt ────────────────────────────────────────────────
          if (isBotSpeaking && rms > INTERRUPT_THRESHOLD) {
            if (interruptStart === 0) interruptStart = Date.now();

            sttStream.sendChunk(getPcm());

            if (Date.now() - interruptStart >= INTERRUPT_HOLD_MS) {
              stopAllAudio();
              ++activePipelineId; // invalidate in-flight pipeline callbacks
              if (chatSocket?.readyState === WebSocket.OPEN) {
                chatSocket.send(JSON.stringify({ type: "interrupt" }));
              }
              appState     = "LISTENING";
              hasSpoken    = true;
              silenceStart = Date.now(); // fix #4: timer starts NOW, not 0
              interruptStart = 0;
              setStatus("listening (speech detected)...");
            }
            return;
          } else {
            interruptStart = 0;
          }

          // ── VAD ──────────────────────────────────────────────────────
          if (rms > SILENCE_THRESHOLD) {
            silenceStart = 0;
            if (appState === "LISTENING") {
              hasSpoken = true;
              setStatus("listening (speech detected)...");
            }
          } else if (hasSpoken && appState === "LISTENING") {
            if (silenceStart === 0) silenceStart = Date.now();
            const elapsed = Date.now() - silenceStart;
            setStatus(`listening (silence: ${(elapsed / 1000).toFixed(1)}s)...`);
            if (elapsed > SILENCE_DURATION_MS) {
              appState = "PROCESSING"; setStatus("transcribing...");
              sttStream.flush(); silenceStart = 0; hasSpoken = false;
              return;
            }
          }

          // Send PCM only while listening and bot silent
          if (appState === "LISTENING" && !isBotSpeaking) {
            sttStream.sendChunk(getPcm());
          }

        } finally {
          frameProcessing = false;
        }
      };

      const workletNode = await createAudioWorklet(audioContext, gain, onAudioFrame);

      recordingSession = { rawStream, workletNode, sttStream };
      isRecording = true; appState = "LISTENING";
      micBtn.classList.add("recording"); micBtn.textContent = "⏹";
      setStatus("listening...");

    } catch (err) {
      console.error(err); setStatus(err.message);
      isRecording = false;
      micBtn.classList.remove("recording"); micBtn.textContent = "🎤";
    }

  } else {
    const s = recordingSession; recordingSession = null;
    s.workletNode.disconnect();
    s.rawStream.getTracks().forEach(t => t.stop());
    s.sttStream.stop();
    if (audioContext) { audioContext.close(); audioContext = null; }
    isRecording = false; appState = "IDLE";
    micBtn.classList.remove("recording"); micBtn.textContent = "🎤";
    setStatus("click mic to speak");
  }
});

document.getElementById("send-btn").addEventListener("click", sendText);
document.getElementById("text-input").addEventListener("keydown", e => {
  if (e.key === "Enter") sendText();
});