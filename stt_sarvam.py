import os
import io
import json
import wave
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydub import AudioSegment
from pydub.effects import normalize
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
ALLOWED_PROVIDERS = {"sarvam", "whisper"}

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("small", device="auto", compute_type="auto")
    return _whisper_model

def convert_to_wav(audio_bytes: bytes) -> bytes:
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read recorded audio. Check browser recording format and FFmpeg. {e}"
        )
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio = normalize(audio, headroom=1.0)
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    return wav_buffer.getvalue()

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_buffer.getvalue()

app = FastAPI(title="STT Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/stt")
async def transcribe(
    file: UploadFile = File(...),
    provider: str = "sarvam"
):
    if provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown STT provider.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    try:
        if provider == "whisper":
            text = await run_in_threadpool(transcribe_whisper, audio_bytes)
        else:
            text = await transcribe_sarvam(audio_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"transcript": text, "provider": provider}

@app.websocket("/ws")
async def transcribe_ws(websocket: WebSocket, provider: str = "sarvam"):
    await websocket.accept()

    if provider not in ALLOWED_PROVIDERS:
        await websocket.send_json({"error": "Unknown STT provider."})
        await websocket.close(code=1008)
        return

    pcm_chunks = bytearray()
    sample_rate = 16000
    streaming = False

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid websocket JSON."})
                    continue

                if payload.get("type") == "start":
                    pcm_chunks.clear()
                    sample_rate = int(payload.get("sampleRate", 16000))
                    streaming = payload.get("audioFormat") == "pcm_s16le"
                    await websocket.send_json({"type": "ready"})
                    continue

                if payload.get("type") == "stop":
                    if not pcm_chunks:
                        # Gracefully return blank instead of crashing frontend
                        await websocket.send_json({"transcript": "", "provider": provider})
                        continue

                    was_streaming = streaming
                    audio_bytes = pcm_to_wav(bytes(pcm_chunks), sample_rate) if was_streaming else bytes(pcm_chunks)
                    pcm_chunks.clear()
                    streaming = False
                    await send_transcript(websocket, provider, audio_bytes, already_wav=was_streaming)
                    continue

                if payload.get("type") == "flush":
                    if not pcm_chunks:
                        continue # Ignore empty flushes
                    
                    was_streaming = streaming
                    audio_bytes = pcm_to_wav(bytes(pcm_chunks), sample_rate) if was_streaming else bytes(pcm_chunks)
                    pcm_chunks.clear()
                    # We purposely do NOT set streaming = False here so it keeps listening
                    await send_transcript(websocket, provider, audio_bytes, already_wav=was_streaming)
                    continue

                await websocket.send_json({"error": "Unknown websocket message."})
                continue

            audio_bytes = message.get("bytes")
            if not audio_bytes:
                await websocket.send_json({"error": "Empty audio file."})
                continue

            if streaming:
                pcm_chunks.extend(audio_bytes)
            else:
                await send_transcript(websocket, provider, audio_bytes)
    except WebSocketDisconnect:
        pass

async def send_transcript(
    websocket: WebSocket,
    provider: str,
    audio_bytes: bytes,
    already_wav: bool = False
):
    try:
        if provider == "whisper":
            text = await run_in_threadpool(transcribe_whisper, audio_bytes, already_wav)
        else:
            text = await transcribe_sarvam(audio_bytes, already_wav)
        await websocket.send_json({"transcript": text, "provider": provider})
    except HTTPException as e:
        await websocket.send_json({"error": e.detail})
    except Exception as e:
        await websocket.send_json({"error": str(e)})

async def transcribe_sarvam(audio_bytes: bytes, already_wav: bool = False) -> str:
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY is not set.")

    wav_bytes = audio_bytes if already_wav else convert_to_wav(audio_bytes)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"language_code": "en-IN", "model": "saaras:v3"},
            timeout=30.0
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Sarvam STT error: {response.text}"
            )
        result = response.json()
        return result.get("transcript", "")

def transcribe_whisper(audio_bytes: bytes, already_wav: bool = False) -> str:
    wav_bytes = audio_bytes if already_wav else convert_to_wav(audio_bytes)
    model = get_whisper_model()
    segments, _ = model.transcribe(io.BytesIO(wav_bytes), language="unknown")
    return " ".join(seg.text for seg in segments).strip()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)