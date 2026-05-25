"""
Local TTS Service using Kokoro.
Exposes:
  POST /generate   — Full text → single WAV response.
  WS   /ws         — Persistent socket: multiple sentences per connection,
                     each reply is: <binary WAV bytes> + {"type":"done"}
  GET  /health     — Liveness + device info.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from starlette.concurrency import run_in_threadpool

from kokoro import KModel, KPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE: int  = 24_000
DEVICE: str       = "cuda" if torch.cuda.is_available() else "cpu"
VOICE_DIR: Path   = Path("voices")
MODEL_CONFIG: str = "config.json"
MODEL_WEIGHTS: str = "kokoro-v1_0.pth"
DEFAULT_SPEED: float = 1.15

ALLOWED_VOICES: set[str] = {
    "af_heart", "af_bella", "af_sarah",
    "am_adam",  "am_michael", "am_santa",
    "bf_emma",  "bm_george",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("tts")

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------
log.info("Loading Kokoro model on %s …", DEVICE)
_model = KModel(config=MODEL_CONFIG, model=MODEL_WEIGHTS).to(DEVICE).eval()
_pipeline = KPipeline(lang_code="a", model=_model)
log.info("Kokoro model ready.")

# ---------------------------------------------------------------------------
# Voice cache — load each .pt file once
# ---------------------------------------------------------------------------
_voice_cache: dict[str, torch.Tensor] = {}

def get_voice(voice_name: str) -> torch.Tensor:
    if voice_name not in _voice_cache:
        path = VOICE_DIR / f"{voice_name}.pt"
        if not path.exists():
            raise RuntimeError(f"Voice file not found: {path}")
        log.info("Caching voice: %s", voice_name)
        _voice_cache[voice_name] = torch.load(path, weights_only=True)
    return _voice_cache[voice_name]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Kokoro TTS Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = DEFAULT_SPEED

    @field_validator("voice")
    @classmethod
    def voice_must_be_valid(cls, v: str) -> str:
        if v not in ALLOWED_VOICES:
            raise ValueError(f"Unknown voice '{v}'.")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text must not be empty.")
        return v

# ---------------------------------------------------------------------------
# Core synthesis (blocking — always call via run_in_threadpool)
# ---------------------------------------------------------------------------
def synthesise(text: str, voice_name: str, speed: float) -> BytesIO:
    voice  = get_voice(voice_name)
    chunks = [r.audio for r in _pipeline(text, voice=voice, speed=speed) if r.audio is not None]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio.")
    audio = torch.cat(chunks).numpy()
    buf = BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    log.info("Synthesised %d chars → %d bytes WAV", len(text), buf.getbuffer().nbytes)
    return buf

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "sample_rate": SAMPLE_RATE,
            "allowed_voices": sorted(ALLOWED_VOICES)}

# ---------------------------------------------------------------------------
# REST: POST /generate
# ---------------------------------------------------------------------------
@app.post("/generate")
async def generate(req: TTSRequest):
    try:
        buf = await run_in_threadpool(synthesise, req.text, req.voice, req.speed)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(buf, media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'})

# ---------------------------------------------------------------------------
# WebSocket: /ws
# ---------------------------------------------------------------------------
# Protocol per message (client → server):
#   {"text": "...", "voice": "af_heart", "speed": 1.15}
#
# Protocol per message (server → client):
#   <binary WAV bytes>          — the synthesised audio
#   {"type": "done"}            — signals this sentence is complete
#   {"error": "..."}            — on failure
#
# The socket stays open for multiple sentences (one request/reply per sentence).
# Client closes the socket when the turn is finished.
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def tts_ws(websocket: WebSocket):
    await websocket.accept()
    log.info("TTS WS connected: %s", websocket.client)

    try:
        while True:
            # Each sentence arrives as a separate JSON text frame
            try:
                payload = await websocket.receive_json()
            except Exception:
                break  # client closed the socket — normal exit

            try:
                req = TTSRequest(**payload)
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid payload: {exc}"})
                continue

            try:
                buf = await run_in_threadpool(synthesise, req.text, req.voice, req.speed)
                await websocket.send_bytes(buf.getvalue())
                await websocket.send_json({"type": "done"})  # sentence complete signal
            except RuntimeError as exc:
                await websocket.send_json({"error": str(exc)})
            except Exception as exc:
                log.exception("TTS synthesis error")
                await websocket.send_json({"error": str(exc)})

    except WebSocketDisconnect:
        pass
    finally:
        log.info("TTS WS disconnected: %s", websocket.client)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")