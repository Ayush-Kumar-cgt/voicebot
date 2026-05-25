"""
Voicebot Orchestrator — serves the frontend and proxies the full pipeline.


Run:
  uvicorn orchestrator:app --host 0.0.0.0 --port 8000 --reload

The browser talks ONLY to this service on port 8000.
This service fans out to:
  STT  → ws://localhost:8003/ws
  Chat → ws://localhost:8002/ws
  TTS  → ws://localhost:8001/ws
"""

import json
import logging
from pathlib import Path

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).parent / "static"

STT_WS  = "ws://localhost:8003/ws"
CHAT_WS = "ws://localhost:8002/ws"
TTS_WS  = "ws://localhost:8001/ws"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Voicebot Orchestrator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "services": {"stt": STT_WS, "chat": CHAT_WS, "tts": TTS_WS}}


# ---------------------------------------------------------------------------
# STT proxy WebSocket  →  /ws/stt
# Browser sends raw PCM binary chunks + JSON control frames.
# Orchestrator forwards everything verbatim to stt.py.
# ---------------------------------------------------------------------------
@app.websocket("/ws/stt")
async def proxy_stt(client: WebSocket):
    await client.accept()
    log.info("STT proxy connected")
    try:
        async with websockets.connect(STT_WS) as upstream:
            async def client_to_upstream():
                while True:
                    message = await client.receive()
                    if "bytes" in message and message["bytes"]:
                        await upstream.send(message["bytes"])
                    elif "text" in message and message["text"]:
                        await upstream.send(message["text"])

            async def upstream_to_client():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await client.send_bytes(message)
                    else:
                        await client.send_text(message)

            import asyncio
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_upstream()),
                    asyncio.create_task(upstream_to_client()),
                ],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
    except WebSocketDisconnect:
        log.info("STT client disconnected")
    except Exception as exc:
        log.warning("STT proxy error: %s", exc)


# ---------------------------------------------------------------------------
# Chat proxy WebSocket  →  /ws/chat
# Browser sends JSON chat payloads.
# Server-side history is managed HERE so JS never needs to track it.
# ---------------------------------------------------------------------------
@app.websocket("/ws/chat")
async def proxy_chat(client: WebSocket):
    await client.accept()
    log.info("Chat proxy connected")

    history: list[dict] = []  # server-side history — JS sends message only

    try:
        async with websockets.connect(CHAT_WS) as upstream:

            async def upstream_to_client():
                """Forward LLM tokens/replies back to browser."""
                full_reply_parts: list[str] = []
                async for raw in upstream:
                    data = json.loads(raw)

                    if data.get("type") == "token":
                        full_reply_parts.append(data["token"])
                        await client.send_text(raw)

                    elif data.get("type") == "done":
                        # Store completed assistant turn in server history
                        history.append({"role": "assistant", "content": data.get("reply", "")})
                        full_reply_parts.clear()
                        await client.send_text(raw)

                    elif "reply" in data:
                        # Non-streaming reply
                        history.append({"role": "assistant", "content": data["reply"]})
                        await client.send_text(raw)

                    else:
                        # error / interrupted / unknown — pass through
                        await client.send_text(raw)

            import asyncio
            upstream_task = asyncio.create_task(upstream_to_client())

            while True:
                message = await client.receive()
                if "text" not in message or not message["text"]:
                    continue

                payload = json.loads(message["text"])

                # Interrupt signal — pass through, no history mutation
                if payload.get("type") == "interrupt":
                    await upstream.send(message["text"])
                    continue

                # Normal chat turn — inject server-side history, strip client history
                user_message = payload.get("message", "")
                if user_message.strip():
                    history.append({"role": "user", "content": user_message})

                enriched = {
                    "message": user_message,
                    "history": history[:-1],  # exclude the just-added user msg (chat.py appends it)
                    "stream": payload.get("stream", True),
                }
                await upstream.send(json.dumps(enriched))

    except WebSocketDisconnect:
        log.info("Chat client disconnected")
    except Exception as exc:
        log.warning("Chat proxy error: %s", exc)


# ---------------------------------------------------------------------------
# TTS proxy WebSocket  →  /ws/tts
# Uses a true bidirectional pipe: both directions run concurrently and the
# connection stays open for the ENTIRE bot turn (multiple sentences).
# Only closes when the browser disconnects or tts_proxy errors.

@app.websocket("/ws/tts")
async def proxy_tts(client: WebSocket):
    await client.accept()
    log.info("TTS proxy connected: %s", client.client)
    try:
        async with websockets.connect(TTS_WS) as upstream:

            async def forward_to_upstream():
                try:
                    while True:
                        msg = await client.receive()
                        log.info("TTS proxy ← browser: %s", str(msg)[:120])
                        if msg["type"] == "websocket.disconnect":
                            log.info("TTS proxy: browser disconnected")
                            break
                        if "text" in msg and msg["text"]:
                            log.info("TTS proxy → upstream: %s", msg["text"][:120])
                            await upstream.send(msg["text"])
                        elif "bytes" in msg and msg["bytes"]:
                            await upstream.send(msg["bytes"])
                except Exception as e:
                    log.warning("TTS forward_to_upstream exception: %s", e)

            async def forward_to_client():
                try:
                    async for msg in upstream:
                        if isinstance(msg, bytes):
                            log.info("TTS proxy ← upstream: binary %d bytes", len(msg))
                            await client.send_bytes(msg)
                        else:
                            log.info("TTS proxy ← upstream: text %s", str(msg)[:120])
                            await client.send_text(msg)
                    log.info("TTS proxy: upstream closed normally")
                except Exception as e:
                    log.warning("TTS forward_to_client exception: %s", e)

            # Run both directions concurrently.
            # gather() keeps both alive; returns when BOTH finish.
            import asyncio
            await asyncio.gather(
                forward_to_upstream(),
                forward_to_client(),
                return_exceptions=True,
            )

    except WebSocketDisconnect:
        log.info("TTS client disconnected: %s", client.client)
    except Exception as exc:
        log.warning("TTS proxy error: %s", exc)
    finally:
        log.info("TTS proxy closed: %s", client.client)



# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")