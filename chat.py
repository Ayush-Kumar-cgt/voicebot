"""
Chat (LLM) Service using Groq.
Exposes:
  POST /chat   — Single-turn request/response.
  WS   /ws     — Multi-turn streaming chat (token-by-token).
  GET  /health — Liveness + model info.

NOTE: This service handles LLM only.
      TTS is the responsibility of tts_proxy.py (separate service on port 8001).
      Do NOT import or duplicate TTS logic here.

Swap-ready for local LLM later:
  Replace the Groq client with an OpenAI-compatible local endpoint
  (e.g. Ollama, llama-cpp-python server) by changing LLM_BASE_URL and LLM_API_KEY.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHAT_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS: int = 300
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT: str = (
    "You are a helpful voice assistant. "
    "Keep answers short and conversational — you are speaking out loud, not writing."
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("chat")

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Chat LLM Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in {"user", "assistant", "system"}:
            raise ValueError(f"Invalid role '{v}'. Must be user, assistant, or system.")
        return v


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message must not be empty.")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_messages(history: list[Message], user_message: str) -> list[dict]:
    """Prepend system prompt, append current user message."""
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + [m.model_dump() for m in history]
        + [{"role": "user", "content": user_message}]
    )


def _call_llm(messages: list[dict]) -> str:
    """Blocking Groq call — run via run_in_threadpool."""
    response = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


def _stream_llm(messages: list[dict]) -> list[str]:
    """
    Blocking Groq streaming call — collects all tokens and returns them.
    Run via run_in_threadpool so the event loop is never blocked.
    Yields tokens as a list so the caller can iterate in async context safely.
    """
    tokens: list[str] = []
    stream = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": CHAT_MODEL,
        "max_tokens": MAX_TOKENS,
    }


# ---------------------------------------------------------------------------
# REST endpoint: POST /chat
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Single-turn chat. Client manages and sends history on each request.

    Returns
    -------
    {"reply": "...", "model": "..."}
    """
    messages = build_messages(req.history, req.message)
    try:
        reply = await run_in_threadpool(_call_llm, messages)
    except Exception as exc:
        log.exception("LLM call failed")
        raise HTTPException(status_code=500, detail=str(exc))

    log.info("Chat reply: %d chars", len(reply))
    return {"reply": reply, "model": CHAT_MODEL}


# ---------------------------------------------------------------------------
# WebSocket endpoint: WS /ws
# ---------------------------------------------------------------------------
# Protocol (client → server, JSON):
#
#   Normal turn:
#   {"message": "Hello", "history": [...], "stream": true}
#       stream=true  → tokens sent one by one, then {"type": "done", "reply": "..."}
#       stream=false → single {"reply": "..."}
#
#   Interrupt (while streaming):
#   {"type": "interrupt"}
#       → server acknowledges with {"type": "interrupted"}
#         (interrupt is best-effort: stops sending further tokens)
#
# Server → client (JSON):
#   {"type": "token",  "token": "..."}   — one per streamed token
#   {"type": "done",   "reply": "..."}   — full reply when streaming ends
#   {"reply": "..."}                     — non-streaming response
#   {"type": "interrupted"}              — interrupt acknowledged
#   {"error": "..."}                     — on any failure
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    log.info("Chat WebSocket client connected: %s", websocket.client)

    interrupted: bool = False

    try:
        while True:
            payload = await websocket.receive_json()

            # ---- Interrupt signal ----
            if payload.get("type") == "interrupt":
                interrupted = True
                await websocket.send_json({"type": "interrupted"})
                log.info("Interrupt received.")
                continue

            # ---- Validate request ----
            try:
                req = ChatRequest(**payload)
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid chat payload: {exc}"})
                continue

            messages = build_messages(req.history, req.message)
            use_stream = bool(payload.get("stream", False))
            interrupted = False  # reset for new turn

            try:
                if use_stream:
                    # Collect all tokens in threadpool (blocking iterator kept off event loop)
                    tokens: list[str] = await run_in_threadpool(_stream_llm, messages)

                    reply_parts: list[str] = []
                    for token in tokens:
                        if interrupted:
                            break
                        reply_parts.append(token)
                        await websocket.send_json({"type": "token", "token": token})

                    full_reply = "".join(reply_parts)
                    await websocket.send_json({"type": "done", "reply": full_reply})
                    log.info("Streamed reply: %d tokens, %d chars", len(reply_parts), len(full_reply))

                else:
                    reply = await run_in_threadpool(_call_llm, messages)
                    await websocket.send_json({"reply": reply})
                    log.info("Reply: %d chars", len(reply))

            except Exception as exc:
                log.exception("LLM WebSocket error")
                await websocket.send_json({"error": str(exc)})

    except WebSocketDisconnect:
        log.info("Chat WebSocket client disconnected: %s", websocket.client)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")