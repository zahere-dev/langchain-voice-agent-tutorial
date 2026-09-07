"""FastAPI web demo: browser mic -> STT -> agent -> Gmail/Notion (MCP) -> TTS -> browser audio.

This wraps the same voice_notion_agent package used by the CLI in main.py.
It exists so the project can be demoed live from a shareable URL (e.g. on
Render) instead of only from a local terminal.

State is kept in a plain in-memory dict keyed by a client-generated session
id. That's fine for a single-instance demo deployment; it is not meant to
survive restarts or scale beyond one process.

SECURITY NOTE: the agent now has real Gmail (read/send) and Notion access
via MCP. If AUTH_PASSWORD is set, every route (including the page itself)
requires HTTP Basic Auth - the browser's native login prompt, checked
against AUTH_USERNAME/AUTH_PASSWORD. Set this before exposing the service
on a public URL, otherwise anyone with the link can read or send from your
real Gmail account.
"""
import base64
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain.messages import HumanMessage
from openai import BadRequestError

from voice_notion_agent import config
from voice_notion_agent.agent import build_agent
from voice_notion_agent.logging_utils import log_stage
from voice_notion_agent.speech_to_text import transcribe_audio
from voice_notion_agent.text_to_speech import synthesize_speech

config.validate()

_agent = None
_sessions: dict[str, list] = {}
_MAX_HISTORY_MESSAGES = 20

_AUTH_USERNAME = os.getenv("AUTH_USERNAME", "demo")
_AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")
_basic_auth = HTTPBasic(auto_error=False)


def _require_auth(credentials: HTTPBasicCredentials | None = Depends(_basic_auth)) -> None:
    """No-op if AUTH_PASSWORD isn't set; otherwise requires matching HTTP Basic Auth.

    auto_error=False on HTTPBasic means a missing Authorization header comes
    back as None instead of an immediate 401 - that's what lets local dev
    (no AUTH_PASSWORD set) skip auth entirely rather than always prompting.
    """
    if not _AUTH_PASSWORD:
        return
    valid = credentials is not None and (
        secrets.compare_digest(credentials.username, _AUTH_USERNAME)
        and secrets.compare_digest(credentials.password, _AUTH_PASSWORD)
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = await build_agent()
    yield


app = FastAPI(title="Voice Gmail/Notion Agent Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", dependencies=[Depends(_require_auth)])
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


async def _run_agent_turn(session_id: str, transcript: str) -> str:
    log_stage("User -> Orchestrator", input=transcript)
    messages = _sessions.setdefault(session_id, [])
    messages.append(HumanMessage(content=transcript))
    result = await _agent.ainvoke({"messages": messages})
    messages = result["messages"][-_MAX_HISTORY_MESSAGES:]
    _sessions[session_id] = messages
    reply = messages[-1].content
    log_stage("Orchestrator -> User", output=reply)
    return reply


@app.post("/api/chat", dependencies=[Depends(_require_auth)])
async def chat(message: str = Form(...), session_id: str = Form(...)) -> JSONResponse:
    """Text-only turn, used by the web UI's fallback text box."""
    reply = await _run_agent_turn(session_id, message)
    return JSONResponse({"transcript": message, "reply": reply})


@app.post("/api/voice", dependencies=[Depends(_require_auth)])
async def voice(audio: UploadFile = File(...), session_id: str = Form(...)) -> JSONResponse:
    """Voice turn: browser sends a recorded clip, we return text + spoken reply."""
    # Keep whatever extension the browser actually sent (webm/ogg/mp4) -
    # Whisper uses it as a hint for which container/codec to decode, so
    # saving everything as .webm regardless of the real format is what
    # causes "could not be decoded" errors on browsers that don't record
    # webm/opus.
    suffix = Path(audio.filename or "command.webm").suffix or ".webm"
    tmp_path = f"/tmp/{uuid.uuid4()}{suffix}"
    audio_bytes = await audio.read()
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    print(
        f"[voice] received {len(audio_bytes)} bytes, "
        f"filename={audio.filename!r}, content_type={audio.content_type!r}, saved to {tmp_path}"
    )

    try:
        transcript = transcribe_audio(tmp_path)
    except BadRequestError as exc:
        print(f"[voice] transcription failed: {exc}")
        return JSONResponse(
            {
                "transcript": "",
                "reply": "That clip was too short or couldn't be understood - press and hold while you speak.",
                "audio_base64": None,
            }
        )

    print(f"[voice] transcript: {transcript!r}")

    if not transcript:
        return JSONResponse(
            {"transcript": "", "reply": "I didn't catch that, try again.", "audio_base64": None}
        )

    reply = await _run_agent_turn(session_id, transcript)

    speech_path = synthesize_speech(reply)
    with open(speech_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    return JSONResponse({"transcript": transcript, "reply": reply, "audio_base64": audio_b64})
