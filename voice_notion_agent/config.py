"""Environment configuration for the voice agent.

Gmail and Notion are no longer authenticated via env vars in this file -
both are reached over MCP (see mcp_client.py), and each authenticates
itself using a token cached on disk by a one-time setup command. See
README.md / tutorial.md for that setup.
"""
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")


def validate() -> None:
    """Fail fast with a clear message if required secrets are missing."""
    missing = [
        name
        for name, value in [("OPENAI_API_KEY", OPENAI_API_KEY), ("TAVILY_API_KEY", TAVILY_API_KEY)]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
