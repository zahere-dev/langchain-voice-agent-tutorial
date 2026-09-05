"""Text-to-speech via the OpenAI TTS API."""
import tempfile

from openai import OpenAI

from . import config
from .logging_utils import log_stage

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def synthesize_speech(text: str) -> str:
    """Convert text to spoken audio and return the path to an MP3 file."""
    log_stage("Orchestrator -> TTS", input=text)
    tmp_path = tempfile.mktemp(suffix=".mp3")
    with _client.audio.speech.with_streaming_response.create(
        model=config.TTS_MODEL,
        voice=config.TTS_VOICE,
        input=text,
    ) as response:
        response.stream_to_file(tmp_path)
    log_stage("TTS -> Speaker", output=f"<audio file: {tmp_path}>")
    return tmp_path
