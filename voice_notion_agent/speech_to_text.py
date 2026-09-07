"""Speech-to-text via the OpenAI Whisper API."""
from openai import OpenAI

from . import config
from .logging_utils import log_stage

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def transcribe_audio(file_path: str) -> str:
    """Transcribe a local audio file to text using OpenAI's Whisper model."""
    log_stage("Mic -> Whisper", input=f"<audio file: {file_path}>")
    with open(file_path, "rb") as audio_file:
        transcript = _client.audio.transcriptions.create(
            model=config.STT_MODEL,
            file=audio_file,
        )
    text = transcript.text.strip()
    log_stage("Whisper -> Orchestrator", output=text)
    return text
