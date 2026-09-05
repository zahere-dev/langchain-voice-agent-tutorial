"""A tiny helper for consistent, truncated stage-handoff logging.

Every stage boundary in the pipeline (STT -> Orchestrator -> specialist ->
TTS) prints one line showing what went in and/or what came out, truncated
so a long transcript or page of research doesn't flood the console. This
is purely a debugging/demo aid - not wired into LangSmith, which already
captures the full untruncated payloads for anything that needs it.
"""

_MAX_LEN = 160


def truncate(text: str, max_len: int = _MAX_LEN) -> str:
    """Collapse newlines and cut long text down to a single readable line."""
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def log_stage(stage: str, **fields: str) -> None:
    """Print one truncated log line for a pipeline stage handoff.

    Example: log_stage("Orchestrator -> notion_expert", input=request)
    """
    parts = " ".join(f"{name}={truncate(str(value))!r}" for name, value in fields.items())
    print(f"[{stage}] {parts}")
