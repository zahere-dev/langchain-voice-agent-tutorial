"""Microphone recording and speaker playback helpers.

Recording and playback are both best-effort: on machines without an audio
device (e.g. some WSL2/CI setups) these functions degrade gracefully so the
rest of the pipeline (STT -> agent -> TTS) can still be exercised via files.
"""
import tempfile
import wave

import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1


def record_audio(seconds: int = 5) -> str:
    """Record from the default microphone for a fixed duration.

    Returns the path to a temporary WAV file containing the recording.
    """
    print(f"Recording for {seconds} seconds... speak now.")
    recording = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    print("Recording finished.")

    tmp_path = tempfile.mktemp(suffix=".wav")
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording.tobytes())
    return tmp_path


def play_audio(path: str) -> None:
    """Play an audio file through the default output device.

    Silently no-ops (with a printed note) if no output device or player
    binary (ffplay/avplay/aplay) is available, so voice mode still works
    for STT/agent testing on machines without audio output (e.g. WSL2).
    """
    try:
        from pydub import AudioSegment
        from pydub.playback import play

        audio = AudioSegment.from_file(path)
        play(audio)
    except Exception as exc:  # noqa: BLE001 - playback is best-effort
        print(f"[audio playback skipped: {exc}]")
