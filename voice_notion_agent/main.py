"""Entry point: voice or text loop that drives the Gmail/Notion MCP agent."""
import argparse
import asyncio

from langchain.messages import HumanMessage
from openai import BadRequestError

from . import config
from .agent import build_agent
from .audio_io import play_audio, record_audio
from .logging_utils import log_stage
from .stt import transcribe_audio
from .tts import synthesize_speech

_MAX_HISTORY_MESSAGES = 20


async def run_voice_loop(seconds: int, speak: bool) -> None:
    agent = await build_agent()
    messages: list = []
    print("Voice Gmail/Notion Agent ready. Press Ctrl+C to exit.")
    while True:
        try:
            input("\nPress Enter to record a command...")
            audio_path = record_audio(seconds=seconds)
            try:
                transcript = transcribe_audio(audio_path)
            except BadRequestError as exc:
                body = exc.body if isinstance(exc.body, dict) else {}
                detail = body.get("error", {}).get("message", str(exc))
                print(f"Couldn't transcribe that clip ({detail}), try again.")
                continue
            print(f"You said: {transcript}")

            if not transcript:
                print("Heard nothing, try again.")
                continue

            log_stage("User -> Orchestrator", input=transcript)
            messages.append(HumanMessage(content=transcript))
            result = await agent.ainvoke({"messages": messages})
            messages = result["messages"][-_MAX_HISTORY_MESSAGES:]
            response_text = messages[-1].content
            log_stage("Orchestrator -> User", output=response_text)
            print(f"Agent: {response_text}")

            if speak:
                speech_path = synthesize_speech(response_text)
                play_audio(speech_path)
        except KeyboardInterrupt:
            print("\nExiting. Bye!")
            break


async def run_text_loop() -> None:
    agent = await build_agent()
    messages: list = []
    print("Text mode Gmail/Notion Agent ready. Type 'exit' to quit.")
    while True:
        transcript = input("\nYou: ").strip()
        if transcript.lower() in {"exit", "quit"}:
            break
        if not transcript:
            continue

        log_stage("User -> Orchestrator", input=transcript)
        messages.append(HumanMessage(content=transcript))
        result = await agent.ainvoke({"messages": messages})
        messages = result["messages"][-_MAX_HISTORY_MESSAGES:]
        log_stage("Orchestrator -> User", output=messages[-1].content)
        print(f"Agent: {messages[-1].content}")


async def _amain(args: argparse.Namespace) -> None:
    if args.text:
        await run_text_loop()
    else:
        await run_voice_loop(seconds=args.seconds, speak=not args.no_speak)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain Gmail/Notion Voice Agent (MCP)")
    parser.add_argument(
        "--text", action="store_true", help="Run in text-only mode (no microphone/speaker needed)."
    )
    parser.add_argument(
        "--seconds", type=int, default=5, help="Recording duration per turn in voice mode."
    )
    parser.add_argument(
        "--no-speak", action="store_true", help="Disable spoken responses in voice mode."
    )
    args = parser.parse_args()

    config.validate()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
