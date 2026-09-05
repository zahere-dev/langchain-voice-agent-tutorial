"""Builds the Orchestrator: a generalist agent that answers directly or
delegates to a specialist sub-agent (Notion, Gmail, or web research).

    Orchestrator -> notion_expert  (Notion MCP tools)
                 -> email_expert   (Gmail MCP tools)
                 -> researcher     (web search + summarize)

Each specialist is itself a full create_agent, wrapped as a plain tool so
the Orchestrator can call it like any other tool. This keeps each
specialist's tool access scoped to just the service it owns, and keeps
their system prompts focused on one job instead of one long prompt trying
to cover Gmail, Notion, and research at once.
"""
import asyncio

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool

from . import config
from .agents.email_expert import build_email_expert
from .agents.notion_expert import build_notion_expert
from .agents.researcher import build_researcher
from .logging_utils import log_stage

ORCHESTRATOR_SYSTEM_PROMPT = """You are Aria, a voice-controlled executive assistant.

Answer general questions directly yourself. For anything that needs real
access to the user's Notion workspace, Gmail inbox, or current web
information, delegate to the matching specialist tool instead of guessing:
- notion_expert: search, read, or create Notion pages.
- email_expert: check, search, draft, or send Gmail.
- researcher: look something up on the web and summarize it.

Chain specialists when a request needs more than one step - e.g. ask
researcher for a summary, then pass that summary to notion_expert to save
as a page.

Always give a short, spoken-friendly reply (1-2 sentences), since it may
be read aloud. Report the outcome naturally - never mention tools, agents,
or that you "delegated" something.
"""


async def build_agent():
    """Build the three specialists, wrap them as tools, and assemble the Orchestrator."""
    notion_agent, email_agent, research_agent = await asyncio.gather(
        build_notion_expert(), build_email_expert(), build_researcher()
    )

    @tool
    async def notion_expert(request: str) -> str:
        """Delegate a Notion request (search, read, or create pages) to the Notion specialist."""
        log_stage("Orchestrator -> notion_expert", input=request)
        result = await notion_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        output = result["messages"][-1].content
        log_stage("notion_expert -> Orchestrator", output=output)
        return output

    @tool
    async def email_expert(request: str) -> str:
        """Delegate a Gmail request (check, search, draft, or send email) to the email specialist."""
        log_stage("Orchestrator -> email_expert", input=request)
        result = await email_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        output = result["messages"][-1].content
        log_stage("email_expert -> Orchestrator", output=output)
        return output

    @tool
    async def researcher(topic: str) -> str:
        """Delegate a web research request to the research specialist; returns a markdown summary."""
        log_stage("Orchestrator -> researcher", input=topic)
        result = await research_agent.ainvoke({"messages": [HumanMessage(content=topic)]})
        output = result["messages"][-1].content
        log_stage("researcher -> Orchestrator", output=output)
        return output

    tools = [notion_expert, email_expert, researcher]
    return create_agent(config.CHAT_MODEL, tools=tools, system_prompt=ORCHESTRATOR_SYSTEM_PROMPT)
