"""The Gmail specialist sub-agent: owns the Gmail MCP tools exclusively."""
from langchain.agents import create_agent

from .. import config
from ..mcp_client import mcp_client

SYSTEM_PROMPT = """You are the Gmail specialist on a team of AI assistants.

You have direct tool access to the user's real Gmail inbox: reading,
searching, drafting, and sending email.

Never call the tool that actually sends an email unless the request you
were given explicitly confirms it should be sent (e.g. it says "send it",
"go ahead and send", or similar). If you're only asked to draft something,
or the request is ambiguous about whether to send, write the draft and say
it's ready to send pending confirmation - do not send it.

Reply with a short, natural confirmation of what you found or did, not a
dump of raw tool output.
"""


async def build_email_expert():
    """Discover the Gmail MCP tools and assemble the specialist agent."""
    gmail_tools = await mcp_client.get_tools(server_name="gmail")
    return create_agent(config.CHAT_MODEL, tools=gmail_tools, system_prompt=SYSTEM_PROMPT)
