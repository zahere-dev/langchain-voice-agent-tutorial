"""The Notion specialist sub-agent: owns the Notion MCP tools exclusively."""
from langchain.agents import create_agent

from .. import config
from ..mcp_client import mcp_client

SYSTEM_PROMPT = """You are the Notion specialist on a team of AI assistants.

You have direct tool access to the user's Notion workspace: searching,
reading, and creating pages. Handle whatever Notion request you're given
using those tools.

Reply with a short, natural confirmation of what you found or did - not a
dump of raw tool output or a page URL, unless the caller specifically asks
for a link.
"""


async def build_notion_expert():
    """Discover the Notion MCP tools and assemble the specialist agent."""
    notion_tools = await mcp_client.get_tools(server_name="notion")
    return create_agent(config.CHAT_MODEL, tools=notion_tools, system_prompt=SYSTEM_PROMPT)
