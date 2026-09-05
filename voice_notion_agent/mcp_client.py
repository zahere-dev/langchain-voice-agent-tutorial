"""MCP server connections for Gmail and Notion.

Both are reached over stdio, via a local Node process spawned through npx.
Neither takes credentials in this config - each authenticates itself using
a token cached on disk by a one-time interactive setup command that you
run yourself before starting the agent:

Gmail (community server, @gongrzhe/server-gmail-autoauth-mcp):
    1. Create a Google Cloud OAuth client (type "Web application",
       redirect URI http://localhost:3000/oauth2callback), download it,
       and save it as ~/.gmail-mcp/gcp-oauth.keys.json.
    2. Run once: npx @gongrzhe/server-gmail-autoauth-mcp auth
       This opens a browser for Google login and caches a token at
       ~/.gmail-mcp/credentials.json.

Notion (official hosted MCP server at mcp.notion.com, reached through the
mcp-remote stdio bridge - Notion's older token-based
@notionhq/notion-mcp-server is deprecated and 400s on tool calls, so don't
use it):
    1. Run once: npx -y mcp-remote https://mcp.notion.com/mcp
       This opens a browser for a Notion OAuth consent screen and caches
       a token under ~/.mcp-auth/.

After both one-time steps, the config below just launches each server and
lets it pick up its own cached token - nothing app-specific to configure.
See README.md / tutorial.md for the full walkthrough.
"""
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient(
    {
        "gmail": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        },
        "notion": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "mcp-remote", "https://mcp.notion.com/mcp"],
        },
    }
)
