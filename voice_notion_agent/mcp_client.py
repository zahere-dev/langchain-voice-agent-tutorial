"""MCP server connections for Gmail and Notion.

Both are reached over stdio, via a local Node process. Neither takes
credentials in this config - each authenticates itself using a token
cached on disk by a one-time interactive setup command that you run
yourself before starting the agent:

Gmail (community server, @gongrzhe/server-gmail-autoauth-mcp):
    1. Create a Google Cloud OAuth client (type "Web application",
       redirect URI http://localhost:3000/oauth2callback), download it,
       and save it as ~/.gmail-mcp/gcp-oauth.keys.json.
    2. Run once: npx @gongrzhe/server-gmail-autoauth-mcp auth
       This opens a browser for Google login and caches a token at
       ~/.gmail-mcp/credentials.json.
    3. Install the server as a global binary so it's on PATH:
       npm install -g @gongrzhe/server-gmail-autoauth-mcp

Notion (official hosted MCP server at mcp.notion.com, reached through the
mcp-remote stdio bridge - Notion's older token-based
@notionhq/notion-mcp-server is deprecated and 400s on tool calls, so don't
use it):
    1. Run once: npx -y mcp-remote https://mcp.notion.com/mcp
       This opens a browser for a Notion OAuth consent screen and caches
       a token under ~/.mcp-auth/.
    2. Install the bridge as a global binary so it's on PATH:
       npm install -g mcp-remote

After the one-time steps above, the config below launches each server's
already-installed binary directly (gmail-mcp / mcp-remote) rather than
going through `npx <pkg>`. This matters even when the package is already
installed: `npx` still performs its own registry/update check on every
invocation before running the command, and on a host where that specific
call is slow or blocked (observed on Render, while other outbound HTTPS
traffic - e.g. Notion's connection to mcp.notion.com - worked fine), npx
hangs with no output until something else times it out, which surfaces
as "the Gmail/Notion integration isn't available" with no real error to
debug from. Calling the installed binary directly skips that check
entirely. The Dockerfile already `npm install -g`s both packages so this
works out of the box in the container; for local dev, run the two
`npm install -g` commands above once. See README.md / tutorial.md for the
full walkthrough.
"""
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient(
    {
        "gmail": {
            "transport": "stdio",
            "command": "gmail-mcp",
            "args": [],
        },
        "notion": {
            "transport": "stdio",
            "command": "mcp-remote",
            "args": ["https://mcp.notion.com/mcp"],
        },
    }
)
