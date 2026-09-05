#!/usr/bin/env python3
"""Pack the local Gmail/Notion MCP auth directories into base64 blobs.

Run this on the machine where you completed the one-time Gmail
(`npx @gongrzhe/server-gmail-autoauth-mcp auth`) and Notion
(`npx -y mcp-remote https://mcp.notion.com/mcp`) logins. It prints two
values to paste into your Render service's environment variables as
GMAIL_MCP_TAR_B64 and MCP_AUTH_TAR_B64.

These blobs are your real Gmail/Notion session credentials, equivalent to
a password. Paste them only into Render's env var fields (never into a
file you commit, a chat, or anywhere else) and treat them with the same
care as an API key.
"""
import base64
import io
import tarfile
from pathlib import Path


def pack(path: Path) -> str | None:
    if not path.exists():
        return None
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(path, arcname=path.name)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    home = Path.home()
    targets = [
        ("GMAIL_MCP_TAR_B64", home / ".gmail-mcp"),
        ("MCP_AUTH_TAR_B64", home / ".mcp-auth"),
    ]

    for env_name, dir_path in targets:
        encoded = pack(dir_path)
        if encoded is None:
            print(f"# {dir_path} not found - skipping {env_name}. "
                  f"Complete the one-time login for this service first.")
            continue
        print(f"# {env_name} (paste this whole value into Render):")
        print(encoded)
        print()


if __name__ == "__main__":
    main()
