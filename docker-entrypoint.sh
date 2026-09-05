#!/bin/sh
# Restores the Gmail/Notion MCP auth directories from base64-encoded env
# vars before starting the server. Render's filesystem is ephemeral, so
# the one-time OAuth tokens cached locally at ~/.gmail-mcp and ~/.mcp-auth
# don't exist in a fresh container - this puts them back on every boot.
#
# Generate the env var values by running scripts/pack_mcp_tokens.py on the
# machine where you completed the one-time Gmail/Notion logins.
set -e

if [ -n "$GMAIL_MCP_TAR_B64" ]; then
  echo "$GMAIL_MCP_TAR_B64" | base64 -d | tar -xzf - -C "$HOME"
  echo "[entrypoint] restored $HOME/.gmail-mcp from GMAIL_MCP_TAR_B64"
fi

if [ -n "$MCP_AUTH_TAR_B64" ]; then
  echo "$MCP_AUTH_TAR_B64" | base64 -d | tar -xzf - -C "$HOME"
  echo "[entrypoint] restored $HOME/.mcp-auth from MCP_AUTH_TAR_B64"
fi

exec uvicorn webapp.server:app --host 0.0.0.0 --port "${PORT:-8000}"
