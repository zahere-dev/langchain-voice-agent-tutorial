# The Gmail/Notion MCP servers run as local Node processes, so the runtime
# image needs Node.js alongside Python - a plain "env: python"
# Render/Railway service won't have it available.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install (not execute) the MCP server packages as global binaries, so
# mcp_client.py can launch `gmail-mcp` / `mcp-remote` directly instead of
# through `npx` at request time - `npx` re-checks the registry on every
# invocation even when the package is already installed, and that check
# hanging on a flaky network (seen in practice on Render) makes the
# specialist silently time out with no real error to debug from. Calling
# the installed binary skips that check entirely. Deliberately
# `npm install` rather than running them here - both packages are stdio
# JSON-RPC servers that block waiting for input, and invoking one directly
# during the image build risks hanging the build.
RUN npm install -g @gongrzhe/server-gmail-autoauth-mcp mcp-remote

COPY . .

RUN chmod +x docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
