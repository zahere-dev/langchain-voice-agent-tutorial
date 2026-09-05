# The Gmail/Notion MCP servers run as local Node processes launched via
# npx, so the runtime image needs Node.js alongside Python - a plain
# "env: python" Render/Railway service won't have npx available.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates ffmpeg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install (not execute) the MCP server packages into the image, so a
# cold start doesn't have to hit the npm registry, and `npx` at runtime
# resolves them locally instead of downloading. Deliberately `npm install`
# rather than running them via npx here - both packages are stdio JSON-RPC
# servers that block waiting for input, and invoking one directly during
# the image build risks hanging the build.
RUN npm install -g @gongrzhe/server-gmail-autoauth-mcp mcp-remote

COPY . .

RUN chmod +x docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
