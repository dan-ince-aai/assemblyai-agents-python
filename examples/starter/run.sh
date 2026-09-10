#!/usr/bin/env bash
# Backend, tunnel and deploy, in one command.
#
#   ./run.sh          start everything and deploy
#   ./run.sh --stop   stop the backend and the tunnel
#
# The tunnel address changes every time it comes up, so re-run this after a
# restart: it updates the same agent in place, and an attached phone number
# keeps working because it points at an agent id rather than a URL.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PY="${PY:-../../.venv/bin/python}"
RUN_DIR=".run"
mkdir -p "$RUN_DIR"

if [ "${1:-}" = "--stop" ]; then
  for name in backend tunnel; do
    [ -f "$RUN_DIR/$name.pid" ] && kill "$(cat "$RUN_DIR/$name.pid")" 2>/dev/null && echo "stopped $name" || true
    rm -f "$RUN_DIR/$name.pid"
  done
  exit 0
fi

: "${ASSEMBLYAI_API_KEY:?set ASSEMBLYAI_API_KEY}"
export TOOL_SECRET="${TOOL_SECRET:-starter-tool-secret}"
export LLM_API_KEY="${LLM_API_KEY:-starter-llm-key}"
export BYO_LLM=1
export AGENT_NAME="${AGENT_NAME:-Sam}"

# How this machine becomes reachable is your call. Set PUBLIC_BASE_URL to a
# staging host or to a tunnel you already run, and no tunnel is started here.
PUBLIC_BASE_URL="https://placeholder.invalid" PORT="$PORT" $PY backend.py > "$RUN_DIR/backend.log" 2>&1 &
echo $! > "$RUN_DIR/backend.pid"

if [ -n "${PUBLIC_BASE_URL:-}" ]; then
  URL="${PUBLIC_BASE_URL%/}"
  echo "using PUBLIC_BASE_URL: $URL"
else
  command -v ngrok >/dev/null || {
    echo "ngrok is not on PATH. Either install it, or set PUBLIC_BASE_URL to an"
    echo "address that reaches port $PORT and run this again."
    exit 1
  }
  ngrok http "$PORT" --log stdout --log-format json > "$RUN_DIR/tunnel.log" 2>&1 &
  echo $! > "$RUN_DIR/tunnel.pid"
  echo -n "waiting for the tunnel"
  for _ in $(seq 1 40); do
    URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null \
      | $PY -c 'import sys,json;print(next((t["public_url"] for t in json.load(sys.stdin).get("tunnels",[]) if t["public_url"].startswith("https")),""))' 2>/dev/null || true)
    [ -n "${URL:-}" ] && break
    echo -n "."; sleep 1
  done
  echo
  [ -n "${URL:-}" ] || { echo "the tunnel never came up; see $RUN_DIR/tunnel.log"; exit 1; }
  echo "tunnel: $URL"
fi
export PUBLIC_BASE_URL="$URL"

# Restart the backend now it can be told its own public address, so its health
# endpoint and the declaration it imports match what gets deployed.
kill "$(cat "$RUN_DIR/backend.pid")" 2>/dev/null || true
sleep 1
PORT="$PORT" $PY backend.py > "$RUN_DIR/backend.log" 2>&1 &
echo $! > "$RUN_DIR/backend.pid"
for _ in $(seq 1 20); do curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && break; sleep 1; done

$PY deploy.py
echo
echo "backend log: $RUN_DIR/backend.log   (tail -f it while you call)"
echo "next: $PY drive.py happy"
