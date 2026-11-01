#!/usr/bin/env bash
# Start the backend and the frontend together for development.
#
#   scripts/dev.sh            # both, Ctrl-C to stop both
#   scripts/dev.sh backend    # backend only
#   scripts/dev.sh frontend   # frontend only
#
# The frontend proxies /api and /ws to the backend, so the browser sees a single
# origin and no CORS preflight is needed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# Prefer the project venv if it exists, else whatever python3 is on PATH.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

start_backend() {
  echo "[dev] backend  http://${BACKEND_HOST}:${BACKEND_PORT}"
  "$PYTHON" -m uvicorn mem20_brainz.server.app:app \
    --host "$BACKEND_HOST" --port "$BACKEND_PORT" --log-level info
}

start_frontend() {
  if [[ ! -d "$ROOT/web/node_modules" ]]; then
    echo "[dev] installing frontend dependencies (first run)…"
    (cd "$ROOT/web" && npm install --no-audit --no-fund)
  fi
  echo "[dev] frontend http://127.0.0.1:${FRONTEND_PORT}"
  (cd "$ROOT/web" && npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT")
}

case "${1:-both}" in
  backend)
    start_backend
    ;;
  frontend)
    start_frontend
    ;;
  both)
    start_backend &
    BACKEND_PID=$!
    # Stop the backend when this script exits, however it exits.
    trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM
    # Give uvicorn a moment to bind before Vite starts proxying to it.
    sleep 2
    start_frontend
    ;;
  *)
    echo "usage: $0 [both|backend|frontend]" >&2
    exit 2
    ;;
esac
