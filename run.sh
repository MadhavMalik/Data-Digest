#!/usr/bin/env bash
# Signal Engine — one command to build and launch everything.
#
#   ./run.sh                 build the UI if needed, then serve on :8000
#   ./run.sh --port 9000     serve on another port
#   ./run.sh --host 0.0.0.0  bind publicly (what you want on EC2)
#   ./run.sh --rebuild       force a UI rebuild
#   ./run.sh --no-ui         API only, skip the UI build
#
# Safe to re-run: it creates the venv and installs dependencies only when they
# are missing, so a second run starts in about a second.

set -euo pipefail
cd "$(dirname "$0")"

HOST="127.0.0.1"
PORT="8000"
REBUILD=0
BUILD_UI=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --rebuild) REBUILD=1; shift ;;
    --no-ui) BUILD_UI=0; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[36m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }

# ---- python ---------------------------------------------------------------
if [[ ! -d .venv ]]; then
  say "creating virtualenv..."
  python3 -m venv .venv
fi
PY=".venv/bin/python"

if ! "$PY" -c "import signal_engine" 2>/dev/null; then
  say "installing python dependencies..."
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -e ".[elastic,dev]"
  "$PY" -m pip install -q python-multipart aiohttp openpyxl pyshp
fi

# ---- ui -------------------------------------------------------------------
if [[ "$BUILD_UI" == "1" ]]; then
  if command -v npm >/dev/null 2>&1; then
    if [[ ! -d web/node_modules ]]; then
      say "installing ui dependencies..."
      (cd web && npm install --silent)
    fi
    if [[ "$REBUILD" == "1" || ! -f ui/dist/index.html ]]; then
      say "building ui..."
      (cd web && npm run build)
    fi
  else
    warn "npm not found — serving the single-file fallback UI instead"
  fi
fi

# ---- data -----------------------------------------------------------------
if ! ls data/raw/*.parquet >/dev/null 2>&1; then
  say "no dataset found; fetching NYC TLC yellow taxi 2026-01..."
  "$PY" scripts/fetch_tlc.py --vehicle yellow --year 2026 --month 1 --quiet || \
    warn "dataset fetch failed — upload one through the UI instead"
fi

# ---- credentials ----------------------------------------------------------
if [[ -f .env ]]; then
  set -a; . ./.env; set +a
else
  warn "no .env found — the engine will run on its deterministic path."
  warn "copy .env.example to .env and add credentials for the full loop."
fi

say ""
say "  Signal Engine  ->  http://${HOST}:${PORT}"
say ""

exec "$PY" -m uvicorn signal_engine.api.app:app --host "$HOST" --port "$PORT"
