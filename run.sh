#!/usr/bin/env bash
# =============================================================================
#  run.sh  -  Single-Command Launcher for CUDA Power Flow Simulation Lab
#
#  Usage:
#    ./run.sh              # Starts server and automatically opens browser
#    ./run.sh --no-open    # Starts server without automatically opening browser
#    ./run.sh 8080         # Starts server on custom port (e.g. 8080)
# =============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Check Node.js
if ! command -v node >/dev/null 2>&1; then
  echo "Error: Node.js is required to run the Cloud Compute Gateway & Web Dashboard."
  echo "Please install Node.js (https://nodejs.org) or run with nvm."
  exit 1
fi

PORT="${PORT:-3000}"
OPEN_BROWSER="1"

# Parse arguments
for arg in "$@"; do
  if [ "$arg" = "--no-open" ]; then
    OPEN_BROWSER="0"
  elif [[ "$arg" =~ ^[0-9]+$ ]]; then
    PORT="$arg"
  fi
done

export PORT
export AUTO_OPEN="$OPEN_BROWSER"

echo "================================================================"
echo "  CUDA SMART GRID POWER FLOW SIMULATION LAB"
echo "  Client/Server Architecture: Browser -> Cloud GPU -> CUDA Solver"
echo "  Launching compute gateway on http://localhost:${PORT}..."
echo "================================================================"

exec node web/server/index.js "$@"
