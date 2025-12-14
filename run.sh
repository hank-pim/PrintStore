#!/usr/bin/env bash
set -euo pipefail

# Run PrintStore viewer on Linux/macOS.
# Uses the workspace virtual environment if present.
# You can pass an optional folder path (including SMB mounts) or flags like --debug.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "ERROR: python3 not found." >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/app.py" "$@"
