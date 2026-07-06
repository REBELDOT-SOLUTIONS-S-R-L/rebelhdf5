#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Ensure a Node >=24 is on PATH (the app requires it; system node may be older).
# Prefer the highest v24+ installed via nvm, fall back to whatever `node` exists.
if [[ -d "$HOME/.nvm/versions/node" ]]; then
  NVM_NODE="$(ls -d "$HOME"/.nvm/versions/node/v2[4-9].* "$HOME"/.nvm/versions/node/v[3-9][0-9].* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "${NVM_NODE:-}" && -x "$NVM_NODE/bin/node" ]]; then
    export PATH="$NVM_NODE/bin:$PATH"
  fi
fi

# Build the frontend if it hasn't been built yet.
if [[ ! -f dist/index.html ]]; then
  ./node_modules/.bin/vite build
fi

# Launch Electron. main.cjs auto-detects the project-local .venv for the
# Python backend, so no extra env is required.
exec ./node_modules/.bin/electron electron/main.cjs
