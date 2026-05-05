#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f dist/index.html ]]; then
  corepack pnpm build
fi

exec corepack pnpm desktop:run
