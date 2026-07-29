#!/usr/bin/env bash
# Launch the Tauri Studio in dev mode on Linux.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../app"

npm run tauri:dev
