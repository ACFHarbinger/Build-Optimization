#!/usr/bin/env bash
# Bootstrap the Build-Optimization dev environment on macOS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

command -v brew >/dev/null || { echo "Homebrew is required: https://brew.sh"; exit 1; }
command -v uv >/dev/null || brew install uv
command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | sh
command -v node >/dev/null || brew install node

uv sync
(cd backend && pixi install)
npm install --workspaces

echo "Setup complete. Activate with: source .venv/bin/activate"
