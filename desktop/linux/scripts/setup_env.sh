#!/usr/bin/env bash
# Bootstrap the Build-Optimization dev environment on Linux.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | sh

uv sync
(cd backend && pixi install)
npm install --workspaces

echo "Setup complete. Activate with: source .venv/bin/activate"
