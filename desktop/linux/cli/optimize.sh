#!/usr/bin/env bash
# Thin CLI wrapper: run an optimization from a terminal / .desktop launcher.
# Usage: ./optimize.sh [game=rpg] [policy=policy_sa] [extra hydra overrides...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

uv run python main.py "$@"
