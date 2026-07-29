#!/usr/bin/env bash
# Build the C++ backend pybind11 extension on Linux.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../backend"

pixi run build
pixi run install
