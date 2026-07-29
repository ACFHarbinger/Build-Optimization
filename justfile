# Build-Optimization — Justfile

set unstable := true

red := '\033[0;31m'
green := '\033[0;32m'
yellow := '\033[0;33m'
blue := '\033[0;34m'
purple := '\033[0;35m'
cyan := '\033[0;36m'
bold := '\033[1m'
reset := '\033[0m'

# Default variables (can be overridden: just train game=darktide model=am)

game := "darktide"
model := "am"
encoder := "gat"
decoder := "transformer"
epochs := "100"
batch_size := "256"
samples := "1"
seed := "42"
strategy := "greedy"
distribution := "default"
n_cores := "10"
policies := "sa,ga,alns"

# --- Submodules ---

mod agent 'tools/agent'
mod app 'tools/app'
mod benchmark 'tools/benchmark'
mod ci 'tools/ci'
mod controller 'tools/controller'
mod database 'tools/database'
mod docs 'tools/docs'
mod export 'tools/export'
mod helper 'tools/helper'
mod infrastructure 'tools/infrastructure'
mod reducer 'tools/reducer'
mod script 'tools/script'
mod test 'tools/test'
mod validation 'tools/validation'

# --- Help ---

# Print available commands with descriptions
help: helper::_print_header
    just helper::help

# --- Shorthands ---

# Initialize environment and install all dependencies
setup: helper::_print_header
    just infrastructure::setup

# Sync dependencies using uv
sync: helper::_print_header
    just infrastructure::sync

# Install dependencies via pip
install: helper::_print_header
    just infrastructure::install

# Train a model with Hydra configs (game=darktide model=am)
train game=game model=model epochs=epochs encoder=encoder decoder=decoder batch_size=batch_size distribution=distribution: helper::_print_header
    just controller::train '{{ game }}' '{{ model }}' '{{ epochs }}' '{{ encoder }}' '{{ decoder }}' '{{ batch_size }}' '{{ distribution }}'

# Run model evaluation with Hydra configs
eval model_path="" dataset="" game=game strategy=strategy: helper::_print_header
    just controller::eval '{{ model_path }}' '{{ dataset }}' '{{ game }}' '{{ strategy }}'

# Run build optimization across the defined policies for a given game
optimize game=game policies=policies budget="" level="" time="": helper::_print_header
    just controller::optimize '{{ game }}' '{{ policies }}' '{{ budget }}' '{{ level }}' '{{ time }}'

# Launch the GUI
gui: helper::_print_header
    just app::gui

# Launch the Tauri Studio desktop app (dev mode)
studio: helper::_print_header
    just app::studio

# Build the full multi-language documentation site
docs: helper::_print_header
    just docs::build

# Run fast unit tests (use `just test::test` for the full suite)
test-fast: helper::_print_header
    just test::test-fast

# Check code quality with ruff
lint: helper::_print_header
    just ci::lint

# Format code with ruff
format: helper::_print_header
    just ci::format

# Clean caches and build artifacts
clean: helper::_print_header
    just reducer::clean

# Generic run command — pass any main.py arguments directly
run *args: helper::_print_header
    just helper::run {{ args }}

# Commit using the .gitmessage template
commit message: helper::_print_header
    just helper::commit '{{ message }}'

# Loop the Claude Code agent on a stateful context with live-streaming reasoning steps
loop-claude prompt="Continue implementing the studio, updating the ROADMAP and CHANGELOG, and commiting your work": helper::_print_header
    just agent::loop-claude '{{ prompt }}'

# Loop the Grok agent on a stateful context with live-streaming reasoning steps
loop-grok prompt="Continue implementing the studio, updating the ROADMAP and CHANGELOG, and commiting your work": helper::_print_header
    just agent::loop-grok '{{ prompt }}'
