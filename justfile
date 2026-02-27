# Build-Optimization Justfile

red := '\033[0;31m'
green := '\033[0;32m'
yellow := '\033[0;33m'
blue := '\033[0;34m'
purple := '\033[0;35m'
cyan := '\033[0;36m'
bold := '\033[1m'
reset := '\033[0m'

# Default variables (can be overridden: just train problem=wcvrp)

game := "default"
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

# --- Setup & Environment ---

# Sync dependencies using uv
sync:
    uv sync --all-groups --all-extras

# Install dependencies via pip
install:
    uv pip install -r requirements.txt || uv pip install -e .

# --- Primary Execution Commands (Hydra-based) ---
# Train a model with Hydra configs

# Usage: just train game=my_game model=am epochs=100
train game=game model=model epochs=epochs encoder=encoder decoder=decoder batch_size=batch_size distribution=distribution:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "🚀 STARTING HYDRA TRAINING SESSION"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Game:" "{{ game }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Model:" "{{ model }} ({{ encoder }})"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Epochs:" "{{ epochs }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Batch Size:" "{{ batch_size }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"

    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" && \
    uv run python main.py train \
        game={{ game }} \
        models={{ model }} \
        model.encoder.type={{ encoder }} \
        train.data_distribution={{ distribution }} \
        train.n_epochs={{ epochs }} \
        train.batch_size={{ batch_size }}

# Run model evaluation with Hydra configs

# Usage: just eval model_path=./weights/best.pt dataset=data/test.pkl game=my_game strategy=greedy
eval model_path="" dataset="" game=game strategy=strategy:
    @printf "{{ cyan }}╔════════════════════════════════════════════════════════════╗{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ bold }}%-58s{{ reset }}   {{ cyan }}║{{ reset }}\n" "📊 STARTING MODEL EVALUATION"
    @printf "{{ cyan }}╠════════════════════════════════════════════════════════════╣{{ reset }}\n"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Model Path:" "{{ model_path }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Dataset:" "{{ dataset }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Game:" "{{ game }}"
    @printf "{{ cyan }}║{{ reset }} {{ yellow }}%-15s{{ reset }} {{ purple }}%-42s{{ reset }} {{ cyan }}║{{ reset }}\n" "Strategy:" "{{ strategy }}"
    @printf "{{ cyan }}╚════════════════════════════════════════════════════════════╝{{ reset }}\n"
    uv run python main.py eval \
        eval.policy.model.load_path={{ model_path }} \
        eval.datasets=[{{ dataset }}] \
        eval.game={{ game }} \
        eval.val_size={{ samples }} \
        eval.decoding.strategy={{ strategy }}

# Launch the GUI
gui:
    uv run python main.py gui

# Launch the dashboard
dashboard:
    uv run streamlit run src/ui/app.py

# --- Test & Quality ---

# Run all tests
test:
    uv run pytest --cov=src --cov-report=xml --cov-report=term-missing

# Run fast unit tests
test-fast:
    uv run pytest -m "fast or unit"

# Check code quality with ruff
lint:
    uv run ruff check . --fix --exclude ".venv"

# Format code with black and ruff
format:
    uv run ruff format . --exclude ".venv"

# --- Maintenance ---

# Clean caches and artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".ruff_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type d -name ".hypothesis" -exec rm -rf {} +
    find . -type f -name "coverage.xml" -exec rm {} +
    find . -type f -name ".coverage" -exec rm {} +
    rm -rf build/
    rm -rf dist/
    rm -rf temp/
    rm -rf wandb/
    rm -rf mlruns/
    rm -rf outputs/
    rm -rf checkpoints/
    rm -rf *.egg-info
    rm -rf logs/
    rm -rf model_weights/

# Generic run command
run *args:
    uv run python main.py {{ args }}
