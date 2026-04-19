"""
Test suite settings and constants.

This module defines the test module registry for the WSmart+ Route test suite.
Used by:
- logic/test/test_suite.py (pytest wrapper for selective test execution)
- main.py (CLI command: python main.py test_suite --module <name>)
- CI/CD pipelines (.github/workflows/ci.yml)

Test Module Organization
------------------------
Tests are organized into 3 categories:

1. **CLI Command Tests**: Validate main.py entry points (train, eval, gen_data, etc.)
   - Fast smoke tests (do not train full models)
   - Check argument parsing, config loading, pipeline orchestration
   - Each command has dedicated test file

2. **Component Tests**: Unit tests for subsystems (models, actions, scheduler, optimizer)
   - Test individual classes and functions in isolation
   - Mock external dependencies (environments, datasets, GPU)
   - Fast execution (<1s per test)

3. **Integration Tests**: End-to-end workflows
   - Test complete pipelines (data gen → train → eval)
   - Use small instances (10-20 nodes, 2-3 epochs)
   - Slow but comprehensive (30s-2min per test)

Usage Examples
--------------
    # Run all tests
    python main.py test_suite

    # Run specific module
    python main.py test_suite --module train

    # Run multiple modules (space-separated)
    python main.py test_suite --module train eval integration

Module Selection
----------------
Use the TEST_MODULES keys as --module argument values. Example:
    python main.py test_suite --module parser actions edge_cases
"""

from typing import Dict

# Test suite module registry
# Maps module selector name → test file name (relative to logic/test/)
# Selector names are used in CLI: python main.py test_suite --module <selector>
TEST_MODULES: Dict[str, str] = {
    # CLI Command Tests (smoke tests for main.py entry points)
    "parser": "test_configs_parser.py",  # Hydra config parsing & composition
    "train": "test_train_command.py",  # Training pipeline (RL, imitation)
    "mrl": "test_mrl_train_command.py",  # Meta-RL training pipeline
    "hp_optim": "test_hp_optim_command.py",  # Hyperparameter optimization (Optuna, DEHB)
    "gen_data": "test_gen_data_command.py",  # Dataset generation (train/val/test)
    "eval": "test_eval_command.py",  # Model evaluation (greedy, sampling, beam search)
    "test_sim": "test_test_command.py",  # Multi-day simulation testing
    "file_system": "test_file_system_command.py",  # File operations (create, read, update, delete)
    "gui": "test_gui_command.py",  # GUI launch and initialization
    # Component Tests (unit tests for subsystems)
    "actions": "test_custom_actions.py",  # Simulation action pattern (Fill, Collect, Log)
    "edge_cases": "test_edge_cases.py",  # Boundary conditions (empty instances, single node, etc.)
    "layers": "test_model_layers.py",  # Neural network layers (attention, encoders, decoders)
    "scheduler": "test_lr_scheduler.py",  # Learning rate scheduling (cosine, linear, exp decay)
    "optimizer": "test_optimizer.py",  # Optimizer configuration (Adam, AdamW, SGD)
    # Integration Tests (end-to-end workflows)
    "integration": "test_integration.py",  # Full pipeline tests (data → train → eval → sim)
}
