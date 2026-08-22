"""Typing shim for the Hydra YAML directory.

mypy prepends the process cwd to its module path, so this directory
(``middleware/configs/``, no ``__init__.py``) would otherwise be an empty
namespace package named ``configs`` that shadows ``src/configs`` and makes
``from configs import Config`` fail with ``attr-defined``.

The runtime package is ``src/configs`` (put on ``sys.path`` by the uv
install). This stub exists only so type-checkers see ``Config``.
"""

from typing import Any

class Config:
    game: Any
    pipeline: Any
    optimization: Any
    train: Any
    rl: Any
    tracking: Any
    model: Any
    hpo: Any
    seed: int
    device: str
    output_dir: str

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
