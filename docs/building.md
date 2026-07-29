# Building the Docs

The full documentation site combines five per-language API generators with an
MkDocs narrative shell. Every step is tolerant — if a tool isn't installed or
a module isn't scaffolded yet, that step is skipped with a notice rather than
failing the whole build.

```bash
bash docs/build_docs.sh
```

This produces `site/` (the MkDocs portal) with each generated API reference
copied into `site/api/<module>/`.

## Individual generators

| Language | Tool | Config | Output |
| -------- | ---- | ------ | ------ |
| Python (`middleware/src`) | [sphinx-autoapi](https://sphinx-autoapi.readthedocs.io/) | `docs/sphinx/conf.py` | `docs/_build/api/python` |
| C++ (`backend/`) | [Doxygen](https://www.doxygen.nl/) | `docs/cpp/Doxyfile` | `docs/_build/api/cpp` |
| TypeScript (`frontend/src`) | [TypeDoc](https://typedoc.org/) | `docs/frontend/typedoc.json` | `docs/_build/api/frontend` |
| TypeScript (`extension/src`) | [TypeDoc](https://typedoc.org/) | `docs/extension/typedoc.json` | `docs/_build/api/extension` |
| Rust (`frontend/src-tauri`) | [rustdoc](https://doc.rust-lang.org/rustdoc/) (`cargo doc`) | `frontend/src-tauri/Cargo.toml` | `docs/_build/api/rust` |

## Serving just the narrative site

While editing prose (this file, `index.md`, `ARCHITECTURE.md`, ...), skip the
API generators and serve MkDocs directly:

```bash
uv sync --extra docs
uv run mkdocs serve -f docs/mkdocs.yml
```

## Prerequisites

```bash
uv sync --extra docs          # mkdocs-material, sphinx, sphinx-autoapi, furo
sudo apt install doxygen      # or: brew install doxygen
npm install                   # TypeDoc ships as a devDependency per workspace
```

Rust's `cargo doc` uses whatever Rust toolchain `just setup` already installed.
