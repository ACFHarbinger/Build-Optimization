#!/usr/bin/env bash
# Build the complete Build-Optimization documentation site.
#
# Layers:
#   1. Per-module API references (Sphinx/autoapi, Doxygen, TypeDoc x2, rustdoc)
#   2. MkDocs narrative site (docs/mkdocs.yml)
#   3. Copy API references into the site under site/api/<module>/
#
# Tolerant by design: modules not yet scaffolded, or tools missing from the
# environment, are skipped with a notice so the docs build at every roadmap
# stage. Run from anywhere; paths resolve to the repo root.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/docs/_build"
API="$BUILD/api"
SITE="$ROOT/site"
mkdir -p "$API"

note()  { printf '\033[1;34m[docs]\033[0m %s\n' "$*"; }
skip()  { printf '\033[1;33m[skip]\033[0m %s\n' "$*"; }
fail=0

# ---------------------------------------------------------------- Python (Sphinx/autoapi)
if command -v uv >/dev/null && (cd "$ROOT" && uv run --no-sync python -c 'import sphinx' 2>/dev/null); then
  note "Sphinx: building Python middleware API"
  (cd "$ROOT" && uv run --no-sync sphinx-build -q -b html docs/sphinx "$API/python") || fail=1
else
  skip "Sphinx not available — run: uv sync --extra docs"
fi

# ---------------------------------------------------------------- C++ backend (Doxygen)
if [ ! -d "$ROOT/backend/src" ]; then
  skip "Doxygen (cpp): backend/src not scaffolded yet"
elif command -v doxygen >/dev/null; then
  note "Doxygen: building C++ backend API"
  (cd "$ROOT" && doxygen docs/cpp/Doxyfile >/dev/null) || fail=1
  rm -rf "$API/cpp"
  mv "$API/cpp-doxygen/html" "$API/cpp" 2>/dev/null && rm -rf "$API/cpp-doxygen"
elif command -v pixi >/dev/null && (cd "$ROOT/backend" && pixi run -q doxygen ../docs/cpp/Doxyfile >/dev/null 2>&1); then
  note "Doxygen (via pixi): building C++ backend API"
  rm -rf "$API/cpp"
  mv "$API/cpp-doxygen/html" "$API/cpp" 2>/dev/null && rm -rf "$API/cpp-doxygen"
else
  skip "doxygen not found — install it, or run: cd backend && pixi install"
fi

# ---------------------------------------------------------------- TypeScript (TypeDoc)
run_typedoc() { # $1=workspace  $2=config
  local ws="$1" cfg="$2"
  if [ ! -d "$ROOT/$ws/src" ] || [ ! -f "$ROOT/$ws/tsconfig.json" ]; then
    skip "TypeDoc ($ws): src/ or tsconfig.json not scaffolded yet"
    return
  fi
  if [ ! -d "$ROOT/node_modules" ]; then
    skip "TypeDoc ($ws): node_modules missing — run: npm install"
    return
  fi
  note "TypeDoc: building $ws API"
  (cd "$ROOT" && npx typedoc --options "$cfg") || fail=1
}
run_typedoc frontend  docs/frontend/typedoc.json
run_typedoc extension docs/extension/typedoc.json

# ---------------------------------------------------------------- Rust (rustdoc)
if command -v cargo >/dev/null; then
  note "rustdoc: building Tauri Rust shell API"
  if (cd "$ROOT" && cargo doc --no-deps -p app -q); then
    rm -rf "$API/rust"
    cp -r "$ROOT/target/doc" "$API/rust"
  else
    fail=1
  fi
else
  skip "cargo not found — install the Rust toolchain"
fi

# ---------------------------------------------------------------- MkDocs umbrella
if command -v uv >/dev/null && (cd "$ROOT" && uv run --no-sync python -c 'import mkdocs' 2>/dev/null); then
  note "MkDocs: building the narrative site"
  (cd "$ROOT" && uv run --no-sync mkdocs build -q -f docs/mkdocs.yml) || fail=1
  if [ -d "$SITE" ]; then
    note "Copying API references into site/api/"
    mkdir -p "$SITE/api"
    for d in "$API"/*/; do
      [ -d "$d" ] && cp -r "$d" "$SITE/api/$(basename "$d")"
    done
    note "Done → open $SITE/index.html"
  fi
else
  skip "MkDocs not available — run: uv sync --extra docs"
fi

exit $fail
