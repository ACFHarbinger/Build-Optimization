"""
Validation utilities module.

Provides static-analysis scripts for code quality and codebase health checks.
Scripts are run via the ``tools/validation/justfile`` recipes.

Attributes:
    check_circular_imports.py   — Detect circular import chains (Tarjan's SCC)
    check_embedded_languages.py — Find embedded non-Python languages in source
    check_interface_compliance.py — Verify ABC/Protocol implementation contracts
    check_multi_classes.py      — Detect multiple top-level classes per file
    check_nested_imports.py     — Find function-level / nested imports
    check_relative_imports.py   — Audit relative import usage
    check_type_coverage.py      — Measure per-file annotation coverage
    check_unused_imports.py     — Detect unused import statements
    count_loc.py                — Count lines of code and comments
    debug_utils.py              — Lightweight debugging helpers
    trace_dependencies.py       — Trace function / class dependency graphs
    tree_loc.py                 — Tree-view LoC display
    visualize_module_graph.py   — Interactive module-level import graph

Example:
    just check-circular-imports
    just check-relative-imports
    just check-type-coverage sort=coverage limit=40
"""
