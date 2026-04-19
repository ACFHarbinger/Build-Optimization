"""
Detect circular import chains in a Python codebase using static AST analysis.
Applies an iterative Tarjan's SCC algorithm on the module-level import graph
and reports all cycles. Optionally generates an interactive HTML visualization.
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "dist", "build"}

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

try:
    from pyvis.network import Network
except ImportError:
    print(f"{YELLOW}pyvis not installed — skipping HTML in generate_html. Install with: uv add pyvis{RESET}")
    Network = None


def file_to_module(filepath: Path, root: Path) -> str:
    rel = filepath.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def collect_module_map(root: Path, exclude: Set[str]) -> Dict[str, Path]:
    module_map: Dict[str, Path] = {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d not in exclude]
        for fname in files:
            if fname.endswith(".py"):
                fpath = Path(dirpath) / fname
                module_map[file_to_module(fpath, root)] = fpath
    return module_map


def resolve_to_module(raw: str, level: int, current: str, known: Set[str]) -> Optional[str]:
    if level > 0:
        parts = current.split(".")
        base = parts[: max(0, len(parts) - level)]
        candidate = ".".join(base + ([raw] if raw else []))
    else:
        candidate = raw

    if candidate in known:
        return candidate
    for m in known:
        if m.startswith(candidate + "."):
            return candidate
    return None


class ImportVisitor(ast.NodeVisitor):
    def __init__(self, module: str, known: Set[str]):
        self.module = module
        self.known = known
        self.deps: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            dep = resolve_to_module(alias.name, 0, self.module, self.known)
            if dep and dep != self.module:
                self.deps.add(dep)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        dep = resolve_to_module(node.module or "", node.level or 0, self.module, self.known)
        if dep and dep != self.module:
            self.deps.add(dep)

    def visit_If(self, node: ast.If):
        # Skip if TYPE_CHECKING:
        test = node.test
        is_type_checking = False
        if (
            isinstance(test, ast.Name)
            and test.id == "TYPE_CHECKING"
            or isinstance(test, ast.Attribute)
            and test.attr == "TYPE_CHECKING"
        ):
            is_type_checking = True

        if is_type_checking:
            return

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Skip lazy imports inside functions
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # Skip lazy imports inside async functions
        return


def build_graph(root: Path, exclude: Set[str]) -> Dict[str, Set[str]]:
    module_map = collect_module_map(root, exclude)
    known = set(module_map.keys())
    graph: Dict[str, Set[str]] = {m: set() for m in known}

    for module, fpath in module_map.items():
        try:
            tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
        except (SyntaxError, UnicodeDecodeError):
            continue

        visitor = ImportVisitor(module, known)
        visitor.visit(tree)
        graph[module] = visitor.deps

    return graph


def tarjan_sccs(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Iterative Tarjan's SCC. Returns only SCCs with more than 1 node (true cycles)."""
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stk: Set[str] = set()
    stk: List[str] = []
    sccs: List[List[str]] = []
    ctr = [0]

    for root_node in list(graph):
        if root_node in idx:
            continue

        call_stack: List[Tuple[str, Iterator[str]]] = []

        # Initialize DFS from root node
        idx[root_node] = low[root_node] = ctr[0]
        ctr[0] += 1
        stk.append(root_node)
        on_stk.add(root_node)
        call_stack.append((root_node, iter(sorted(graph.get(root_node, set())))))
        while call_stack:
            v, nbrs = call_stack[-1]
            advanced = False
            for w in nbrs:
                if w not in idx:
                    # Initialize DFS for neighbor w
                    idx[w] = low[w] = ctr[0]
                    ctr[0] += 1
                    stk.append(w)
                    on_stk.add(w)
                    call_stack.append((w, iter(sorted(graph.get(w, set())))))

                    advanced = True
                    break
                elif w in on_stk:
                    low[v] = min(low[v], idx[w])
            if not advanced:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    low[parent] = min(low[parent], low[v])
                if low[v] == idx[v]:
                    scc: List[str] = []
                    while True:
                        w = stk.pop()
                        on_stk.discard(w)
                        scc.append(w)
                        if w == v:
                            break
                    if len(scc) > 1:
                        sccs.append(scc)

    return sccs


def generate_html(cycles: List[List[str]], graph: Dict[str, Set[str]], output: Path) -> None:
    cycle_nodes = {m for cycle in cycles for m in cycle}
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.set_options(
        '{"physics":{"stabilization":{"iterations":200}},"edges":{"arrows":{"to":{"enabled":true,"scaleFactor":0.5}}}}'
    )
    for module in graph:
        color = "#e74c3c" if module in cycle_nodes else "#3498db"
        net.add_node(module, label=module.split(".")[-1], title=module, color=color, size=15)
    for src, targets in graph.items():
        for tgt in targets:
            if tgt in graph:
                is_cycle_edge = src in cycle_nodes and tgt in cycle_nodes
                net.add_edge(
                    src,
                    tgt,
                    color="#e74c3c" if is_cycle_edge else "#cccccc",
                    width=2 if is_cycle_edge else 1,
                )
    net.save_graph(str(output))
    print(f"{CYAN}Graph saved → {output}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect circular imports in a Python codebase.")
    parser.add_argument("directory", help="Root directory to scan")
    parser.add_argument("--exclude", nargs="+", default=[], help="Directory names to skip")
    parser.add_argument("--html", default="", metavar="PATH", help="Write interactive HTML graph to PATH")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.")
        sys.exit(1)

    exclude = set(args.exclude)
    print(f"Scanning '{root}'...")
    graph = build_graph(root, exclude)
    print(f"  {DIM}{len(graph)} modules found.{RESET}\n")

    cycles = tarjan_sccs(graph)

    if not cycles:
        print(f"{GREEN}✓  No circular imports detected.{RESET}")
    else:
        print(f"{RED}{BOLD}✗  Found {len(cycles)} circular import group(s):{RESET}\n")
        for i, cycle in enumerate(sorted(cycles, key=len, reverse=True), 1):
            cycle_set = set(cycle)
            print(f"  {YELLOW}Group {i}  ({len(cycle)} modules):{RESET}")
            for mod in sorted(cycle):
                arrows = sorted(graph.get(mod, set()) & cycle_set)
                print(f"    {RED}▶{RESET} {mod}  {DIM}→ {', '.join(arrows)}{RESET}")
            print()

    cyclic_count = sum(len(c) for c in cycles)
    print(f"{DIM}Summary: {cyclic_count}/{len(graph)} modules involved in cycles.{RESET}")

    if args.html:
        generate_html(cycles, graph, Path(args.html))

    sys.exit(1 if cycles else 0)


if __name__ == "__main__":
    main()
