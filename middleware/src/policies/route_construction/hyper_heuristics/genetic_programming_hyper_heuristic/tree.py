"""
GP Tree definitions for the GPHH Constructive Heuristic Generator.

This module implements **Heuristic Generation** via Genetic Programming
(Burke et al., 2009).  Each GP tree acts as a *scoring function* for
candidate node insertions during constructive solution building.

**Deep Genetic Operators**:

Both crossover and mutation use a ``_collect_mutable_points()`` helper that
traverses the entire tree and returns a list of ``(parent, side)`` pairs —
one per child slot at every depth.  Selecting uniformly from this list
gives each subtree an equal probability of being chosen, so the operators
reach arbitrarily deep nodes rather than being confined to level 1.

**Terminal Set** (local tactical features, synchronised with solver):

    node_profit         Revenue of the candidate node (wastes[n] × R)
    distance_to_route   Min distance from candidate to nearest route node
    insertion_cost      Delta route distance at cheapest insertion position
    remaining_capacity  Vehicle capacity remaining after insertion

Reference:
    Burke, E. K., Hyde, M. R., Kendall, G., Ochoa, G., Ozcan, E., & Woodward, J. R.
    "Exploring Hyper-heuristic Methodologies with Genetic Programming", 2009
"""

import random
from typing import Callable, Dict, List, Optional, Tuple, Union


def protected_div(a: float, b: float) -> float:
    """Protected division: returns 1.0 if denominator is near-zero."""
    return a / b if abs(b) > 1e-9 else 1.0


class ConstantNode:
    """
    Ephemeral Random Constant (ERC) node: stores a fixed random scalar.

    Attributes:
        val: Randomly generated float value.
    """

    __slots__ = ("val",)

    def __init__(self, val: float):
        self.val = val

    def evaluate(self, ctx: Dict[str, float]) -> float:
        """Return the stored constant value."""
        return self.val

    def copy(self) -> "ConstantNode":
        """Create a deep copy of this constant node."""
        return ConstantNode(self.val)

    def size(self) -> int:
        """Return the size of this constant node (always 1)."""
        return 1

    def depth(self) -> int:
        """Return the maximum depth of this constant (always 1)."""
        return 1

    def compile(self) -> str:
        """Return the string representation of the constant."""
        return str(self.val)


class TerminalNode:
    """
    GP terminal node: extracts a continuous feature from the insertion context.

    Attributes:
        feature: Name of the feature to extract from the context dictionary.
    """

    __slots__ = ("feature",)

    def __init__(self, feature: str):
        self.feature = feature

    def evaluate(self, ctx: Dict[str, float]) -> float:
        """
        Evaluate the terminal by extracting its feature value.

        Args:
            ctx: Insertion-context dictionary.

        Returns:
            Float value of the feature (default 0.0 if not found).
        """
        return ctx.get(self.feature, 0.0)

    def copy(self) -> "TerminalNode":
        """Create a deep copy of this terminal node."""
        return TerminalNode(self.feature)

    def size(self) -> int:
        """Return the size of this terminal node (always 1)."""
        return 1

    def depth(self) -> int:
        """Return the maximum depth of this terminal (always 1)."""
        return 1

    def compile(self) -> str:
        """Return the feature name as a raw Python variable string."""
        return self.feature


class FunctionNode:
    """
    GP function node: applies an arithmetic operation to two child subtrees.

    All operations return continuous float values, maintaining the Closure
    Property required for type-consistent GP.

    Attributes:
        fn: Name of the arithmetic function (ADD, SUB, MUL, DIV, MAX, MIN).
        left: Left child subtree.
        right: Right child subtree.
    """

    __slots__ = ("fn", "left", "right")

    def __init__(self, fn: str, left: "GPNode", right: "GPNode"):
        self.fn = fn
        self.left = left
        self.right = right

    def evaluate(self, ctx: Dict[str, float]) -> float:
        """
        Recursively evaluate the arithmetic function.

        Uses protected division (returns 1.0 when denominator ≈ 0) to
        prevent runtime errors and maintain closure.

        Args:
            ctx: Insertion-context dictionary passed to children.

        Returns:
            Float result of the arithmetic operation.
        """
        left_val = self.left.evaluate(ctx)
        right_val = self.right.evaluate(ctx)

        if self.fn == "ADD":
            return left_val + right_val
        elif self.fn == "SUB":
            return left_val - right_val
        elif self.fn == "MUL":
            return left_val * right_val
        elif self.fn == "DIV":
            # Protected division: return 1.0 if denominator is near-zero
            return protected_div(left_val, right_val)
        elif self.fn == "MAX":
            return max(left_val, right_val)
        elif self.fn == "MIN":
            return min(left_val, right_val)
        else:
            return 0.0  # Fallback for unknown functions

    def copy(self) -> "FunctionNode":
        """Create a deep copy of this function node and its subtrees."""
        return FunctionNode(
            self.fn,
            self.left.copy(),
            self.right.copy(),
        )

    def size(self) -> int:
        """Recursively calculate the total number of nodes in this subtree."""
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        """Recursively calculate the maximum depth of this subtree."""
        return 1 + max(self.left.depth(), self.right.depth())

    def compile(self) -> str:
        """Recursively build a Python-parsable mathematical expression string."""
        l = self.left.compile()
        r = self.right.compile()

        if self.fn == "ADD":
            return f"({l} + {r})"
        elif self.fn == "SUB":
            return f"({l} - {r})"
        elif self.fn == "MUL":
            return f"({l} * {r})"
        elif self.fn == "DIV":
            # Map DIV to the protected_div helper in the compiled string
            return f"protected_div({l}, {r})"
        elif self.fn == "MAX":
            return f"max({l}, {r})"
        elif self.fn == "MIN":
            return f"min({l}, {r})"
        return "0.0"


GPNode = Union[TerminalNode, FunctionNode, ConstantNode]

# ---------------------------------------------------------------------------
# Terminals & Functions
# ---------------------------------------------------------------------------
# Local insertion features — synchronised with solver._build_insertion_context().
_TERMINALS = [
    "node_profit",  # Revenue of candidate node
    "distance_to_route",  # Min distance from candidate to nearest route node
    "insertion_cost",  # Delta route distance at cheapest insertion position
    "remaining_capacity",  # Vehicle capacity remaining after insertion
]

# Available arithmetic functions (all return float → Closure Property)
_FUNCTIONS = ["ADD", "SUB", "MUL", "DIV", "MAX", "MIN"]

# ---------------------------------------------------------------------------
# Mutable point collection — key to deep operators
# ---------------------------------------------------------------------------

# A MutablePoint identifies a subtree location.
# (parent, side) identifies a child slot.
# If parent is None, the point identifies the ROOT of the tree.
MutablePoint = Tuple[Optional["FunctionNode"], Optional[str]]  # (parent, side)


def _collect_mutable_points(tree: GPNode) -> List[MutablePoint]:
    """
    Traverse the tree and collect every node (including the root) as a mutable point.

    Each ``(parent, side)`` pair represents a location in the tree where a
    subtree can be replaced.  By including ``(None, None)``, we allow the
    root node itself to be replaced by crossover or mutation, preventing
    "frozen root" syndrome.

    Args:
        tree: Root node of the GP tree.

    Returns:
        List of (parent, side) pairs.  Parent is None for the root node.
    """
    # Start with the root itself as a mutable point
    points: List[MutablePoint] = [(None, None)]

    def _recurse(node: GPNode, parent: FunctionNode, side: str):
        points.append((parent, side))
        if isinstance(node, FunctionNode):
            _recurse(node.left, node, "left")
            _recurse(node.right, node, "right")

    if isinstance(tree, FunctionNode):
        _recurse(tree.left, tree, "left")
        _recurse(tree.right, tree, "right")

    return points


def _get_subtree(parent: "FunctionNode", side: str) -> GPNode:
    """Return the child subtree at the given side."""
    return parent.left if side == "left" else parent.right


def _set_subtree(parent: "FunctionNode", side: str, subtree: GPNode) -> None:
    """Replace the child subtree at the given side in-place."""
    if side == "left":
        parent.left = subtree
    else:
        parent.right = subtree


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------


def _random_tree(depth: int, rng: random.Random) -> GPNode:
    """
    Generate a random GP tree with pure arithmetic structure (grow method).

    At each level, there is a 40% chance of generating a terminal to produce
    trees of varying shape and depth.

    Args:
        depth: Maximum depth of the tree (0 = terminal only).
        rng: Random number generator.

    Returns:
        Root node of the generated tree.
    """
    if depth == 0 or rng.random() < 0.4:
        # Terminal branch: 80% feature terminal, 20% ERC constant
        if rng.random() < 0.8:
            return TerminalNode(rng.choice(_TERMINALS))
        else:
            return ConstantNode(rng.uniform(-1.0, 1.0))

    fn = rng.choice(_FUNCTIONS)
    return FunctionNode(
        fn,
        _random_tree(depth - 1, rng),
        _random_tree(depth - 1, rng),
    )


# ---------------------------------------------------------------------------
# Deep crossover & mutation
# ---------------------------------------------------------------------------


def _subtree_crossover(t1: GPNode, t2: GPNode, rng: random.Random, max_depth: int) -> Tuple[GPNode, GPNode]:
    """
    Deep subtree crossover: swap randomly chosen subtrees at any depth.

    By allowing the ROOT (parent=None) to be selected, the genetic operator
    can completely replace one tree with a subtree of the other.

    Koza-style depth limit: if either offspring exceeds max_depth, the
    swap is rejected and the original parents are returned.

    Algorithm:
        1. Collect all points (including root) in copies of t1 and t2.
        2. Pick one point uniformly at random from each.
        3. Extract the subtrees and swap them.
        4. Validate offspring depth; reject if bloated.

    Args:
        t1: First parent tree.
        t2: Second parent tree.
        rng: Random number generator.
        max_depth: Maximum allowed tree depth.

    Returns:
        Tuple of two offspring root nodes.
    """
    c1 = t1.copy()
    c2 = t2.copy()

    pts1 = _collect_mutable_points(c1)
    pts2 = _collect_mutable_points(c2)

    # Choose random points (guaranteed to have at least one: the root)
    p1, s1 = rng.choice(pts1)
    p2, s2 = rng.choice(pts2)

    # Get subtrees: use the node itself if parent is None (root)
    sub1 = _get_subtree(p1, s1) if p1 is not None else c1  # type: ignore[arg-type]
    sub2 = _get_subtree(p2, s2) if p2 is not None else c2  # type: ignore[arg-type]

    # Swap subtree at point 1: if p1 is None, we are replacing the root of c1
    if p1 is None:
        c1 = sub2.copy()
    else:
        _set_subtree(p1, s1, sub2.copy())  # type: ignore[arg-type]

    # Swap subtree at point 2: if p2 is None, we are replacing the root of c2
    if p2 is None:
        c2 = sub1.copy()
    else:
        _set_subtree(p2, s2, sub1.copy())  # type: ignore[arg-type]

    # --- Depth limit enforcement (Tree Bloat mitigation) ---
    # If either offspring exceeds the configured depth limit, discard the
    # change and return copies of the original parents.
    if c1.depth() > max_depth or c2.depth() > max_depth:
        return t1.copy(), t2.copy()

    return c1, c2


def _mutate(
    tree: GPNode,
    depth: int,
    rng: random.Random,
    max_depth: int,
    replacement_depth: Optional[int] = None,
) -> GPNode:
    """
    Deep point mutation: replace a randomly chosen subtree at any depth.

    If the ROOT (parent=None) is selected for mutation, the entire tree
    is replaced by a fresh random tree.

    Koza-style depth limit: if the new tree exceeds max_depth, the
    mutation is rejected and the original tree is returned.

    Args:
        tree: Tree to mutate.
        depth: Maximum depth of the original tree.
        rng: Random number generator.
        max_depth: Strict enforcement depth limit.
        replacement_depth: Max depth for the replacement subtree.

    Returns:
        Mutated root node (or original if bloat check fails).
    """
    if replacement_depth is None:
        replacement_depth = max(1, depth - 1)

    # Create a backup to allow rejection of bloated offspring
    original_copy = tree.copy()

    pts = _collect_mutable_points(tree)
    parent, side = rng.choice(pts)

    # Get the target node to check if it's a constant
    target = _get_subtree(parent, side) if parent is not None else tree  # type: ignore[arg-type]

    # --- ERC Perturbation Optimization ---
    # If the selected node is an Ephemeral Random Constant, we have a 50%
    # chance to slightly perturb its value (Gaussian noise) rather than
    # replacing the entire branch. This allows the GP to "fine-tune" weights.
    if isinstance(target, ConstantNode) and rng.random() < 0.5:
        target.val += rng.gauss(0.0, 0.1)
        # No depth check needed for perturbation; return the tree
        return tree

    # Generate replacement subtree (standard mutation)
    new_sub = _random_tree(replacement_depth, rng)

    if parent is None:
        # Replace the entire root node
        mutated_tree = new_sub
    else:
        _set_subtree(parent, side, new_sub)  # type: ignore[arg-type]
        mutated_tree = tree

    # --- Depth limit enforcement (Tree Bloat mitigation) ---
    if mutated_tree.depth() > max_depth:
        return original_copy

    return mutated_tree


def compile_tree(tree: GPNode) -> str:
    """
    Export the GP tree to a raw Python expression string.

    This string can be passed to ``eval(compile(...))`` in the solver
    to create a fast lambda function for O(1) candidate evaluation.
    """
    expr = tree.compile()
    # Terminals are local variables in the resulting lambda execution
    return f"lambda node_profit, distance_to_route, insertion_cost, remaining_capacity: {expr}"


def to_callable(tree: GPNode) -> Callable:
    """
    Compile the GP tree into a fast-executing Python lambda function.

    Transforms the recursive object structure into a flat mathematical
    expression that bypasses dictionary lookups and recursion overhead.
    """
    code_str = compile_tree(tree)
    # Compile the lambda string into a code object for faster execution
    code_obj = compile(code_str, "<gp_tree_eval>", "eval")
    # Evaluate the code object, injecting the protected_div helper into scope
    return eval(code_obj, {"protected_div": protected_div, "max": max, "min": min})
