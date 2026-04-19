"""tsp.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import tsp
"""

from .base import EdgeEmbedding


class TSPEdgeEmbedding(EdgeEmbedding):
    """
    Edge embedding for TSP-like problems.

    Uses pairwise Euclidean distances as edge features, optionally
    sparsified to k-nearest neighbors per node.
    """

    node_dim = 1
