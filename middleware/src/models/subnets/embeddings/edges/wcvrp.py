"""wcvrp.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import wcvrp
"""

from .cvrpp import CVRPPEdgeEmbedding


class WCVRPEdgeEmbedding(CVRPPEdgeEmbedding):
    """
    Edge embedding for Waste Collection VRP problems.

    Inherits from CVRPEdgeEmbedding since WCVRP has the same graph
    structure (depot + customer nodes with capacity constraints).
    """

    pass
