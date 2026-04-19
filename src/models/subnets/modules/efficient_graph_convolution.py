"""Optimized Graph Convolution implementation with multiple aggregators."""

from typing import Any, Iterable, Optional, Tuple, cast

import torch
from torch import Tensor
from torch.nn import Linear, Parameter

try:
    import torch_sparse
    from torch_geometric.nn import MessagePassing
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.nn.inits import glorot, zeros
    from torch_geometric.typing import Adj, OptTensor, SparseTensor
    from torch_geometric.utils import add_remaining_self_loops, scatter

    PYG_AVAILABLE = True
except (ImportError, OSError):
    # Handle both missing package and DLL load failures (common with CUDA mismatches)
    PYG_AVAILABLE = False
    MessagePassing = object  # type: ignore[assignment, misc]
    gcn_norm = None
    glorot = None
    zeros = None
    add_remaining_self_loops = None
    scatter = None

    # Define minimal types for structural compatibility
    Adj = Any
    OptTensor = Optional[Tensor]
    SparseTensor = Any


# Adapted from https://github.com/shyam196/egc
class EfficientGraphConvolution(MessagePassing):
    """
    Efficient Graph Convolution (EGC) with multiple aggregators.

    This layer computes node updates using a linear combination of different
    neighborhood aggregations (mean, max, sum, var, std, symnorm) and self-features.
    Supports multi-head weights and basis functions for efficiency.
    """

    _cached_edge_index: Optional[Tuple[Tensor, OptTensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggrs: Iterable[str] = ("symnorm",),
        n_heads: int = 8,
        num_bases: int = 4,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        sigmoid: bool = False,
        **kwargs,
    ):
        """
        Initialize the EGC layer.
        """
        if not PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric and torch_sparse are required for EfficientGraphConvolution. "
                "The libraries could not be loaded, possibly due to a CUDA/PyTorch version mismatch."
            )

        super(EfficientGraphConvolution, self).__init__(node_dim=1, **kwargs)
        if out_channels % n_heads != 0:
            raise ValueError("out_channels must be divisible by the number of heads")

        for a in aggrs:
            if a not in {"sum", "mean", "symnorm", "min", "max", "var", "std"}:
                raise ValueError("Unsupported aggregator: {}".format(a))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.num_bases = num_bases
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.aggregators = tuple(aggrs)  # Convert to tuple for mypy compatibility
        self.sigmoid = sigmoid

        self.bases_weight = Parameter(torch.Tensor(in_channels, (out_channels // n_heads) * num_bases))
        self.comb_weight = Linear(in_channels, n_heads * num_bases * len(self.aggregators))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the layer using Glorot initialization."""
        glorot(self.bases_weight)
        self.comb_weight.reset_parameters()
        zeros(self.bias)
        self._cached_adj_t = None
        self._cached_edge_index = None

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass for Efficient Graph Convolution.

        Args:
            *args: Positional arguments, expecting (x, edge_index).
            **kwargs: Keyword arguments, expecting x=Tensor, edge_index=Adj.

        Returns:
            Updated node features tensor.
        """
        x = kwargs.get("x", args[0] if len(args) > 0 else None)
        edge_index = kwargs.get("edge_index", args[1] if len(args) > 1 else None)

        if x is None or edge_index is None:
            raise ValueError("Forward requires x and edge_index")

        symnorm_weight: OptTensor = None
        if "symnorm" in self.aggregators or self.add_self_loops:
            edge_index, symnorm_weight = self._norm_and_cache(x, edge_index)

        batch_size = x.size(0)
        num_nodes = x.size(1)

        # [num_nodes, (out_channels // n_heads) * num_bases]
        bases = torch.matmul(x, self.bases_weight)
        # [num_nodes, n_heads * num_bases * num_aggrs]
        weightings = self.comb_weight(x)
        if self.sigmoid:
            weightings = torch.sigmoid_(weightings)

        if symnorm_weight is not None:
            symnorm_weight = symnorm_weight.view(-1, 1)

        # [num_nodes, num_aggregators, (out_channels // n_heads) * num_bases]
        # propagate_type: (x: Tensor, symnorm_weight: OptTensor)
        aggregated = self.propagate(edge_index, x=bases, symnorm_weight=symnorm_weight, size=None)

        weightings = weightings.view(
            batch_size,
            num_nodes,
            self.n_heads,
            self.num_bases * len(self.aggregators),
        )
        aggregated = aggregated.view(
            batch_size,
            num_nodes,
            len(self.aggregators) * self.num_bases,
            self.out_channels // self.n_heads,
        )

        # [num_nodes, n_heads, out_channels // n_heads]
        out = torch.matmul(weightings, aggregated)
        out = out.view(batch_size, num_nodes, self.out_channels)
        if self.bias is not None:
            out += self.bias

        return out

    def _norm_and_cache(self, x: Tensor, edge_index: Adj) -> Tuple[Adj, OptTensor]:
        """Helper to handle graph normalization and caching."""
        num_nodes = x.size(self.node_dim)

        if "symnorm" in self.aggregators:
            if isinstance(edge_index, Tensor):
                if self._cached_edge_index is None:
                    edge_index, sw = gcn_norm(edge_index, num_nodes=num_nodes, add_self_loops=self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, sw)

                    return edge_index, sw
                assert self._cached_edge_index is not None
                return self._cached_edge_index

            if self._cached_adj_t is None:
                st_edge_index = cast(torch_sparse.SparseTensor, edge_index)
                adj_t = gcn_norm(st_edge_index, num_nodes=num_nodes, add_self_loops=self.add_self_loops)
                if self.cached:
                    self._cached_adj_t = cast(SparseTensor, adj_t)

                return cast(Adj, adj_t), None
            assert self._cached_adj_t is not None
            return cast(Adj, self._cached_adj_t), None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                if self._cached_edge_index is None:
                    edge_index, _ = add_remaining_self_loops(edge_index)
                    if self.cached:
                        self._cached_edge_index = (edge_index, None)

                    return edge_index, None
                assert self._cached_edge_index is not None
                return self._cached_edge_index[0], None

            if self._cached_adj_t is None:
                st_edge_index = cast(torch_sparse.SparseTensor, edge_index)
                adj_t = torch_sparse.fill_diag(st_edge_index, 1.0)
                if self.cached:
                    self._cached_adj_t = cast(SparseTensor, adj_t)

                return cast(Adj, adj_t), None
            assert self._cached_adj_t is not None
            return cast(Adj, self._cached_adj_t), None

        return edge_index, None

    def message(self, x_j: Tensor) -> Tensor:  # type: ignore[override]
        """
        Passes messages along edges.
        """
        return x_j

    def aggregate(  # type: ignore[override]
        self,
        inputs: Tensor,
        index: Tensor,
        dim_size: Optional[int] = None,
        symnorm_weight: OptTensor = None,
    ) -> Tensor:
        """Aggregates messages from neighbors using multiple aggregators."""
        aggregated = []
        inputs = inputs.permute(1, 0, 2)
        for aggregator in self.aggregators:
            out = self._run_aggregator(aggregator, inputs, index, dim_size, symnorm_weight)
            aggregated.append(out)
        return torch.stack(aggregated, dim=1)

    def _run_aggregator(
        self,
        aggregator: str,
        inputs: Tensor,
        index: Tensor,
        dim_size: Optional[int],
        symnorm_weight: OptTensor,
    ) -> Tensor:
        """Execute a single neighborhood aggregator."""
        if aggregator == "sum":
            return scatter(inputs, index, 0, dim_size, reduce="sum")
        if aggregator == "symnorm":
            assert symnorm_weight is not None
            return scatter(inputs * symnorm_weight.unsqueeze(-1), index, 0, dim_size, reduce="sum")
        if aggregator == "mean":
            return scatter(inputs, index, 0, dim_size, reduce="mean")
        if aggregator == "min":
            return scatter(inputs, index, 0, dim_size, reduce="min")
        if aggregator == "max":
            return scatter(inputs, index, 0, dim_size, reduce="max")
        if aggregator in {"var", "std"}:
            mean = scatter(inputs, index, 0, dim_size, reduce="mean")
            mean_squares = scatter(inputs * inputs, index, 0, dim_size, reduce="mean")
            out = mean_squares - mean * mean
            if aggregator == "std":
                out = torch.sqrt(torch.relu(out) + 1e-5)
            return out
        raise ValueError(f'Unknown aggregator "{aggregator}".')

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:  # type: ignore[override]
        """
        Performs message passing and aggregation in a single step for sparse tensors.
        """
        aggregated = []
        if len(self.aggregators) > 1 and "symnorm" in self.aggregators:
            st_adj_t = cast(torch_sparse.SparseTensor, adj_t)
            adj_t_nonorm = st_adj_t.set_value(None, layout=None)
        else:
            adj_t_nonorm = adj_t

        for aggregator in self.aggregators:
            out = self._run_sparse_aggregator(aggregator, adj_t, adj_t_nonorm, x)
            aggregated.append(out)

        return torch.stack(aggregated, dim=1)

    def _run_sparse_aggregator(
        self, aggregator: str, adj_t: SparseTensor, adj_t_nonorm: SparseTensor, x: Tensor
    ) -> Tensor:
        """Execute a single sparse neighborhood aggregator."""
        if aggregator == "symnorm":
            out = torch_sparse.matmul(adj_t, x, reduce="sum")
            return cast(Tensor, out)

        if aggregator in ["var", "std"]:
            mean = torch_sparse.matmul(adj_t_nonorm, x, reduce="mean")
            mean_sq = torch_sparse.matmul(adj_t_nonorm, x * x, reduce="mean")
            assert isinstance(mean, Tensor)
            assert isinstance(mean_sq, Tensor)
            out = mean_sq - mean * mean
            if aggregator == "std":
                out = torch.sqrt(torch.relu(out) + 1e-5)
            return out

        out = torch_sparse.matmul(adj_t_nonorm, x, reduce=aggregator)
        return cast(Tensor, out)

    def __repr__(self):
        """String representation of the layer."""
        return "{}({}, {}, {})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.aggregators,
        )
