"""
Simulator day logic for Neural Agent.
"""

from __future__ import annotations

import torch

from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import (
    get_route_cost,
)
from logic.src.tracking.hooks.attention_hooks import add_attention_hooks


class SimulationMixin:
    """
    Mixin for single-day simulation logic in NeuralAgent.
    """

    def compute_simulator_day(
        self,
        input,
        graph,
        distC,
        profit_vars=None,
        waste_history=None,
        cost_weights=None,
        mandatory=None,
        **kwargs,
    ):
        """Execute neural routing policy for a single simulation day."""
        edges, dist_matrix = graph
        hook_data = add_attention_hooks(self.model.encoder)  # type: ignore[attr-defined]

        # Check mandatory: if provided and empty (no bins to visit), skip routing
        if mandatory is not None:
            has_mandatory = self._validate_mandatory(mandatory)
            if not has_mandatory:
                for handle in hook_data["handles"]:
                    handle.remove()
                return ([0], 0, {"attention_weights": torch.tensor([]), "graph_masks": [], "mandatory_empty": True})

        mask = None
        input_for_model = input.copy()  # Shallow copy
        if "edges" not in input_for_model and edges is not None:
            input_for_model["edges"] = edges
        if "dist" not in input_for_model and dist_matrix is not None:
            input_for_model["dist"] = dist_matrix

        if mandatory is not None:
            input_for_model["mandatory"] = mandatory.unsqueeze(0) if mandatory.dim() == 1 else mandatory

        self._prepare_temporal_features(input_for_model, waste_history)

        # Expand dist_matrix for route improvement if needed (like in original code)
        if dist_matrix is not None and dist_matrix.dim() == 2:
            dist_matrix = dist_matrix.unsqueeze(0)
            # We don't have embeddings size yet to expand against batch.
            # But model forward will handle batching.

        out = self.model(  # type: ignore[attr-defined]
            input_for_model,
            return_pi=True,
            mask=mask,
            profit_vars=profit_vars,
            cost_weights=cost_weights,
        )
        if isinstance(out, dict):
            pi = out.get("actions") or out.get("pi")
        else:
            _, _, _, pi, _ = out

        if pi is None:
            raise ValueError("Model output does not contain 'actions' or 'pi'")

        route = torch.cat((torch.tensor([0]).to(pi.device), pi.squeeze(0)))
        cost = get_route_cost(distC * 100, route)
        for handle in hook_data["handles"]:
            handle.remove()

        attention_weights = torch.tensor([])
        if hook_data["weights"]:
            attention_weights = torch.stack(hook_data["weights"])

        route_list = route if isinstance(route, list) else route.cpu().tolist()
        n_visited = sum(1 for n in route_list if n != 0)
        if hasattr(self, "_viz_record"):
            self._viz_record(cost=float(cost), n_visited=n_visited, route_len=len(route_list))
        return (route_list, cost, {"attention_weights": attention_weights, "graph_masks": hook_data["masks"]})

    def _validate_mandatory(self, mandatory):
        """Validates if any bins are marked as mandatory."""
        if not isinstance(mandatory, torch.Tensor):
            mandatory = torch.tensor(mandatory, dtype=torch.bool)
        if mandatory.dim() == 1:
            mandatory = mandatory.unsqueeze(0)
        return mandatory[:, 1:].any() if mandatory.size(1) > 1 else mandatory.any()

    def _prepare_temporal_features(self, input_for_model, waste_history):
        """Populates temporal features (fill1, fill2, ...) based on waste history."""
        horizon = getattr(self.model, "temporal_horizon", 0)  # type: ignore[attr-defined]
        if horizon <= 0 or waste_history is None:
            return

        loc_tensor = (
            input_for_model.get("locs") if "locs" in input_for_model.keys() else input_for_model.get("loc", None)
        )
        if loc_tensor is None:
            raise KeyError("Input must contain 'loc' or 'locs'")

        h_feat = waste_history.unsqueeze(0) if waste_history.dim() == 2 else waste_history
        n_bins = loc_tensor.size(1) if loc_tensor.dim() == 3 else loc_tensor.size(0)

        if h_feat.size(1) != n_bins and h_feat.size(2) == n_bins:
            h_feat = h_feat.permute(0, 2, 1)

        if h_feat.max() > 2.0:
            h_feat = h_feat / 100.0

        for h in range(1, horizon + 1):
            if h_feat.size(2) >= horizon + 1:
                input_for_model[f"fill{h}"] = h_feat[:, :, -horizon - 1 : -1][:, :, h - 1]
            else:
                input_for_model[f"fill{h}"] = torch.zeros_like(h_feat[:, :, 0])
