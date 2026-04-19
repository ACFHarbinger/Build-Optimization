"""
Batch simulation logic for Neural Agent.
"""

from __future__ import annotations

import torch

from logic.src.tracking.hooks import add_attention_hooks


class BatchMixin:
    """
    Mixin for batch simulation logic in NeuralAgent.
    """

    def compute_batch_sim(
        self,
        input,
        dist_matrix,
        hrl_manager=None,
        waste_history=None,
        threshold=0.5,
        mask_threshold=0.5,
    ):
        """
        Compute simulation step for a batch of problem instances.

        Used during training and batch evaluation. Optionally integrates with
        HRL manager for gating (decide whether to dispatch) and mandatory selection
        (which bins must be collected).

        Args:
            input (dict): Batch of problem data with 'loc', 'waste', etc.
            dist_matrix (torch.Tensor): Distance matrix (B x N x N) or (N x N)
            hrl_manager (optional): HRL manager network for dispatch and mandatory decisions
            waste_history (torch.Tensor, optional): Historical waste levels (B x N x T)
            threshold (float): Gating probability threshold. Default: 0.5
            mask_threshold (float): Mandatory selection probability threshold. Default: 0.5

        Returns:
            Tuple[torch.Tensor, dict, dict]: Costs, result metrics, and attention data
                - ucost: Unweighted costs tensor
                - ret_dict: Dictionary with 'overflows', 'kg', 'waste', 'km'
                - output_dict: Dictionary with 'attention_weights', 'graph_masks'
        """
        hook_data = add_attention_hooks(self.model.encoder)  # type: ignore[attr-defined]

        mask = None
        if hrl_manager is not None and waste_history is not None:
            # Static: Customer Locations (Batch, N, 2)
            static_feat = input["locs"]
            # Dynamic: Waste History (Batch, N, History)
            dynamic_feat = waste_history

            # Compute Global Features
            current_waste = dynamic_feat[:, :, -1]

            # 1. Critical ratio Now
            critical_mask = current_waste > hrl_manager.critical_threshold
            critical_ratio = critical_mask.float().mean(
                dim=1, keepdim=True
            )  # (B, 1)  # pyrefly: ignore[missing-attribute]

            # 2. Max Current Waste
            max_current_waste = current_waste.max(dim=1, keepdim=True)[0]  # (B, 1)

            # Combine: (B, 2)
            global_features = torch.cat(
                [
                    critical_ratio,
                    max_current_waste,
                ],
                dim=1,
            )

            # Get Action (Deterministic)
            # mandatory_action: 1 = must collect, 0 = optional
            mandatory_action, gate_action, _ = hrl_manager.select_action(
                static_feat,
                dynamic_feat,
                global_features,
                deterministic=True,
                threshold=threshold,
                mandatory_threshold=mask_threshold,
            )

            # Construct model mask from mandatory_action
            # mandatory_action: 1=must collect, 0=optional
            # Model mask: True=masked(can skip), False=must visit
            mask = mandatory_action == 0

            # Apply Gate: If Gate=0, Mask ALL (no routing)
            gate_mask = (gate_action == 0).unsqueeze(1).expand_as(mask)
            mask = mask | gate_mask

        # Use model forward pass
        out = self.model(input, cost_weights=None, return_pi=True, pad=False, mask=mask)  # type: ignore[attr-defined]
        if isinstance(out, dict):
            pi = out.get("actions") or out.get("pi")
            cost = out.get("cost")
        else:
            cost, _, _, pi, _ = out

        if pi is None:
            raise ValueError("Model output does not contain 'actions' or 'pi'")

        # Calculate ucost (unweighted cost)
        ucost, cost_dict, _ = self.problem.get_costs(input, pi, cw_dict=None)  # type: ignore[attr-defined]

        src_vertices, dst_vertices = pi[:, :-1], pi[:, 1:]
        dst_mask = dst_vertices != 0
        pair_mask = (src_vertices != 0) & (dst_mask)
        # To avoid index error on dst_vertices if size is 0 or something?
        if dst_vertices.size(1) > 0:
            last_dst = torch.max(
                dst_mask * torch.arange(dst_vertices.size(1), device=dst_vertices.device),
                dim=1,
            ).indices
        else:
            last_dst = torch.zeros(dst_vertices.size(0), dtype=torch.long, device=dst_vertices.device)

        travelled = dist_matrix[src_vertices, dst_vertices] * pair_mask.float()  # pyrefly: ignore[missing-attribute]

        ret_dict = {}
        ret_dict["overflows"] = cost_dict["overflows"]
        ret_dict["kg"] = cost_dict["waste"] * 100
        ret_dict["waste"] = cost_dict["waste"]

        if dist_matrix.dim() == 2:
            # Assuming batch dim 0 is implied?
            # dist_matrix: (N, N). src_vertices: (B, L).
            # This logic mimics original code but might need careful checking if dist_matrix is (B, N, N)
            ret_dict["km"] = (
                travelled.sum(dim=1)
                + dist_matrix[0, src_vertices[:, 0]]
                + dist_matrix[
                    dst_vertices[
                        torch.arange(dst_vertices.size(0), device=dst_vertices.device),
                        last_dst,
                    ],
                    0,
                ]
            )
        else:
            ret_dict["km"] = (
                travelled.sum(dim=1)
                + dist_matrix[0, 0, src_vertices[:, 0]]
                + dist_matrix[
                    0,
                    dst_vertices[
                        torch.arange(dst_vertices.size(0), device=dst_vertices.device),
                        last_dst,
                    ],
                    0,
                ]
            )

        attention_weights = torch.tensor([])
        if hook_data["weights"]:
            attention_weights = torch.stack(hook_data["weights"])

        graph_masks = torch.tensor([])
        if hook_data["masks"]:
            graph_masks = torch.stack(hook_data["masks"])

        if hasattr(self, "_viz_record"):
            self._viz_record(
                mean_cost=float(ucost.mean().item()),
                mean_kg=float(ret_dict["kg"].mean().item()),
                mean_km=float(ret_dict["km"].mean().item()),
                batch_size=int(ucost.size(0)),
            )

        return (
            ucost,
            ret_dict,
            {"attention_weights": attention_weights, "graph_masks": graph_masks},
        )
