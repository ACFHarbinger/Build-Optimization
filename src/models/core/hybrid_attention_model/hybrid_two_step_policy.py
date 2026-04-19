from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.autoregressive.policy import AutoregressivePolicy
from logic.src.models.policies.adaptive_large_neighborhood_search import VectorizedALNS
from logic.src.models.policies.ant_colony_system import VectorizedACOPolicy
from logic.src.models.policies.hybrid_genetic_search import VectorizedHGS
from logic.src.models.policies.operators import (
    # Destroy
    vectorized_cluster_removal,
    # Exchange
    vectorized_cross_exchange,
    vectorized_ejection_chain,
    # Repair
    vectorized_greedy_insertion,
    vectorized_lambda_interchange,
    # Route
    vectorized_lkh,
    vectorized_or_opt,
    vectorized_random_removal,
    vectorized_regret_k_insertion,
    # Move
    vectorized_relocate,
    vectorized_shaw_removal,
    vectorized_string_removal,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
    # Unstringing
    vectorized_type_i_unstringing,
    vectorized_type_ii_unstringing,
    vectorized_type_iii_unstringing,
    vectorized_type_iv_unstringing,
    vectorized_worst_removal,
)
from logic.src.models.subnets.embeddings import get_init_embedding
from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder

from .improvement_step_decoder import ImprovementStepDecoder


class HybridTwoStagePolicy(AutoregressivePolicy):
    """
    Hybrid Policy with 2-Stage Decoder:
    1. Initialization: Select HGS, ALNS, or ACO.
    2. Refinement: Iteratively select vectorized operators.
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        refine_steps: int = 10,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(env_name=env_name, embed_dim=embed_dim)

        self.refine_steps = refine_steps
        self.seed = seed
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        self.init_embedding = get_init_embedding(env_name, embed_dim)
        self.encoder = GraphAttentionEncoder(
            n_heads=n_heads, embed_dim=embed_dim, feed_forward_hidden=hidden_dim, n_layers=n_encode_layers, **kwargs
        )

        # Stage 1: Init Selector (3 choices: HGS, ALNS, ACO)
        self.init_router = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.INIT_ALGOS = ["hgs", "alns", "aco"]

        # Stage 2: Improvement Decoder
        # Available Operators
        self.OPERATORS = [
            # Destroy
            vectorized_cluster_removal,
            vectorized_random_removal,
            vectorized_shaw_removal,
            vectorized_string_removal,
            vectorized_worst_removal,
            # Exchange
            vectorized_cross_exchange,
            vectorized_ejection_chain,
            vectorized_lambda_interchange,
            vectorized_or_opt,
            # Move
            vectorized_relocate,
            vectorized_swap,
            # Repair
            vectorized_greedy_insertion,
            vectorized_regret_k_insertion,
            # Route
            vectorized_lkh,
            vectorized_swap_star,
            vectorized_three_opt,
            vectorized_two_opt,
            vectorized_two_opt_star,
            # Unstringing
            vectorized_type_i_unstringing,
            vectorized_type_ii_unstringing,
            vectorized_type_iii_unstringing,
            vectorized_type_iv_unstringing,
        ]
        self.improvement_decoder = ImprovementStepDecoder(
            embed_dim=embed_dim, n_operators=len(self.OPERATORS), hidden_dim=hidden_dim
        )

    def __getstate__(self) -> Dict[str, Any]:
        """Return state for pickling."""
        state = self.__dict__.copy()
        # Handle torch.Generator which is not picklable
        if "generator" in state:
            gen = state["generator"]
            state["generator_state"] = gen.get_state()
            state["generator_device"] = str(gen.device)
            del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after pickling."""
        if "generator_state" in state:
            gen_state = state.pop("generator_state")
            gen_device = state.pop("generator_device")
            # Create a new generator on the same device
            gen = torch.Generator(device=gen_device)
            gen.set_state(gen_state)
            state["generator"] = gen
        self.__dict__.update(state)

    def _initialize_tours(
        self, td: TensorDict, env: RL4COEnvBase, strategy: str, embeddings: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: Initialization."""
        batch_size = td.size(0)
        device = td["locs"].device

        graph_context = embeddings.mean(dim=1)
        init_logits = self.init_router(graph_context)

        if strategy == "greedy":
            init_choice_idx = init_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(init_logits, dim=-1)
            init_choice_idx = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)

        tours = torch.zeros(batch_size, td["locs"].size(1), dtype=torch.long, device=device)

        if not hasattr(self, "aco_model"):
            self.aco_model = VectorizedACOPolicy(env_name=self.env_name, embed_dim=self.encoder.embed_dim, **kwargs)  # type: ignore[arg-type, union-attr]

        for algo_idx, algo_name in enumerate(self.INIT_ALGOS):
            mask = init_choice_idx == algo_idx
            if not mask.any():
                continue

            sub_td = td[mask]
            dist_matrix = self._get_dist_matrix(sub_td)
            wastes = sub_td.get("waste", None)

            if algo_name == "hgs":
                solver = VectorizedHGS(dist_matrix, wastes, vehicle_capacity=1.0, device=device)
                rand_sol = self._get_random_tours(sub_td)
                sub_tours, _ = solver.solve(rand_sol, n_generations=5, population_size=10)
            elif algo_name == "alns":
                solver = VectorizedALNS(dist_matrix, wastes, vehicle_capacity=1.0, device=device)  # type: ignore[assignment]
                rand_sol = self._get_random_tours(sub_td)
                sub_tours, _ = solver.solve(rand_sol, n_iterations=10)  # type: ignore[call-arg]
            elif algo_name == "aco":
                out = self.aco_model(sub_td, env, **kwargs)
                sub_tours = out["actions"]
            else:
                raise ValueError(f"Unknown initialization algorithm: {algo_name}")

            if not isinstance(sub_tours, list):
                tours[mask] = sub_tours.to(dtype=torch.long)

        return tours, init_choice_idx

    def _apply_operator_step(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        embeddings: torch.Tensor,
        current_tours: torch.Tensor,
        dist_matrix_all: torch.Tensor,
        removed_nodes_state: torch.Tensor,
        strategy: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a single refinement step."""
        batch_size = td.size(0)
        device = td.device
        op_logits, _ = self.improvement_decoder(td, embeddings, env)

        log_p = torch.zeros(batch_size, device=device)
        if strategy == "greedy":
            op_idx = op_logits.argmax(dim=-1)
        else:
            probs = torch.softmax(op_logits, dim=-1)
            op_idx = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
            log_p = torch.log(probs.gather(1, op_idx.unsqueeze(-1)) + 1e-10).squeeze(-1)

        next_tours = current_tours.clone()
        updated_removed_nodes_list = [None] * batch_size

        for o_i, operator_fn in enumerate(self.OPERATORS):
            mask = op_idx == o_i
            if not mask.any():
                continue

            sub_tours, sub_dist = current_tours[mask], dist_matrix_all[mask]
            sub_removed = (
                removed_nodes_state[mask]
                if removed_nodes_state.size(0) == batch_size
                else torch.zeros((sub_tours.size(0), 0), dtype=torch.long, device=device)
            )

            new_sub_tours, new_sub_removed = self._execute_refinement_operator(
                operator_fn, sub_tours, sub_dist, sub_removed, device
            )

            # Handle shape mismatch: Expand next_tours if new tours are longer
            if new_sub_tours.size(1) > next_tours.size(1):
                diff = new_sub_tours.size(1) - next_tours.size(1)
                padding = torch.zeros((batch_size, diff), dtype=next_tours.dtype, device=device)
                next_tours = torch.cat([next_tours, padding], dim=1)

            # Pad new_sub_tours if shorter than current max width
            if new_sub_tours.size(1) < next_tours.size(1):
                diff = next_tours.size(1) - new_sub_tours.size(1)
                padding = torch.zeros((new_sub_tours.size(0), diff), dtype=new_sub_tours.dtype, device=device)
                new_sub_tours = torch.cat([new_sub_tours, padding], dim=1)

            next_tours[mask] = new_sub_tours
            sub_indices = torch.nonzero(mask).squeeze(-1)
            if sub_indices.dim() == 0:
                sub_indices = sub_indices.unsqueeze(0)
            for i, idx in enumerate(sub_indices):
                updated_removed_nodes_list[idx.item()] = new_sub_removed[i]

        return next_tours, self._assemble_removed_state(updated_removed_nodes_list, device), log_p

    def _execute_refinement_operator(
        self,
        operator_fn,
        sub_tours: torch.Tensor,
        sub_dist: torch.Tensor,
        sub_removed: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to safely execute a refinement operator."""
        sig = inspect.signature(operator_fn)
        valid_kwargs = {}
        if "max_iterations" in sig.parameters:
            valid_kwargs["max_iterations"] = 5
        if "n_remove" in sig.parameters:
            valid_kwargs["n_remove"] = int(sub_tours.size(1) * 0.1) + 1
        if "removed_nodes" in sig.parameters:
            valid_kwargs["removed_nodes"] = sub_removed  # type: ignore[assignment]

        try:
            result = operator_fn(sub_tours, sub_dist, **valid_kwargs)
        except Exception:
            result = sub_tours

        if isinstance(result, tuple):
            return result

        return result, (
            torch.zeros((sub_tours.size(0), 0), dtype=torch.long, device=device)
            if "removed_nodes" in sig.parameters
            else sub_removed
        )

    def _assemble_removed_state(self, removed_list: list, device: torch.device) -> torch.Tensor:
        """Helper to re-assemble removed state tensor."""
        batch_size = len(removed_list)
        max_len = max([r.size(0) if r is not None else 0 for r in removed_list])
        if max_len == 0:
            return torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        new_state = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
        for b, r in enumerate(removed_list):
            if r is not None:
                new_state[b, : r.size(0)] = r
        return new_state

    def forward(  # type: ignore[override]
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "greedy",
        **kwargs,
    ) -> Dict[str, Any]:
        # 1. Encode Graph
        init_embeds = self.init_embedding(td)
        embeddings = self.encoder(init_embeds)  # type: ignore[misc]

        # 2. Stage 1: Initialization
        current_tours, init_choice_idx = self._initialize_tours(td, env, strategy, embeddings, **kwargs)

        # 3. Stage 2: Refinement Loop
        dist_matrix_all = self._get_dist_matrix(td)
        removed_nodes_state = torch.zeros((td.size(0), 0), dtype=torch.long, device=td.device)
        log_likelihood = torch.zeros(td.size(0), device=td.device)

        for _step in range(self.refine_steps):
            current_tours, removed_nodes_state, step_log_p = self._apply_operator_step(
                td, env, embeddings, current_tours, dist_matrix_all, removed_nodes_state, strategy
            )
            log_likelihood = log_likelihood + step_log_p  # type: ignore[assignment]

        # Calculate final reward
        reward = env.get_reward(td, current_tours)

        return {
            "actions": current_tours,
            "reward": reward,
            "log_likelihood": log_likelihood,
            "init_choice": init_choice_idx,
        }

    def _get_dist_matrix(self, td):
        locs = td["locs"]
        return torch.cdist(locs, locs)

    def _get_random_tours(self, td):
        B, N, _ = td["locs"].shape
        # Simple random permutation for N-1 nodes (excluding depot 0)
        perms = torch.stack([torch.randperm(N - 1) + 1 for _ in range(B)]).to(td.device)
        return perms
