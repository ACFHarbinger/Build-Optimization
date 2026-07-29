r"""
Analytical recourse subproblem evaluator for the Integer L-Shaped Method.
"""

from typing import Dict, List, Tuple

import numpy as np


class RecourseEvaluator:
    r"""Analytical evaluator for the Stage 2 recourse function Q̄(ŷ).
    Optimized for multi-period simulation over T days.
    """

    def evaluate(
        self,
        y_hat: Dict[int, float],
        scenarios: List[List[np.ndarray]],
        overflow_penalty: float,
        undervisit_penalty: float,
        collection_threshold: float,
    ) -> Tuple[float, float, Dict[int, float]]:
        r"""Compute Q̄(ŷ) and Benders optimality cut coefficients over T days.

        Args:
            y_hat: Current master problem visit decisions {node_idx: value}.
            scenarios: List of scenario paths. Each path is a list of [node_fill_array] for T days.
            overflow_penalty: Cost per %-fill unit of overflow.
            undervisit_penalty: Cost of wasted trip Day 0.
            collection_threshold: Trigger level.

        Returns:
            Tuple of (Q_bar, e, d).
        """
        n_scenarios = len(scenarios)
        if n_scenarios == 0:
            return 0.0, 0.0, {}

        node_ids = list(y_hat.keys())
        inv_s = 1.0 / n_scenarios
        horizon = len(scenarios[0]) - 1  # T

        # Benders coefficients
        E_overflow: Dict[int, float] = {i: 0.0 for i in node_ids}
        E_undervisit: Dict[int, float] = {i: 0.0 for i in node_ids}

        for path in scenarios:
            for i in node_ids:
                # 1. Day 0 Undervisit Penalty (Deterministic-like given y_0)
                # This only depends on Day 0 fill level
                w_0 = path[0][i - 1]
                E_undervisit[i] += inv_s * undervisit_penalty * max(0.0, collection_threshold - w_0)

                # 2. Multi-Period Overflow Prediction
                # We simulate what happens if we SKIP node i today (y_i = 0)
                # versus if we VISIT node i today (y_i = 1).

                skip_overflow = 0.0
                visit_overflow = 0.0

                # Case y_i = 0 (Skip today)
                for t in range(1, horizon + 1):
                    w_t = path[t][i - 1]
                    skip_overflow += overflow_penalty * max(0.0, w_t - 100.0)

                # Case y_i = 1 (Visit today)
                for t in range(1, horizon + 1):
                    w_t_reset = max(0.0, path[t][i - 1] - path[0][i - 1])
                    visit_overflow += overflow_penalty * max(0.0, w_t_reset - 100.0)

                # Add to expected penalties
                E_overflow[i] += inv_s * skip_overflow
                E_undervisit[i] += inv_s * visit_overflow

        # Q_bar = sum(E_overflow_i * (1-y_i) + E_undervisit_i * y_i)
        Q_bar = 0.0
        for i in node_ids:
            y_i = float(y_hat.get(i, 0.0))
            Q_bar += E_overflow[i] * (1.0 - y_i)
            Q_bar += E_undervisit[i] * y_i

        # d_i = E_undervisit_i - E_overflow_i
        d = {i: E_undervisit[i] - E_overflow[i] for i in node_ids}
        e = Q_bar - sum(d[i] * float(y_hat.get(i, 0.0)) for i in node_ids)

        return Q_bar, e, d
