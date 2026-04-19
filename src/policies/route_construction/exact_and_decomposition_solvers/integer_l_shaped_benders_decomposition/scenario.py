r"""
SAA Scenario Generator for the Integer L-Shaped Method.

Generates discrete Sample Average Approximation (SAA) scenarios representing
alternative realizations of absolute end-of-day bin fill levels w_i(ω) as
percentages in [0, 100].

Representation choice (B):
    Each scenario ω is a vector of *absolute* end-of-day fill levels, i.e.,
    w_i(ω) ∈ [0, 100] is the fraction of bin i's capacity that has accumulated
    by the end of the day under scenario ω.  This matches the ``sub_wastes``
    interface already used throughout the BaseRoutingPolicy pipeline.

References:
    Kleywegt, A. J., Shapiro, A., & Homem-de-Mello, T. (2002). "The sample
    average approximation method for stochastic discrete optimization". SIAM
    Journal on Optimization, 12(2), 479-502.
"""

from typing import Dict, List, Optional

import numpy as np


class ScenarioGenerator:
    r"""Generates discrete SAA scenarios of absolute bin fill levels.

    Each scenario ω represents a plausible end-of-day bin fill vector
    w(ω) = {w_1(ω), …, w_n(ω)}, where w_i(ω) ∈ [0, 100] is a percentage.

    Scenarios are drawn independently per bin from a Gamma distribution:
        w_i(ω) ~ Γ(α, β)   where α = 1/CV²,  β = μᵢ / α
        E[w_i] = μᵢ  (observed fill level),   Var[w_i] = (CV · μᵢ)²

    For bins with observed fill μᵢ ≈ 0, the Gamma parameterisation is
    degenerate; such bins are assigned a deterministic scenario value of 0.0.

    All sampled values are clipped to [0, 100] to remain in the valid fill range.

    Note:
        The generator produces S *independent* scenarios of equal probability 1/S.
        This is the standard SAA formulation of Kleywegt et al. (2002).  More
        sophisticated stratification methods (Latin Hypercube, quasi-Monte Carlo)
        are not implemented but could be added without changing the ILS interface.
    """

    def generate(
        self,
        sub_wastes: Dict[int, float],
        n_scenarios: int,
        fill_rate_cv: float = 0.3,
        seed: Optional[int] = 42,
    ) -> List[Dict[int, float]]:
        r"""Generate N SAA scenarios of absolute end-of-day fill levels.

        Args:
            sub_wastes: Observed fill levels {local_node_idx: fill_%}.
                        Node indices must be consistent with the master problem's
                        local numbering (0 = depot, 1..N = customers).
            n_scenarios: Number of scenarios S to generate.
            fill_rate_cv: Coefficient of variation CV = σ/μ for the Gamma
                          distribution.  Larger values increase scenario spread.
                          Must be strictly positive; values < 1e-9 produce
                          near-deterministic (high-α) scenarios.
            seed: Integer seed for the NumPy random Generator for reproducibility.

        Returns:
            List of S scenario dicts {local_node_idx: fill_%}, each representing
            one realisation of the fill vector.  All fill levels are guaranteed
            to lie in [0, 100].
        """
        rng = np.random.default_rng(seed)

        # Gamma shape parameter: α = 1 / CV²
        alpha = 1.0 / (fill_rate_cv**2) if fill_rate_cv > 1e-9 else 1e6

        scenarios: List[Dict[int, float]] = []

        for _ in range(n_scenarios):
            scenario: Dict[int, float] = {}

            for node_idx, fill in sub_wastes.items():
                if fill < 1e-9:
                    # Degenerate case: observed-empty bin → deterministic 0
                    scenario[node_idx] = 0.0
                else:
                    # β = μ / α  ⟹  E[Γ(α, β)] = α · β = μ
                    scale = fill / alpha
                    sampled = float(rng.gamma(alpha, scale))
                    scenario[node_idx] = float(np.clip(sampled, 0.0, 100.0))

            scenarios.append(scenario)

        return scenarios
