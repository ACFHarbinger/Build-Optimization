"""
LKH-3 Tour Adapter Module.

Provides :class:`TourAdapter`, a thin shim that lets the flat ``List[int]``
tour representation used by the LKH-3 solver be consumed directly by the
operator interface shared across the local-search framework.

Background
----------
The framework's intra-route operators (:func:`move_kopt_intra`) and
perturbation operators (:func:`double_bridge`) are written against a
``LocalSearch``-style object that exposes:

- ``routes``     — a list of mutable open route lists (no closing duplicate).
- ``d``          — a 2-D distance matrix indexed by node id.
- ``C``          — a cost scale factor (positive = minimise).
- ``_update_map(route_set)`` — called by an operator after it mutates a route.

:class:`TourAdapter` satisfies that contract for a single route, allowing all
existing operators to be reused without modification inside the LKH-3 solver.

Public API
----------
TourAdapter(tour, distance_matrix)
    Wrap a closed or open tour list and a distance matrix.

TourAdapter.routes
    ``[route]`` where *route* is the mutable open node list.

TourAdapter.to_closed_tour() -> List[int]
    Reconstruct the closed tour (first node appended at end) after an
    operator may have mutated ``routes[0]`` in-place.

TourAdapter._updated
    Boolean flag set to ``True`` the first time ``_update_map`` is called,
    allowing callers to detect whether an operator applied a move.

Typical usage
-------------
>>> from logic.src.policies.helpers.operators.heuristics._tour_adapter import TourAdapter
>>> adapter = TourAdapter(tour, distance_matrix)
>>> applied = move_kopt_intra(adapter, u=u, v=v, r_u=0, p_u=i, r_v=0, p_v=j, k=2)
>>> if applied:
...     tour = adapter.to_closed_tour()
"""

from __future__ import annotations

from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# LocalSearch adapter
# ---------------------------------------------------------------------------


class TourAdapter:
    """
    Minimal adapter that presents a flat tour list as a single-route
    ``LocalSearch``-compatible object.

    Both ``move_kopt_intra`` and ``double_bridge`` expect an object with:

    - ``routes``: list of mutable route lists (open, no closing duplicate).
    - ``d``: distance matrix (2-D array-like, indexed by node id).
    - ``C``: cost scale factor (1.0 for minimisation).
    - ``_update_map(route_set)``: called after a move mutates the route.

    The adapter wraps a single flat tour (without the closing duplicate) as
    ``routes[0]``, exposes the distance matrix and a unit cost scale, and
    records whether ``_update_map`` was invoked so callers can detect that a
    move was applied.
    """

    def __init__(self, tour: List[int], distance_matrix: np.ndarray) -> None:
        # Operators work on open routes (no closing duplicate).
        route = tour[:-1] if (len(tour) > 1 and tour[0] == tour[-1]) else tour[:]
        self.routes: List[List[int]] = [route]
        self.d = distance_matrix
        # C = 1 ⟹ delta * C < -1e-4 ⟺ delta < -1e-4 (strict cost reduction)
        self.C: float = 1.0
        self._updated: bool = False

    def _update_map(self, route_set) -> None:  # noqa: ANN001
        """Record that the route was mutated by an operator."""
        self._updated = True

    def to_closed_tour(self) -> List[int]:
        """Return the current route as a closed tour (first node repeated at end)."""
        route = self.routes[0]
        if not route:
            return []
        return route + [route[0]]
