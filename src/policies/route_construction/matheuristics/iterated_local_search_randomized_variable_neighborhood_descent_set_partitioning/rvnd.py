"""
Randomized Variable Neighborhood Descent (RVND) Implementation.

This module implements the strict RVND procedure as defined in:
    Subramanian, A., Drummond, L. M. A., Bentes, C., Ochi, L. S., & Farias, R. (2013).
    "A parallel heuristic for the vehicle routing problem with simultaneous pickup and delivery".
    Computers & Operations Research, 40(7), 1899-1911.

RVND systematically explores multiple neighborhood structures in a randomized sequence.
When an improvement is found in any neighborhood, the search restarts with a fresh,
reshuffled neighborhood list. The algorithm terminates only when all neighborhoods
have been explored without finding any improvement.
"""

from random import Random
from typing import Callable, List


class RVND:
    """
    Randomized Variable Neighborhood Descent.

    Implements the RVND metaheuristic that explores multiple neighborhood structures
    in a randomized order. Unlike standard VND with a fixed neighborhood ordering,
    RVND randomizes the sequence and resets it completely upon finding any improvement.

    The algorithm maintains a neighborhood list (NL) of operator indices. At each step:
    1. A random neighborhood is selected from NL
    2. The corresponding operator is applied to the current solution
    3. If improvement is found:
       - Accept the improved solution
       - Reset and reshuffle NL to include all neighborhoods again
    4. If no improvement is found:
       - Remove the tested neighborhood from NL
    5. Terminate when NL is empty (local optimum reached)

    This randomization strategy prevents the algorithm from getting stuck in patterns
    and provides better exploration of the solution space.
    """

    def __init__(self, operators: List[Callable], rng: Random):
        """
        Initialize RVND with a list of local search operators.

        Args:
            operators: List of callable local search operators. Each operator should:
                      - Accept routes as input: operator(routes) -> (improved_routes, improved)
                      - Return tuple of (new_routes, improvement_flag)
                      - Return improved=True if an improving move was found
            rng: Random number generator for neighborhood selection.

        Example:
            >>> from random import Random
            >>> def relocate_op(routes):
            ...     # Apply relocate operator
            ...     return routes, False  # No improvement
            >>> def swap_op(routes):
            ...     # Apply swap operator
            ...     return routes, True   # Found improvement
            >>> rvnd = RVND(operators=[relocate_op, swap_op], rng=Random())
        """
        if not operators:
            raise ValueError("RVND requires at least one local search operator")

        self.operators = operators
        self.rng = rng
        self.num_operators = len(operators)

    def apply(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Apply RVND until a local optimum is reached with respect to all neighborhoods.

        Implements the core RVND loop:
        1. Initialize neighborhood list NL = [0, 1, ..., num_operators-1]
        2. While NL is not empty:
           a. Randomly select a neighborhood index n_idx from NL
           b. Apply the corresponding operator to the current routes
           c. If improvement found:
              - Update routes with the improved solution
              - Reset NL to [0, 1, ..., num_operators-1] and reshuffle
           d. If no improvement:
              - Remove n_idx from NL
        3. Return the locally optimal routes

        Args:
            routes: Current list of routes (list of node sequences).
                   Each route is a list of node indices.

        Returns:
            List[List[int]]: Improved routes representing a local optimum
                           with respect to all provided neighborhood operators.

        Example:
            >>> routes = [[1, 2, 3], [4, 5, 6]]
            >>> improved_routes = rvnd.apply(routes)
        """
        # Initialize the neighborhood list with all operator indices
        NL = list(range(self.num_operators))

        # Main RVND loop: continue until all neighborhoods exhausted
        while NL:
            # Randomly select a neighborhood index from the current list
            n_idx = self.rng.choice(NL)

            # Apply the selected operator
            operator = self.operators[n_idx]
            new_routes, improved = operator(routes)

            if improved:
                # Accept the improvement
                routes = new_routes

                # CRITICAL: Reset and reshuffle the entire neighborhood list
                # This is the key difference from standard VND
                NL = list(range(self.num_operators))
                self.rng.shuffle(NL)
            else:
                # No improvement found: remove this neighborhood from consideration
                NL.remove(n_idx)

        # Local optimum reached: all neighborhoods explored without improvement
        return routes

    def apply_with_stats(self, routes: List[List[int]]) -> tuple[List[List[int]], dict]:
        """
        Apply RVND with detailed statistics tracking.

        Same as apply() but also returns statistics about the search process.

        Args:
            routes: Current list of routes.

        Returns:
            Tuple of (improved_routes, stats) where stats is a dictionary containing:
                - 'iterations': Total number of neighborhood evaluations
                - 'improvements': Number of improvements found
                - 'operator_counts': List of how many times each operator was applied
                - 'operator_improvements': List of how many times each operator found improvements

        Example:
            >>> routes = [[1, 2, 3], [4, 5, 6]]
            >>> improved_routes, stats = rvnd.apply_with_stats(routes)
            >>> print(f"Found {stats['improvements']} improvements in {stats['iterations']} iterations")
        """
        # Initialize statistics
        iterations = 0
        improvements = 0
        operator_counts = [0] * self.num_operators
        operator_improvements = [0] * self.num_operators

        # Initialize the neighborhood list
        NL = list(range(self.num_operators))

        # Main RVND loop
        while NL:
            # Randomly select a neighborhood
            n_idx = self.rng.choice(NL)

            # Apply the selected operator
            operator = self.operators[n_idx]
            new_routes, improved = operator(routes)

            # Update statistics
            iterations += 1
            operator_counts[n_idx] += 1

            if improved:
                # Accept the improvement
                routes = new_routes
                improvements += 1
                operator_improvements[n_idx] += 1

                # Reset and reshuffle the neighborhood list
                NL = list(range(self.num_operators))
                self.rng.shuffle(NL)
            else:
                # Remove this neighborhood from consideration
                NL.remove(n_idx)

        # Compile statistics
        stats = {
            "iterations": iterations,
            "improvements": improvements,
            "operator_counts": operator_counts,
            "operator_improvements": operator_improvements,
        }

        return routes, stats
