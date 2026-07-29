"""pareto_front.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import pareto_front
"""


class ParetoFront:
    """Maintains a set of non-dominated solutions."""

    def __init__(self, max_size=50):
        """
        Initialize Pareto front.

        Args:
            max_size: Maximum number of solutions to keep.
        """
        self.solutions = []
        self.max_size = max_size

    def add_solution(self, solution):
        """
        Add a solution to the Pareto front if non-dominated.

        Args:
            solution: ParetoSolution to add.

        Returns:
            bool: True if solution was added.
        """
        # Check dominance
        for existing in self.solutions:
            if existing.dominates(solution):
                return False

        # Remove dominated
        self.solutions = [s for s in self.solutions if not solution.dominates(s)]
        self.solutions.append(solution)

        if len(self.solutions) > self.max_size:
            self.solutions.pop(0)  # Simple pruning for now
        return True
