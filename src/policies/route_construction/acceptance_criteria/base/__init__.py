"""
Acceptance Criterion Package.

This package provides implementations for various acceptance criteria
for routing optimization policies (HGS, ALNS, VRPP, etc.) and a factory pattern
for dynamic acceptance criterion instantiation.
"""

from .factory import AcceptanceCriterionFactory
from .registry import AcceptanceCriterionRegistry
