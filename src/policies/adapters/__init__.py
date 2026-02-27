"""
Policy Adapters Package.

This package provides adapter implementations for various routing
optimization policies (HGS, ALNS, VRPP, etc.) and a factory pattern
for dynamic policy instantiation.
"""

from .factory import IPolicy, PolicyFactory, PolicyRegistry
