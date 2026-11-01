"""Synapse models.

A synapse carries a weight, a conduction delay and a sign. Today only a static
scalar weight exists; plasticity rules (STDP, homeostatic scaling) and
multi-receptor kinetics are roadmap items.
"""

from .base import Synapse

__all__ = ["Synapse"]
