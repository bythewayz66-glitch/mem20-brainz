"""Synapse primitives.

A `Synapse` is a directed edge from a presynaptic neuron to a postsynaptic
neuron. Weights are unitless placeholders: no calibration against measured
postsynaptic potentials has been performed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Synapse:
    """A single directed synaptic connection."""

    pre: int
    post: int
    weight: float = 1.0
    delay_ms: float = 1.0
    sign: int = 1  # +1 excitatory, -1 inhibitory

    @property
    def effective_weight(self) -> float:
        """Weight with sign applied."""
        return self.weight * self.sign
