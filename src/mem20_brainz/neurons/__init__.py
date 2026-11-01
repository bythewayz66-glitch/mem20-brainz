"""Neuron models.

Holds the per-neuron state and dynamics. Today this is a single leaky
integrate-and-fire (LIF) placeholder; the roadmap calls for a model hierarchy
ranging from point neurons to multi-compartment morphologically detailed cells.
"""

from .base import LIFNeuron, Neuron

__all__ = ["LIFNeuron", "Neuron"]
