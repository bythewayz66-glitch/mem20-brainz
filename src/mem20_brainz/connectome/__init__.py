"""Connectome storage.

Sparse, index-based storage for synaptic connectivity. The in-memory format is
a coordinate list (pre, post, weight, delay) that can be converted to CSR-like
per-neuron adjacency. No measured connectome is bundled with this repository.
"""

from .sparse import SparseConnectome

__all__ = ["SparseConnectome"]
