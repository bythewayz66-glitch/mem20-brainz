"""Brain regions.

A region groups neurons and declares which cell types and connection rules it
contains. Today a region is a named container with a neuron-count budget; no
atlas parcellation is loaded.
"""

from .base import Region

__all__ = ["Region"]
