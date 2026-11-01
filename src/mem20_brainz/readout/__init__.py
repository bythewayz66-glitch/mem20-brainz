"""Affective readout — explicitly a proxy layer, not a model of feeling.

The simulation has no feelings. This package maps measurable simulation state
(firing rates, excitatory/inhibitory drive, activity dispersion, population
synchrony) onto a small set of *named* states so a viewer can watch the network
respond to input. Every value is tagged ``provenance="derived_proxy"`` and
carries the raw inputs and formula it came from.

Read ``docs/readout.md`` before using any of these numbers in a claim.
"""

from .proxy import (
    ALL_READOUTS,
    DERIVED_PROXY,
    Readout,
    ReadoutInputs,
    arousal,
    compute_all,
    confusion,
    fear_like,
    surprise,
    synchrony_index,
    to_dicts,
    valence,
)

__all__ = [
    "ALL_READOUTS",
    "DERIVED_PROXY",
    "Readout",
    "ReadoutInputs",
    "arousal",
    "compute_all",
    "confusion",
    "fear_like",
    "surprise",
    "synchrony_index",
    "to_dicts",
    "valence",
]
