"""Wire format for streamed simulation frames.

This module owns the *schema* and nothing else. It contains no simulation
logic: it takes already-computed arrays and packs them for the wire. That
separation is the point of ADR-0007 — the server orchestrates and serialises,
the engine computes.

Design constraints
------------------
* A frame must be small enough to send at interactive rates. Membrane potential
  is therefore quantised to 8 bits over a declared range, and optionally binned
  when the population is large. The declared range and bin size travel with the
  frame so a consumer can reconstruct the mapping exactly.
* A frame must be self-describing about honesty. Every frame carries the
  scenario provenance and the readout provenance tags, so a viewer never has to
  trust prose to know whether data is measured or synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Bump when the frame shape changes incompatibly. The frontend checks this.
FRAME_SCHEMA_VERSION = 1

#: Above this many neurons, membrane potential is averaged into bins before
#: quantisation, so the payload stays bounded regardless of population size.
DEFAULT_MAX_V_VALUES = 2000

#: Quantisation levels for membrane potential (8-bit).
V_LEVELS = 256


@dataclass
class RegionActivity:
    """Aggregate activity for one region in one step."""

    name: str
    n_neurons: int
    #: Spikes in this step divided by neuron count, in Hz.
    rate_hz: float
    #: Fraction of the region's neurons that spiked this step, in [0, 1].
    active_fraction: float
    #: Mean membrane potential of the region this step, in mV.
    mean_v_mv: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_neurons": self.n_neurons,
            "rate_hz": round(self.rate_hz, 4),
            "active_fraction": round(self.active_fraction, 6),
            "mean_v_mv": round(self.mean_v_mv, 3),
        }


@dataclass
class Frame:
    """One simulation step, ready to serialise."""

    step: int
    t_ms: float
    dt_ms: float
    #: Quantised membrane potential, 0..255, one entry per value (or per bin).
    v_q: list[int]
    #: The [min, max] mV range the quantisation maps onto.
    v_range: list[float]
    #: 1 means one value per neuron; >1 means neurons were averaged in groups.
    v_bin_size: int
    #: Indices of neurons that spiked this step.
    spikes: list[int]
    regions: list[RegionActivity]
    readouts: list[dict[str, Any]]
    status: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FRAME_SCHEMA_VERSION,
            "step": self.step,
            "t_ms": round(self.t_ms, 4),
            "dt_ms": self.dt_ms,
            "v_q": self.v_q,
            "v_range": self.v_range,
            "v_bin_size": self.v_bin_size,
            "spikes": self.spikes,
            "regions": [r.to_dict() for r in self.regions],
            "readouts": self.readouts,
            "status": self.status,
            "provenance": self.provenance,
        }


def quantise_v(v: np.ndarray, v_min: float, v_max: float) -> list[int]:
    """Map membrane potential in mV onto 0..255 over ``[v_min, v_max]``.

    Values outside the range are clipped, not wrapped: a clipped value is
    visibly saturated, whereas a wrapped one would be a lie about the voltage.
    """
    if v_max <= v_min:
        return [0] * int(v.size)
    scaled = (np.asarray(v, dtype=np.float64) - v_min) / (v_max - v_min)
    return np.clip(np.rint(scaled * (V_LEVELS - 1)), 0, V_LEVELS - 1).astype(np.uint8).tolist()


def bin_v(v: np.ndarray, max_values: int) -> tuple[np.ndarray, int]:
    """Average ``v`` into contiguous bins so at most ``max_values`` remain.

    Returns the binned array and the bin size used (1 when no binning happened).
    Binning is a display reduction: it is stated in the frame so nobody mistakes
    a binned value for a single neuron's voltage.
    """
    n = int(v.size)
    if n <= max_values or max_values <= 0:
        return np.asarray(v, dtype=np.float64), 1
    bin_size = int(np.ceil(n / max_values))
    n_bins = int(np.ceil(n / bin_size))
    padded = np.full(n_bins * bin_size, np.nan, dtype=np.float64)
    padded[:n] = v
    binned = np.nanmean(padded.reshape(n_bins, bin_size), axis=1)
    return binned, bin_size


def build_frame(
    *,
    step: int,
    t_ms: float,
    dt_ms: float,
    v: np.ndarray,
    spikes: list[int],
    region_activity: list[RegionActivity],
    readouts: list[dict[str, Any]],
    status: dict[str, Any],
    provenance: dict[str, Any],
    v_min: float = -80.0,
    v_max: float = -40.0,
    max_v_values: int = DEFAULT_MAX_V_VALUES,
) -> Frame:
    """Pack one step into a :class:`Frame`."""
    binned, bin_size = bin_v(v, max_v_values)
    return Frame(
        step=step,
        t_ms=t_ms,
        dt_ms=dt_ms,
        v_q=quantise_v(binned, v_min, v_max),
        v_range=[float(v_min), float(v_max)],
        v_bin_size=bin_size,
        spikes=spikes,
        regions=region_activity,
        readouts=readouts,
        status=status,
        provenance=provenance,
    )
