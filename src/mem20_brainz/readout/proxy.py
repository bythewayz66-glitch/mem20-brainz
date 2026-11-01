"""Affective readout: named states derived from measured simulation quantities.

WHAT THIS IS
------------
A fixed, documented mapping from *measurable* simulation state (firing rates,
excitatory/inhibitory drive, activity variance, population synchrony) to a small
set of *named* states (arousal, valence, surprise, confusion, fear-like).

WHAT THIS IS NOT
----------------
It is not a model of emotion, affect, feeling, or subjective experience. The
simulation has no feelings. Nothing here is a measurement of anything
psychological, because there is nothing psychological to measure: the substrate
is a leaky integrate-and-fire network with placeholder parameters and synthetic
connectivity.

Every readout therefore carries ``provenance="derived_proxy"`` and returns the
raw inputs and the formula it used, so a viewer can check the arithmetic rather
than trust a label. The names are chosen to be legible, not to be claims.

The mapping is a *design choice*, not a finding. See ``docs/readout.md``.

Units
-----
* rates: Hz (spikes per neuron per second of simulated time)
* drive: summed absolute synaptic weight arriving in one step (unitless
  placeholder weight units, as in :mod:`mem20_brainz.synapses.base`)
* synchrony: dimensionless Fano-factor-derived index in [0, 1]
* every readout ``value`` is dimensionless and clipped to [0, 1] except
  ``valence``, which is signed in [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The only provenance tag this module ever emits.
DERIVED_PROXY = "derived_proxy"

#: Small constant guarding divisions. Chosen so that a zero denominator yields a
#: saturated (not infinite, not NaN) readout.
_EPS = 1e-9

#: Display calibration for the synchrony index: the Fano-factor excess above 1
#: that maps to a full-scale reading. A display choice, not a measured constant.
#: See :func:`synchrony_index` and ``docs/readout.md``.
SYNCHRONY_SCALE = 1.0


@dataclass(frozen=True)
class Readout:
    """One named state, with everything needed to audit it.

    Attributes
    ----------
    name:
        Stable machine name, e.g. ``"arousal"``.
    value:
        The derived number. Dimensionless.
    unit:
        Human-readable unit string. Always ``"dimensionless"`` here.
    inputs:
        The raw measured quantities the value was computed from, with units in
        the key names. This is the audit trail.
    formula:
        The arithmetic, as a string, in terms of the ``inputs`` keys.
    provenance:
        Always :data:`DERIVED_PROXY`. Present so a consumer can filter on it
        without reading prose.
    interpretation:
        One sentence saying what the number tracks, phrased as a proxy.
    """

    name: str
    value: float
    unit: str
    inputs: dict[str, float]
    formula: str
    provenance: str = DERIVED_PROXY
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "inputs": self.inputs,
            "formula": self.formula,
            "provenance": self.provenance,
            "interpretation": self.interpretation,
        }


@dataclass
class ReadoutInputs:
    """Measured quantities a readout pass consumes.

    All fields are computed from the simulation, never invented. The runner
    fills this in from the frame it just produced.
    """

    #: Population mean firing rate over the recent window, in Hz.
    rate_hz: float = 0.0
    #: The stated reference rate the arousal ratio is taken against, in Hz.
    baseline_rate_hz: float = 5.0
    #: Summed absolute weight of excitatory synapses that fired into the
    #: population this step (placeholder weight units).
    exc_drive: float = 0.0
    #: Same, for inhibitory synapses.
    inh_drive: float = 0.0
    #: Running expectation of ``rate_hz`` (exponential moving average), in Hz.
    expected_rate_hz: float = 0.0
    #: Per-region firing rates, in Hz. Used for the coherence term.
    region_rates_hz: list[float] = field(default_factory=list)
    #: Fano-factor-derived population synchrony index in [0, 1].
    synchrony: float = 0.0
    #: Synchrony restricted to the configured threat pathway, in [0, 1].
    threat_synchrony: float = 0.0
    #: EMA of the arousal readout, in [0, 1]. Gives "sustained".
    sustained_arousal: float = 0.0


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def synchrony_index(spike_counts_per_step: list[int], n_neurons: int) -> float:
    """Population synchrony from a window of per-step spike counts.

    Uses the Fano factor of the population spike count, ``F = var/mean``. An
    asynchronous population with independent neurons has ``F ~= 1``; a fully
    synchronised one has ``F ~= n_neurons``. We map that onto [0, 1] with

        synchrony = clip((F - 1) / SYNCHRONY_SCALE, 0, 1)

    so ``F = 1`` (asynchronous) gives 0 and ``F >= 1 + SYNCHRONY_SCALE`` gives 1.

    ``SYNCHRONY_SCALE`` is a **display calibration, not a measured constant**.
    It is set to 1.0 because the demo network's observed Fano factor sits in
    roughly 1.0-1.5 across its operating range: a divisor of 9 (the obvious
    "n_neurons" choice) would map that onto 0.00-0.05 and the gauge would never
    move. The value is stated here, and in ``docs/readout.md``, so nobody
    mistakes it for a physiological quantity. A different network would need a
    different calibration.

    Returns 0.0 for windows shorter than two steps or with no spikes, because
    variance is undefined there.
    """
    if n_neurons <= 0 or len(spike_counts_per_step) < 2:
        return 0.0
    counts = np.asarray(spike_counts_per_step, dtype=np.float64)
    mean = float(counts.mean())
    if mean <= 0.0:
        return 0.0
    fano = float(counts.var()) / mean
    return _clip01((fano - 1.0) / SYNCHRONY_SCALE)


def arousal(inp: ReadoutInputs) -> Readout:
    """Normalised population firing rate against a stated baseline.

    ``ratio = rate_hz / baseline_rate_hz``; ``value = clip(ratio / 2, 0, 1)``, so
    0.5 means "at baseline" and 1.0 means "at or above twice baseline". The
    factor of 2 is a display scale, not a physiological constant.
    """
    baseline = max(inp.baseline_rate_hz, _EPS)
    ratio = inp.rate_hz / baseline
    return Readout(
        name="arousal",
        value=_clip01(ratio / 2.0),
        unit="dimensionless",
        inputs={
            "rate_hz": float(inp.rate_hz),
            "baseline_rate_hz": float(inp.baseline_rate_hz),
            "ratio": float(ratio),
        },
        formula="clip((rate_hz / baseline_rate_hz) / 2, 0, 1)",
        interpretation="How far population firing sits above its stated baseline.",
    )


def valence(inp: ReadoutInputs) -> Readout:
    """Signed balance of excitatory versus inhibitory drive.

    ``value = (exc - inh) / (exc + inh)``, in [-1, 1]. Positive means the
    arriving synaptic drive this step was net excitatory, negative net
    inhibitory. This is a *drive balance*, not a preference, reward, or
    approach/avoid tendency: the network has no goals.
    """
    total = inp.exc_drive + inp.inh_drive
    value = 0.0 if total <= _EPS else (inp.exc_drive - inp.inh_drive) / total
    return Readout(
        name="valence",
        value=float(max(-1.0, min(1.0, value))),
        unit="dimensionless",
        inputs={
            "exc_drive": float(inp.exc_drive),
            "inh_drive": float(inp.inh_drive),
            "total_drive": float(total),
        },
        formula="(exc_drive - inh_drive) / (exc_drive + inh_drive), clipped to [-1, 1]",
        interpretation="Net sign of arriving synaptic drive: excitatory (+) or inhibitory (-).",
    )


def surprise(inp: ReadoutInputs) -> Readout:
    """Deviation of current activity from a running expectation.

    ``value = clip(|rate - expected| / (expected + baseline), 0, 1)``. The
    denominator is the expectation plus the baseline rate, which keeps the
    quantity bounded and scale-free when the expectation is near zero. This is a
    prediction-error *proxy*: the "expectation" is an exponential moving average
    of the population rate, not a learned predictive model.
    """
    denom = max(inp.expected_rate_hz + inp.baseline_rate_hz, _EPS)
    deviation = abs(inp.rate_hz - inp.expected_rate_hz)
    return Readout(
        name="surprise",
        value=_clip01(deviation / denom),
        unit="dimensionless",
        inputs={
            "rate_hz": float(inp.rate_hz),
            "expected_rate_hz": float(inp.expected_rate_hz),
            "baseline_rate_hz": float(inp.baseline_rate_hz),
            "abs_deviation_hz": float(deviation),
        },
        formula="clip(|rate_hz - expected_rate_hz| / (expected_rate_hz + baseline_rate_hz), 0, 1)",
        interpretation="How far current firing departs from its own running average.",
    )


def confusion(inp: ReadoutInputs) -> Readout:
    """Dispersion of activity across regions: high variance, low coherence.

    ``value = clip(CV(region_rates), 0, 1)`` where ``CV = std / mean``. A
    population where every region fires at the same rate scores 0; one where
    regions disagree wildly scores high. This measures *spatial incoherence of
    activity*, nothing more.
    """
    rates = np.asarray(inp.region_rates_hz, dtype=np.float64)
    if rates.size < 2:
        cv = 0.0
        mean = float(rates.mean()) if rates.size else 0.0
        std = 0.0
    else:
        mean = float(rates.mean())
        std = float(rates.std())
        cv = 0.0 if mean <= _EPS else std / mean
    return Readout(
        name="confusion",
        value=_clip01(cv),
        unit="dimensionless",
        inputs={
            "region_rate_mean_hz": mean,
            "region_rate_std_hz": std,
            "region_count": float(rates.size),
            "coefficient_of_variation": float(cv),
        },
        formula="clip(std(region_rates_hz) / mean(region_rates_hz), 0, 1)",
        interpretation="How unevenly activity is distributed across regions.",
    )


def fear_like(inp: ReadoutInputs) -> Readout:
    """Sustained high arousal combined with high synchrony in a threat pathway.

    ``value = clip(sustained_arousal * threat_synchrony, 0, 1)``. Both factors
    must be high for the product to be high, which is the whole point: neither
    arousal alone nor synchrony alone produces a score.

    The name is a label for a *pattern of measurements*, chosen because the
    pattern is the one a threat-detection circuit would be expected to produce.
    It is not a claim that the network is afraid. The "threat pathway" is a
    region the scenario config designates; it has no biological identity.
    """
    return Readout(
        name="fear_like",
        value=_clip01(inp.sustained_arousal * inp.threat_synchrony),
        unit="dimensionless",
        inputs={
            "sustained_arousal": float(inp.sustained_arousal),
            "threat_synchrony": float(inp.threat_synchrony),
        },
        formula="clip(sustained_arousal * threat_synchrony, 0, 1)",
        interpretation="Co-occurrence of sustained high firing and threat-pathway synchrony.",
    )


#: The readouts computed on every frame, in display order.
ALL_READOUTS = (arousal, valence, surprise, confusion, fear_like)


def compute_all(inp: ReadoutInputs) -> list[Readout]:
    """Compute every readout for one frame."""
    return [fn(inp) for fn in ALL_READOUTS]


def to_dicts(readouts: list[Readout]) -> list[dict[str, Any]]:
    """Serialise readouts for the wire."""
    return [r.to_dict() for r in readouts]
