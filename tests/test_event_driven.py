"""Event-driven delivery: tolerance-based equivalence, not bit-identity.

Event-driven delivery sums incoming weights in **CSR order** (grouped by
presynaptic neuron) rather than **edge-list order**. Floating-point addition is
not associative, so the two orders produce different sums. This is measured, not
assumed: ``test_delivery_sums_are_not_bit_identical`` asserts that the two
paths disagree in the last bits on a high-fan-in case (max relative difference
~5e-9 with float64 weights).

What that does *not* do, empirically, is flip a spike. Across every regime
tested here — subthreshold, near-threshold, refractory-dominated, 10^4 neurons —
the spike output was **identical**, because a ~1e-9 relative perturbation in the
membrane potential is far smaller than the margin a neuron needs to cross
threshold within one 0.1 ms step. So the honest statement is:

* the delivery sums are **not** bit-identical (proven below), and
* the spike output **was** identical in every regime tested (also asserted).

That second fact is an observation about this LIF model and these parameters,
not a guarantee. A different model, a smaller timestep, or a neuron parked
exactly on threshold could flip. So the gate is a tolerance test, not a
bit-identity claim — and the tolerance is stated explicitly rather than being
implicit in a loose assertion.

ADR-0006 records the decision.

The tolerance is stated in two parts, because they measure different things:

* ``SPIKE_COUNT_TOLERANCE`` — relative difference in total spike count. This is
  the number that matters scientifically: a run whose spike count drifts by more
  than a fraction of a percent is a different run.
* ``PER_STEP_AGREEMENT`` — fraction of timesteps whose spike set is *identical*.
  This is the strict per-step check.

One case *is* exact by construction and is asserted as such: when the edge list
is already grouped by ``pre``, CSR order equals edge-list order, so the sums are
identical and the spikes are bit-identical. That case is a sanity check on the
CSR build, not a claim about the general case.
"""

from __future__ import annotations

import numpy as np
import pytest

from mem20_brainz.connectome import CSRConnectome, SparseConnectome
from mem20_brainz.neurons import LIFNeuron
from mem20_brainz.simulation import (
    EventDrivenEngine,
    LIFState,
    SimulationEngine,
    VectorizedEngine,
)

#: Relative tolerance on total spike count between the reference and the
#: event-driven path. 1% is deliberately loose: the point of the test is to
#: catch a *broken* delivery path (which produces wildly different activity),
#: not to pin down last-bit drift, which is expected and unavoidable.
SPIKE_COUNT_TOLERANCE = 0.01
