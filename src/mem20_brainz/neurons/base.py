"""Neuron primitives.

`Neuron` is the minimal interface every neuron model must satisfy: it owns an
identifier, a membrane state, and a `step` method that advances one timestep and
reports whether the cell emitted a spike.

`LIFNeuron` is a deliberately simple leaky integrate-and-fire placeholder. It is
NOT biologically validated and is not intended to reproduce any measured
firing statistics. It exists so the simulation loop, connectome and I/O layers
have something concrete to run against.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Neuron:
    """Base neuron: identity plus membrane potential.

    Subclasses override :meth:`step` with their own dynamics.
    """

    id: int
    v: float = -70.0

    def step(self, i_inj: float, dt: float, t: float) -> bool:
        """Advance one timestep. Return True if the neuron spiked."""
        raise NotImplementedError("Neuron subclasses must implement step()")


@dataclass
class LIFNeuron(Neuron):
    """Leaky integrate-and-fire placeholder neuron.

    Parameters are textbook defaults in millivolts / milliseconds / megaohms.
    They are placeholders, not fitted values.
    """

    v_rest: float = -70.0
    v_reset: float = -75.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    r_m: float = 10.0
    refractory_ms: float = 2.0
    v: float = -70.0
    #: Model tier, declared per neuron and never defaulted implicitly (ADR-0002).
    tier: str = "tier0_point_lif"
    _refractory_until: float = field(default=0.0, repr=False)

    def step(self, i_inj: float, dt: float, t: float) -> bool:
        if t < self._refractory_until:
            return False

        dv = (-(self.v - self.v_rest) + self.r_m * i_inj) / self.tau_m
        self.v += dv * dt

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self._refractory_until = t + self.refractory_ms
            return True
        return False
