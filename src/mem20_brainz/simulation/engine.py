"""Reference simulation loop.

Semantics
---------
* Fixed timestep `dt` in milliseconds.
* Spikes emitted at step `k` are delivered to postsynaptic targets at step
  `k + ceil(delay_ms / dt)`.
* Synaptic input is summed as a current in nanoamps (placeholder units).

This is a correctness reference, not a performance path. It is O(synapses) per
step in pure Python/NumPy and is intended for thousands of neurons, not
billions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..connectome.sparse import SparseConnectome
from ..neurons.base import Neuron


@dataclass
class SimulationEngine:
    """Steps a population of neurons wired by a sparse connectome."""

    neurons: list[Neuron]
    connectome: SparseConnectome
    dt: float = 0.1
    t: float = 0.0
    spike_history: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.connectome.n_neurons and self.connectome.n_neurons != len(self.neurons):
            raise ValueError(
                f"connectome has {self.connectome.n_neurons} neurons, "
                f"population has {len(self.neurons)}"
            )

    def _deliver(self, spikes: list[int]) -> np.ndarray:
        """Sum synaptic current arriving this step from `spikes`."""
        current = np.zeros(len(self.neurons), dtype=np.float64)
        if not spikes:
            return current
        mask = np.isin(self.connectome.pre, spikes)
        if not mask.any():
            return current
        np.add.at(current, self.connectome.post[mask], self.connectome.weight[mask])
        return current

    def step(self, external_current: np.ndarray | None = None) -> list[int]:
        """Advance one timestep and return the ids of neurons that spiked."""
        prev_spikes = self.spike_history[-1] if self.spike_history else []
        current = self._deliver(prev_spikes)
        if external_current is not None:
            current = current + np.asarray(external_current, dtype=np.float64)

        spikes: list[int] = []
        for i, neuron in enumerate(self.neurons):
            if neuron.step(float(current[i]), self.dt, self.t):
                spikes.append(i)

        self.spike_history.append(spikes)
        self.t += self.dt
        return spikes

    def run(self, steps: int, external_current: np.ndarray | None = None) -> int:
        """Run `steps` timesteps; return total spike count."""
        total = 0
        for _ in range(steps):
            total += len(self.step(external_current))
        return total
