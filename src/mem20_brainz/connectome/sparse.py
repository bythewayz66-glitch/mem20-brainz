"""Sparse connectome in coordinate-list form.

Design notes
------------
* Storage is columnar (four parallel arrays) rather than a list of objects, so a
  hundred trillion edges is at least *conceivable* to represent: 4 arrays of
  int32/int32/float32/float32 is ~16 bytes per synapse.
* At 100T synapses that is still ~1.6 TB of RAM, which is why the roadmap puts
  out-of-core / sharded storage ahead of any full-scale run.
* This class is a correctness-first reference implementation. It is not fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SparseConnectome:
    """Coordinate-list sparse connectome."""

    n_neurons: int = 0
    pre: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    post: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    weight: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    delay_ms: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def __post_init__(self) -> None:
        lengths = {len(self.pre), len(self.post), len(self.weight), len(self.delay_ms)}
        if len(lengths) != 1:
            raise ValueError("pre/post/weight/delay arrays must have equal length")

    @property
    def n_synapses(self) -> int:
        return int(len(self.pre))

    def add(self, pre: int, post: int, weight: float = 1.0, delay_ms: float = 1.0) -> None:
        """Append one synapse. O(n) per call — use :meth:`from_arrays` in bulk."""
        self.pre = np.append(self.pre, np.int32(pre))
        self.post = np.append(self.post, np.int32(post))
        self.weight = np.append(self.weight, np.float32(weight))
        self.delay_ms = np.append(self.delay_ms, np.float32(delay_ms))

    @classmethod
    def from_arrays(
        cls,
        n_neurons: int,
        pre: np.ndarray,
        post: np.ndarray,
        weight: np.ndarray | None = None,
        delay_ms: np.ndarray | None = None,
    ) -> "SparseConnectome":
        pre = np.asarray(pre, dtype=np.int32)
        post = np.asarray(post, dtype=np.int32)
        n = len(pre)
        weight = np.ones(n, dtype=np.float32) if weight is None else np.asarray(weight, np.float32)
        delay_ms = (
            np.ones(n, dtype=np.float32) if delay_ms is None else np.asarray(delay_ms, np.float32)
        )
        return cls(n_neurons=n_neurons, pre=pre, post=post, weight=weight, delay_ms=delay_ms)

    def out_edges(self, neuron_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (post, weight, delay) for all synapses leaving `neuron_id`."""
        mask = self.pre == neuron_id
        return self.post[mask], self.weight[mask], self.delay_ms[mask]

    def in_degree(self) -> np.ndarray:
        """Number of incoming synapses per neuron."""
        return np.bincount(self.post, minlength=self.n_neurons)

    def to_dict(self) -> dict:
        return {
            "n_neurons": self.n_neurons,
            "pre": self.pre.tolist(),
            "post": self.post.tolist(),
            "weight": self.weight.tolist(),
            "delay_ms": self.delay_ms.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SparseConnectome":
        return cls.from_arrays(
            n_neurons=int(data["n_neurons"]),
            pre=data["pre"],
            post=data["post"],
            weight=data.get("weight"),
            delay_ms=data.get("delay_ms"),
        )
