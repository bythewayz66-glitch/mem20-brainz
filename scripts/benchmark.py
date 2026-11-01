#!/usr/bin/env python3
"""Benchmark helper for the reference engine.

Measures synapses/second for the pure-Python stepping loop so future work has a
baseline to beat. This is a micro-benchmark, not a brain-scale benchmark.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mem20_brainz.connectome import SparseConnectome
from mem20_brainz.neurons import LIFNeuron
from mem20_brainz.simulation import SimulationEngine


def build_random(n_neurons: int, p: float, seed: int = 0) -> SparseConnectome:
    rng = np.random.default_rng(seed)
    mask = rng.random((n_neurons, n_neurons)) < p
    pre, post = np.nonzero(mask)
    return SparseConnectome.from_arrays(n_neurons=n_neurons, pre=pre, post=post)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    conn = build_random(args.neurons, args.p)
    neurons = [LIFNeuron(id=i) for i in range(args.neurons)]
    engine = SimulationEngine(neurons=neurons, connectome=conn, dt=0.1)
    drive = np.full(args.neurons, 20.0)

    start = time.perf_counter()
    spikes = engine.run(steps=args.steps, external_current=drive)
    elapsed = time.perf_counter() - start

    syn_per_step = conn.n_synapses
    print(f"neurons:        {args.neurons}")
    print(f"synapses:       {conn.n_synapses}")
    print(f"steps:          {args.steps}")
    print(f"spikes:         {spikes}")
    print(f"elapsed:        {elapsed:.3f} s")
    print(f"synapses/s:     {syn_per_step * args.steps / elapsed:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
