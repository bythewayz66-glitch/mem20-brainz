# Roadmap

Phases are sequential: each one's exit criterion is the next one's entry
condition. Dates are deliberately absent — this is a side project, and a
schedule that slips is worse than no schedule. What matters is the exit
criterion.

**Current phase: 1 — Vectorised core (in progress).**

---

## Phase 0 — Scaffold *(complete)*

**Goal:** a repository that is honest, installable, testable, and has frozen
interfaces.

- [x] Public repo, MIT license, Python-focused `.gitignore` with large-data rules
- [x] Installable package (`pyproject.toml`, `src/` layout)
- [x] Module skeleton: `neurons`, `synapses`, `regions`, `connectome`,
      `simulation`, `io`, `viz`
- [x] Reference `LIFNeuron`, `Synapse`, `Region`, `SparseConnectome`,
      `SimulationEngine`
- [x] Smoke test suite
- [x] JSON Schemas for fixtures, neuron records, dataset manifests
- [x] Mission README with an explicit implemented/aspirational split
- [x] ADR directory populated (`docs/adr/0001`–`0009`) with an index
- [x] CI: `pytest` + `ruff` on push and pull request (Python 3.11 and 3.12)

**Exit criterion:** `pip install -e ".[dev]"` and `pytest` pass on a clean
checkout, and the interfaces in `ARCHITECTURE.md` are stable enough that Phase 1
can be built behind them without changing callers. **Met.**

---

## Phase 1 — Vectorised core *(in progress)*

**Goal:** prove the engine can be fast, without changing semantics.

- [x] Struct-of-arrays neuron state (`LIFState`); per-neuron Python objects
      removed from the hot path
- [x] `Backend` protocol: `step(state, i_inj, dt, t) -> spike_mask`
- [x] NumPy backend (vectorised state update), **bit-identical** to the reference
- [x] Equivalence tests: bit-identical spikes on a 10⁴-neuron fixed-seed case
- [x] Numba backend (JIT inner loop), **bit-identical** to the reference
- [x] JAX backend (JIT; enables autodiff for parameter fitting) —
      tolerance-gated, not bit-identical (ADR-0008)
- [x] Gradient-based parameter fitting via a straight-through estimator
      (`scripts/fit_jax_example.py`; surrogate gradient, ADR-0008)
- [x] CSR connectome view indexed by presynaptic neuron
- [x] Event-driven delivery: only touch edges whose `pre` fired
      (tolerance-gated, not bit-identical — ADR-0006)
- [x] Segmented reduction (`np.bincount`) replacing the `np.add.at` scatter —
      turns the work reduction into a wall-clock win (1.3×–2.7×, ADR-0006)
- [x] Numeric precision decision: float64 default, float32 as a tested opt-in
      (ADR-0007)
- [x] Benchmark harness with a recorded baseline (synapses/s, neurons/s)
- [x] Live observability layer: streaming server, frame schema, React frontend,
      headless mode (ADR-0009)
- [x] Affective readout proxies with inputs, formula and `derived_proxy` tag
      ([`docs/readout.md`](docs/readout.md))

**Exit criterion:** 10⁶ LIF neurons for 1 second of biological time (10,000
steps) on a single GPU, with spike output bit-identical to the reference
implementation on a 10⁴-neuron test case.

**Note on CSR event-driven delivery:** it is *not* bit-identical to the
reference, because summing incoming weights in CSR order rather than edge order
is a different order of float additions. It is gated on a tolerance-based
equivalence test with a stated bound (ADR-0006). Measured: the delivery sums
differ by up to 5.4 × 10⁻⁹ relative, and no spike flips were observed in any