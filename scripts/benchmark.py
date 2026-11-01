#!/usr/bin/env python3
"""Benchmark harness: neurons/s and synapses/s for every engine path.

What it measures
----------------
Four paths, on identical inputs (same seed, same connectome, same drive):

* ``reference``     — ``SimulationEngine``, the per-neuron Python loop
* ``numpy``         — ``VectorizedEngine`` + ``NumPyBackend``
* ``numba``         — ``VectorizedEngine`` + ``NumbaBackend`` (skipped if absent)
* ``event_driven``  — ``EventDrivenEngine`` over a CSR connectome

Two metrics, because they answer different questions:

* **synapses/s** — edges processed per second. For the scan-based paths this is
  ``n_synapses * steps / elapsed``, i.e. work *attempted*. For the event-driven
  path it is ``edges_visited / elapsed``, i.e. work *actually done*. Comparing
  the two directly is the whole point of the crossover measurement, so the JSON
  records both ``synapses_attempted`` and ``synapses_visited``.
* **neurons/s** — ``n_neurons * steps / elapsed``. The number the roadmap's
  Phase 1 exit criterion is phrased in.

Reproducibility
---------------
Results are written to a committed JSON file with the machine and environment
metadata attached, so a number in the docs can always be traced to a run. The
``--check`` mode fails if the committed baseline no longer matches the current
code, which is what stops the docs from silently drifting away from reality.

Honesty
-------
These numbers come from **one specific machine** (recorded in the JSON). They
are not a claim about whole-brain performance, and they are not a comparison
with other simulators. See ``docs/benchmarks/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from mem20_brainz import __version__
from mem20_brainz.connectome import CSRConnectome, SparseConnectome
from mem20_brainz.neurons import LIFNeuron
from mem20_brainz.simulation import (
    EventDrivenEngine,
    LIFState,
    NumPyBackend,
    SimulationEngine,
    VectorizedEngine,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "benchmarks" / "baseline.json"

#: Fixed benchmark configuration. Changing any of these invalidates the
#: committed baseline, which is exactly what ``--check`` is for.
CONFIG = {
    "n_neurons": 2000,
    "p": 0.01,
    "steps": 200,
    "drive": 45.0,
    "dt": 0.1,
    "seed": 20261101,
}


def build_connectome(n_neurons: int, p: float, seed: int) -> SparseConnectome:
    """Erdos-Renyi edge list, sampled directly (never a dense matrix)."""
    rng = np.random.default_rng(seed)
    n_edges = round(n_neurons * n_neurons * p)
    pre = rng.integers(0, n_neurons, size=n_edges)
    post = rng.integers(0, n_neurons, size=n_edges)
    keys = pre.astype(np.int64) * n_neurons + post.astype(np.int64)
    _, unique = np.unique(keys, return_index=True)
    pre, post = pre[unique], post[unique]
    weights = rng.uniform(0.5, 2.0, size=len(pre)).astype(np.float32)
    delays = np.full(len(pre), 1.0, dtype=np.float32)
    return SparseConnectome.from_arrays(
        n_neurons=n_neurons, pre=pre, post=post, weight=weights, delay_ms=delays
    )


def _time(fn) -> tuple[float, int]:
    """Run ``fn`` once, return (elapsed_seconds, spike_count)."""
    start = time.perf_counter()
    spikes = fn()
    return time.perf_counter() - start, spikes


def run_benchmarks(config: dict) -> dict:
    n = config["n_neurons"]
    steps = config["steps"]
    conn = build_connectome(n, config["p"], config["seed"])
    external = np.full(n, config["drive"], dtype=np.float64)
    n_synapses = conn.n_synapses

    results: dict[str, dict] = {}

    # --- reference -------------------------------------------------------
    def _reference():
        engine = SimulationEngine(
            neurons=[LIFNeuron(id=i) for i in range(n)], connectome=conn, dt=config["dt"]
        )
        return engine.run(steps=steps, external_current=external)

    elapsed, spikes = _time(_reference)
    results["reference"] = {
        "elapsed_s": elapsed,
        "spikes": spikes,
        "synapses_attempted": n_synapses * steps,
        "synapses_visited": n_synapses * steps,
        "synapses_per_s": n_synapses * steps / elapsed,
        "neurons_per_s": n * steps / elapsed,
    }

    # --- numpy -----------------------------------------------------------
    def _numpy():
        engine = VectorizedEngine(
            state=LIFState.uniform(n), connectome=conn, dt=config["dt"], backend=NumPyBackend()
        )
        return engine.run(steps=steps, external_current=external)

    elapsed, spikes = _time(_numpy)
    results["numpy"] = {
        "elapsed_s": elapsed,
        "spikes": spikes,
        "synapses_attempted": n_synapses * steps,
        "synapses_visited": n_synapses * steps,
        "synapses_per_s": n_synapses * steps / elapsed,
        "neurons_per_s": n * steps / elapsed,
    }

    # --- numba (optional) ------------------------------------------------
    try:
        from mem20_brainz.simulation.numba_backend import NumbaBackend

        backend = NumbaBackend()
        # Warm the JIT so compilation time is not counted as runtime.
        backend.step(LIFState.uniform(4), np.zeros(4), config["dt"], 0.0)

        def _numba():
            engine = VectorizedEngine(
                state=LIFState.uniform(n), connectome=conn, dt=config["dt"], backend=backend
            )
            return engine.run(steps=steps, external_current=external)

        elapsed, spikes = _time(_numba)
        results["numba"] = {
            "elapsed_s": elapsed,
            "spikes": spikes,
            "synapses_attempted": n_synapses * steps,
            "synapses_visited": n_synapses * steps,
            "synapses_per_s": n_synapses * steps / elapsed,
            "neurons_per_s": n * steps / elapsed,
        }
    except ImportError:
        results["numba"] = {"skipped": "numba not installed"}

    # --- event-driven ----------------------------------------------------
    csr = CSRConnectome.from_sparse(conn)

    def _event_driven():
        engine = EventDrivenEngine(state=LIFState.uniform(n), connectome=csr, dt=config["dt"])
        spikes = engine.run(steps=steps, external_current=external)
        return spikes, engine.edges_visited

    start = time.perf_counter()
    spikes, visited = _event_driven()
    elapsed = time.perf_counter() - start
    results["event_driven"] = {
        "elapsed_s": elapsed,
        "spikes": spikes,
        "synapses_attempted": n_synapses * steps,
        "synapses_visited": visited,
        "synapses_per_s": visited / elapsed,
        "neurons_per_s": n * steps / elapsed,
        "edge_hit_rate": visited / (n_synapses * steps),
    }

    return {
        "config": config,
        "n_synapses": n_synapses,
        "results": results,
    }


#: Sizes swept by the crossover measurement. The question is not "is
#: event-driven faster" but "at what population size does it start to win",
#: because the answer depends on the fixed per-step overhead of the CSR lookup
#: versus the per-edge cost of the scan.
CROSSOVER_SIZES = (2_000, 5_000, 10_000, 20_000, 40_000)


def measure_crossover(steps: int = 100, drive: float = 45.0) -> list[dict]:
    """Sweep population size and record scan vs event-driven wall time.

    ``p`` is scaled as ``0.02 * (2000 / n)`` so the mean out-degree stays roughly
    constant and the comparison is about population size, not density.
    """
    rows = []
    for n in CROSSOVER_SIZES:
        p = 0.02 * (2000 / n)
        conn = build_connectome(n, p, seed=1)
        external = np.full(n, drive, dtype=np.float64)

        scan = VectorizedEngine(
            state=LIFState.uniform(n), connectome=conn, dt=0.1, backend=NumPyBackend()
        )
        start = time.perf_counter()
        scan.run(steps=steps, external_current=external)
        scan_s = time.perf_counter() - start

        csr = CSRConnectome.from_sparse(conn)
        ev = EventDrivenEngine(state=LIFState.uniform(n), connectome=csr, dt=0.1)
        start = time.perf_counter()
        ev.run(steps=steps, external_current=external)
        ev_s = time.perf_counter() - start

        rows.append(
            {
                "n_neurons": n,
                "n_synapses": conn.n_synapses,
                "scan_s": scan_s,
                "event_driven_s": ev_s,
                "speedup": scan_s / ev_s,
                "edge_hit_rate": ev.edges_visited / (conn.n_synapses * steps),
            }
        )
    return rows


def environment() -> dict:
    """Machine and environment metadata, so a number can be traced to a run."""
    try:
        cpu = subprocess.run(
            ["bash", "-c", "lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        cpu = "unknown"

    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu or platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "numba": numba_version,
        "mem20_brainz": __version__,
    }


def render_markdown(payload: dict) -> str:
    """Render the JSON payload as a Markdown table."""
    cfg = payload["config"]
    lines = [
        f"Benchmark: {cfg['n_neurons']:,} neurons, {payload['n_synapses']:,} synapses, "
        f"{cfg['steps']} steps, dt={cfg['dt']} ms, seed {cfg['seed']}.",
        "",
        "| path | elapsed (s) | synapses/s | neurons/s | spikes | edges visited |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in payload["results"].items():
        if "skipped" in r:
            lines.append(f"| {name} | — | — | — | — | {r['skipped']} |")
            continue
        lines.append(
            f"| {name} | {r['elapsed_s']:.3f} | {r['synapses_per_s']:,.0f} | "
            f"{r['neurons_per_s']:,.0f} | {r['spikes']:,} | {r['synapses_visited']:,} |"
        )
    env = payload["environment"]
    lines += [
        "",
        f"Measured on: {env['processor']} ({env['cpu_count']} cores), "
        f"Python {env['python']}, NumPy {env['numpy']}, "
        f"Numba {env['numba'] or 'not installed'}.",
        "",
        "These numbers are from one machine and are not a claim about whole-brain performance.",
    ]

    if payload.get("crossover"):
        lines += [
            "",
            "### Crossover: scan vs event-driven",
            "",
            "| neurons | synapses | scan (s) | event-driven (s) | speedup | edge hit rate |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in payload["crossover"]:
            lines.append(
                f"| {row['n_neurons']:,} | {row['n_synapses']:,} | {row['scan_s']:.3f} | "
                f"{row['event_driven_s']:.3f} | {row['speedup']:.2f}x | "
                f"{row['edge_hit_rate'] * 100:.2f}% |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="baseline JSON path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-run and fail if the committed baseline is stale relative to the code",
    )
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args(argv)

    payload = run_benchmarks(CONFIG)
    payload["environment"] = environment()
    payload["crossover"] = measure_crossover()

    if args.check:
        if not args.out.exists():
            print(f"FAIL: no committed baseline at {args.out}", file=sys.stderr)
            return 1
        committed = json.loads(args.out.read_text())
        problems = []

        if committed.get("config") != payload["config"]:
            problems.append(
                f"benchmark config changed: committed={committed.get('config')} "
                f"current={payload['config']}"
            )
        if (
            committed.get("environment", {}).get("mem20_brainz")
            != payload["environment"]["mem20_brainz"]
        ):
            problems.append(
                "package version changed: committed="
                f"{committed.get('environment', {}).get('mem20_brainz')} "
                f"current={payload['environment']['mem20_brainz']}"
            )

        # Spike counts are deterministic for a fixed seed, so they are a real
        # correctness signal: if they move, the engine changed.
        for name, current in payload["results"].items():
            old = committed.get("results", {}).get(name, {})
            if "skipped" in current or "skipped" in old:
                continue
            if old.get("spikes") != current.get("spikes"):
                problems.append(
                    f"{name}: spike count changed {old.get('spikes')} -> {current.get('spikes')}"
                )

        if problems:
            print("FAIL: committed baseline is stale:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            print(
                "\nRe-run without --check to regenerate, then commit the new baseline.",
                file=sys.stderr,
            )
            return 1

        print(f"OK: baseline at {args.out} matches the current code.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")

    if args.markdown:
        print()
        print(render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
