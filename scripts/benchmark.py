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

Scale runs (``--scale``)
------------------------
The committed baseline is small on purpose: it must run in CI on every push, so
it tops out at a few tens of thousands of neurons. ``--scale`` is the other
question — *where does this break?* — and sweeps up to 10^6 neurons, the Phase 1
exit target. It is opt-in, never runs in CI, writes to its own file
(``docs/benchmarks/scale.json``), and reports peak RSS alongside time, because
at 10^6 the memory wall is as interesting as the clock. The reference loop is
excluded: it is a correctness oracle, not a scale path.

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
DEFAULT_SCALE_OUT = REPO_ROOT / "docs" / "benchmarks" / "scale.json"

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


#: Sizes swept by ``--scale``. 10^6 is the Phase 1 exit target; the smaller
#: rows are there so a wall can be attributed to a size rather than guessed at.
SCALE_SIZES = (10_000, 100_000, 1_000_000)

#: Steps per size. 100 matches the crossover sweep: enough for the population
#: to reach a firing regime, so the event-driven row measures real delivery
#: rather than an idle network (the hit rate is recorded to make that visible).
SCALE_STEPS = 100

#: Fixed mean out-degree, so edges grow linearly with the population and the
#: sweep is about size, not density.
SCALE_MEAN_OUT_DEGREE = 10.0


def _peak_rss_mb() -> float | None:
    """Process high-water RSS in MiB, or ``None`` where unavailable."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kibibytes, macOS reports bytes.
    return rss / 1024 if sys.platform != "darwin" else rss / (1024 * 1024)


def measure_scale(sizes: tuple[int, ...] = SCALE_SIZES, steps: int = SCALE_STEPS) -> list[dict]:
    """Sweep population size on the vectorised paths to find the wall.

    ``peak_rss_mb`` is the process high-water mark, so it is monotonically
    non-decreasing across rows: it shows the ceiling each row pushed the process
    to, not that row's working set. It is recorded for exactly that reason — the
    question at 10^6 is how close the process gets to the machine's limit.
    """
    rows: list[dict] = []
    for n in sizes:
        p = SCALE_MEAN_OUT_DEGREE / n
        conn = build_connectome(n, p, seed=1)
        external = np.full(n, 45.0, dtype=np.float64)
        row: dict = {
            "n_neurons": n,
            "n_synapses": conn.n_synapses,
            "steps": steps,
            "peak_rss_mb_before": _peak_rss_mb(),
        }

        scan = VectorizedEngine(
            state=LIFState.uniform(n), connectome=conn, dt=0.1, backend=NumPyBackend()
        )
        start = time.perf_counter()
        scan.run(steps=steps, external_current=external)
        scan_s = time.perf_counter() - start
        row["numpy_s"] = scan_s
        row["numpy_neurons_per_s"] = n * steps / scan_s
        row["numpy_synapses_per_s"] = conn.n_synapses * steps / scan_s

        csr = CSRConnectome.from_sparse(conn)
        ev = EventDrivenEngine(state=LIFState.uniform(n), connectome=csr, dt=0.1)
        start = time.perf_counter()
        ev.run(steps=steps, external_current=external)
        ev_s = time.perf_counter() - start
        row["event_driven_s"] = ev_s
        row["event_driven_neurons_per_s"] = n * steps / ev_s
        row["event_driven_synapses_per_s"] = ev.edges_visited / ev_s
        row["speedup"] = scan_s / ev_s
        row["edge_hit_rate"] = ev.edges_visited / (conn.n_synapses * steps)
        row["peak_rss_mb_after"] = _peak_rss_mb()

        rows.append(row)
        print(
            f"  {n:>9,} neurons  {conn.n_synapses:>11,} synapses  "
            f"numpy {scan_s:.3f}s  event-driven {ev_s:.3f}s  "
            f"peak RSS {row['peak_rss_mb_after']:.0f} MiB",
            file=sys.stderr,
        )
    return rows


def render_scale_markdown(payload: dict) -> str:
    env = payload["environment"]
    lines = [
        "Scale sweep: where the vectorised paths run out of room.",
        "",
        "| neurons | synapses | numpy (s) | event-driven (s) | speedup | "
        "edge hit rate | numpy neurons/s | peak RSS (MiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["scale"]:
        lines.append(
            f"| {row['n_neurons']:,} | {row['n_synapses']:,} | {row['numpy_s']:.3f} | "
            f"{row['event_driven_s']:.3f} | {row['speedup']:.2f}x | "
            f"{row['edge_hit_rate'] * 100:.2f}% | "
            f"{row['numpy_neurons_per_s']:,.0f} | {row['peak_rss_mb_after']:.0f} |"
        )
    lines += [
        "",
        f"Measured on: {env['processor']} ({env['cpu_count']} cores), "
        f"Python {env['python']}, NumPy {env['numpy']}.",
        "",
        "Peak RSS is a process high-water mark, so it is non-decreasing down the "
        "column. This is a single-machine measurement, not a scaling projection.",
    ]
    return "\n".join(lines)


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
    parser.add_argument("--out", type=Path, default=None, help="output JSON path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-run and fail if the committed baseline is stale relative to the code",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="sweep up to 10^6 neurons to find the memory and time wall (opt-in, not CI)",
    )
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args(argv)

    if args.scale:
        if args.check:
            parser.error("--scale and --check are mutually exclusive")
        out = args.out or DEFAULT_SCALE_OUT
        print(f"scale sweep: {', '.join(f'{n:,}' for n in SCALE_SIZES)} neurons", file=sys.stderr)
        payload = {"environment": environment(), "scale": measure_scale()}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {out}")
        if args.markdown:
            print()
            print(render_scale_markdown(payload))
        return 0

    out = args.out or DEFAULT_OUT
    payload = run_benchmarks(CONFIG)
    payload["environment"] = environment()
    payload["crossover"] = measure_crossover()

    if args.check:
        if not out.exists():
            print(f"FAIL: no committed baseline at {out}", file=sys.stderr)
            return 1
        committed = json.loads(out.read_text())
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

        print(f"OK: baseline at {out} matches the current code.")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")

    if args.markdown:
        print()
        print(render_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
