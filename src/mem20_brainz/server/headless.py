"""Headless mode: run the simulation and print a text summary.

This exists so the observability layer is testable in CI without a browser, and
so a terminal user can see the network respond to a stimulus. It uses the same
:class:`~mem20_brainz.server.runner.RunRunner` the WebSocket server uses, so what
it prints is what the frontend would receive.

Usage::

    python -m mem20_brainz.server.headless --steps 400 --stimulus assoc_a
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .runner import RunConfig, RunRunner


def _fmt_readouts(readouts: list[dict[str, Any]]) -> str:
    parts = []
    for r in readouts:
        parts.append(f"{r['name']}={r['value']:.3f}")
    return "  ".join(parts)


def run_headless(
    steps: int = 400,
    stimulus_region: str | None = None,
    stimulus_strength: float = 6.0,
    stimulus_at: int = 100,
    stimulus_duration_ms: float = 200.0,
    print_every: int = 50,
    config: RunConfig | None = None,
) -> dict[str, Any]:
    """Run ``steps`` frames, optionally injecting a stimulus, and summarise.

    Returns a dict with the final frame and a per-checkpoint trace, so a test can
    assert on it without parsing stdout.
    """
    runner = RunRunner(config or RunConfig())
    runner.start()

    trace: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    for i in range(steps):
        if stimulus_region and i == stimulus_at:
            runner.inject(stimulus_region, stimulus_strength, stimulus_duration_ms)
        frame = runner.frame().to_dict()
        final = frame
        if i % print_every == 0 or i == steps - 1:
            trace.append(
                {
                    "step": frame["step"],
                    "t_ms": frame["t_ms"],
                    "spikes": len(frame["spikes"]),
                    "readouts": {r["name"]: r["value"] for r in frame["readouts"]},
                }
            )

    assert final is not None
    return {
        "n_neurons": runner.network.n_neurons,
        "n_synapses": runner.network.n_synapses,
        "backend": runner.engine.backend.name,
        "data_kind": runner.network.provenance.get("kind"),
        "steps": runner.stats.steps,
        "total_spikes": runner.stats.total_spikes,
        "steps_per_sec": round(runner.stats.steps_per_sec, 2),
        "trace": trace,
        "final_frame": final,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mem20_brainz.server.headless",
        description="Run the mem20-brainz simulation headlessly and print a summary.",
    )
    parser.add_argument("--steps", type=int, default=400, help="frames to run")
    parser.add_argument("--stimulus", default=None, help="region to stimulate")
    parser.add_argument("--strength", type=float, default=6.0, help="stimulus current")
    parser.add_argument("--at", type=int, default=100, help="frame index to stimulate at")
    parser.add_argument("--duration-ms", type=float, default=200.0, help="stimulus duration")
    parser.add_argument("--every", type=int, default=50, help="print every N frames")
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = parser.parse_args(argv)

    result = run_headless(
        steps=args.steps,
        stimulus_region=args.stimulus,
        stimulus_strength=args.strength,
        stimulus_at=args.at,
        stimulus_duration_ms=args.duration_ms,
        print_every=args.every,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("mem20-brainz — headless run")
    print("=" * 68)
    print(f"backend      : {result['backend']}")
    print(f"neurons      : {result['n_neurons']:,}")
    print(f"synapses     : {result['n_synapses']:,}")
    print(f"data kind    : {result['data_kind']}  (synthetic — not measured)")
    print(f"steps        : {result['steps']:,}")
    print(f"total spikes : {result['total_spikes']:,}")
    print(f"throughput   : {result['steps_per_sec']:,.1f} steps/s")
    if args.stimulus:
        print(f"stimulus     : {args.stimulus} @ {args.strength} for {args.duration_ms} ms")
    print()
    print(f"{'step':>6} {'t_ms':>9} {'spikes':>7}  readouts")
    print("-" * 68)
    for row in result["trace"]:
        readouts = [{"name": k, "value": v} for k, v in row["readouts"].items()]
        print(f"{row['step']:>6} {row['t_ms']:>9.1f} {row['spikes']:>7}  {_fmt_readouts(readouts)}")
    print()
    print("NOTE: the readouts above are derived proxies over simulation state.")
    print("      The model has no feelings. See docs/readout.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
