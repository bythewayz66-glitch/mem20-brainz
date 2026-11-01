#!/usr/bin/env python3
"""Connectome ingestion placeholder.

Intended job: pull a connectome dataset from external storage, validate it
against data/schemas/connectome.schema.json, and emit a sharded internal format.

NOT IMPLEMENTED. This script only validates a small local JSON fixture so the
validation path exists and is testable. It deliberately refuses to run on
anything that looks like a real dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_FIXTURE_BYTES = 5 * 1024 * 1024  # 5 MB: fixtures only, never real data


def validate(path: Path) -> int:
    size = path.stat().st_size
    if size > MAX_FIXTURE_BYTES:
        print(
            f"refusing to ingest {path} ({size} bytes): this script handles small "
            "fixtures only. Real connectome data must be streamed from external "
            "storage (see data/README.md).",
            file=sys.stderr,
        )
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"n_neurons", "pre", "post"}
    missing = required - data.keys()
    if missing:
        print(f"invalid connectome fixture: missing keys {sorted(missing)}", file=sys.stderr)
        return 1
    if len(data["pre"]) != len(data["post"]):
        print("invalid connectome fixture: pre/post length mismatch", file=sys.stderr)
        return 1

    print(f"ok: {path} — {data['n_neurons']} neurons, {len(data['pre'])} synapses")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="path to a small connectome fixture (JSON)")
    args = parser.parse_args()
    return validate(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
