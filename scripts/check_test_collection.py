#!/usr/bin/env python3
"""Fail if a test module collects no tests, or the suite shrinks past a floor.

Why this exists: commit 84be2d8 set out to correct one docstring in
``tests/test_event_driven.py`` but deleted lines 58-414 with it, leaving the
module as a docstring plus orphaned imports. ``pytest`` was perfectly happy —
the module simply collected nothing — and the loss only showed up as nine
``ruff`` F401s on the stranded imports. A green test run is not evidence that
the tests are still there, so this guard checks the count.

A module guarded by ``pytest.importorskip`` is allowed to collect nothing when
its optional extra (numba, jax) is absent. Everything else must collect at least
one test, and the suite as a whole must clear ``TOTAL_FLOOR``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

#: Loose sanity floor. The suite collects 127 tests today, 14 of them in
#: ``test_event_driven.py`` — the module this guard was written to protect. It
#: sits a little below that so ordinary churn does not trip it, while a whole
#: module disappearing still does. The per-module check below is the primary
#: guard; this is the backstop for a loss spread across several modules.
TOTAL_FLOOR = 120

#: One line per module in `pytest --collect-only -q`, e.g. "tests/test_smoke.py: 10".
COUNT_LINE = re.compile(r"^(tests/[^:]+\.py):\s+(\d+)\s*$")


def collected() -> dict[str, int]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-header",
            "-p",
            "no:warnings",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        match = COUNT_LINE.match(line.strip())
        if match:
            counts[match.group(1)] = int(match.group(2))
    if not counts:
        sys.exit(
            "error: could not parse pytest collection output\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return counts


def main() -> int:
    counts = collected()
    failures: list[str] = []

    for path in sorted(TESTS.glob("test_*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if counts.get(rel, 0) > 0:
            continue
        if "pytest.importorskip" in path.read_text(encoding="utf-8"):
            continue  # optional extra absent: expected, not a regression
        failures.append(f"{rel} collects 0 tests")

    total = sum(counts.values())
    if total < TOTAL_FLOOR:
        failures.append(f"total collected tests {total} < floor {TOTAL_FLOOR}")

    if failures:
        print("test-collection guard failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"ok: {total} tests collected across {len(counts)} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
