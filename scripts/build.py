#!/usr/bin/env python3
"""Build helper: install the package in editable mode and run the test suite."""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    steps = [
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]"],
        [sys.executable, "-m", "pytest"],
    ]
    for step in steps:
        code = run(step)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
