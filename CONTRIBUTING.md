# Contributing to mem20-brainz

Thanks for being here. This project is ambitious and early, which means the most
valuable contributions right now are **interfaces, data provenance, and honest
documentation** — not features.

## The two rules

These outrank every style preference in this document.

### 1. Never claim a capability the code does not have

If a module is a placeholder, its docstring says so. If a function is O(n) and
should not be used in a loop, the docstring says so. If a phase is aspirational,
the README marks it aspirational. A contributor who writes "supports brain-scale
simulation" above a 1,000-neuron Python loop has done more damage than one who
writes nothing.

### 2. Never mix measured and synthetic data silently

Every dataset manifest carries a mandatory `measured: true/false` flag. Every
run records which datasets it used. Every figure derived from a run states which
parts of the model were measured and which were generated. If you are unsure
whether a source counts as measured, it is synthetic — say so.

## Getting set up

```bash
git clone https://github.com/bythewayz66-glitch/mem20-brainz.git
cd mem20-brainz
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## What to work on

Check [`ROADMAP.md`](ROADMAP.md) for the current phase. Phase 0 and Phase 1 work
is well-scoped and does not require neuroscience background:

- **Phase 0:** CI, ADRs, docstring accuracy, test coverage of the existing
  primitives.
- **Phase 1:** a `Backend` protocol and a NumPy backend that reproduces the
  reference engine's spikes exactly.

If you have a neuroscience background, the highest-value contribution is a
**validation suite** — a concrete list of measured statistics a run must
reproduce. That is currently the project's biggest gap.

## Changing an interface

Open an issue first. Describe:

1. The interface as it is today.
2. What you want it to be.
3. Which callers break.
4. Why the change is worth breaking them.

Interfaces in `ARCHITECTURE.md` are treated as frozen within a phase. Changing
one mid-phase needs a reason, not just a preference.

## Code standards

- **Python ≥ 3.10**, `from __future__ import annotations` at the top of modules.
- **Type hints** on public functions and methods.
- **Dataclasses** for data containers; avoid mutable default arguments.
- **Line length 100** (`ruff` config in `pyproject.toml`).
- **Docstrings** on every module and public class. State what is a placeholder.
- **No dense N×N allocations.** Ever. If you need adjacency, use a sparse form.
- **No silent fallbacks.** Missing dependency or dataset → raise, with a message
  that says how to fix it.
- **No secrets, no large binaries.** `.gitignore` blocks the common formats; do
  not work around it.

## Tests

- Every new module gets a test that imports it and exercises its contract.
- Numerical code gets an equivalence test against the reference implementation.
- Tests must not depend on network access or on data outside `data/fixtures/`.
- `pytest` must pass before you open a pull request.

## Commits and pull requests

- Small commits, imperative subject line, body explaining *why*.
- One logical change per pull request.
- If your change alters behaviour, say so in the description and update the
  relevant doc.
- If your change adds a dataset, add its manifest in the same pull request.

## Reporting a problem with the science

If you believe a modelling choice is wrong — a parameter, a data source, a
simplification — open an issue with the label `science`. Disagreement about the
model is a feature of this project, not a nuisance. Bring a citation.

## License

By contributing you agree that your contributions are licensed under the MIT
License (see [`LICENSE`](LICENSE)).
