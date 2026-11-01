"""Config and fixture I/O.

Only JSON is supported today. YAML configs are parsed by a tiny fallback that
handles flat `key: value` files, so the example configs work without PyYAML.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..connectome.sparse import SparseConnectome


def _parse_flat_yaml(text: str) -> dict:
    """Parse a flat `key: value` YAML subset (no nesting, no lists)."""
    out: dict = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            out[key.strip()] = value.lower() == "true"
            continue
        try:
            out[key.strip()] = int(value)
            continue
        except ValueError:
            pass
        try:
            out[key.strip()] = float(value)
            continue
        except ValueError:
            pass
        out[key.strip()] = value.strip("'\"")
    return out


def load_config(path: str | Path) -> dict:
    """Load a JSON or flat-YAML config file into a dict."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".json"}:
        return json.loads(text)
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            return _parse_flat_yaml(text)
        return yaml.safe_load(text) or {}
    raise ValueError(f"unsupported config extension: {p.suffix}")


def load_connectome(path: str | Path) -> SparseConnectome:
    """Load a small connectome fixture from JSON."""
    return SparseConnectome.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_connectome(connectome: SparseConnectome, path: str | Path) -> None:
    """Write a small connectome fixture to JSON."""
    Path(path).write_text(json.dumps(connectome.to_dict()), encoding="utf-8")
