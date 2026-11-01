"""Region primitives.

Regions are the coarse-grained unit of composition: a named group of neurons
with a declared cell-type composition. Real parcellations (e.g. a cortical
atlas) are roadmap items; nothing here is anatomically derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Region:
    """A named group of neurons."""

    name: str
    neuron_count: int = 0
    cell_types: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.neuron_count < 0:
            raise ValueError("neuron_count must be non-negative")
        total = sum(self.cell_types.values())
        if self.cell_types and abs(total - 1.0) > 1e-6:
            raise ValueError(f"cell_types fractions must sum to 1.0 (got {total})")
