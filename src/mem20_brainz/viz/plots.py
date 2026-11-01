"""Plotting helpers (optional matplotlib dependency)."""

from __future__ import annotations


def plot_raster(spike_history: list[list[int]], dt: float = 0.1, path: str | None = None):
    """Raster-plot a spike history.

    `spike_history[k]` is the list of neuron ids that spiked at step k.
    Returns the matplotlib Figure. Raises ImportError if matplotlib is absent.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plot_raster requires matplotlib: pip install 'mem20-brainz[viz]'") from exc

    xs, ys = [], []
    for step, spikes in enumerate(spike_history):
        for neuron_id in spikes:
            xs.append(step * dt)
            ys.append(neuron_id)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(xs, ys, s=2, marker="|")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron id")
    ax.set_title("Spike raster")
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
