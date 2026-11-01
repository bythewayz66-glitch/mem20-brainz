"""Run orchestration for the live observability layer.

This module advances the engine and assembles frames. It deliberately contains
**no simulation arithmetic**: every voltage update happens inside the
:class:`~mem20_brainz.simulation.backend.Backend` behind
:class:`~mem20_brainz.simulation.backend.VectorizedEngine`. If you find yourself
writing a membrane equation here, it belongs in the backend instead (ADR-0007).

What the runner does own
------------------------
* the run lifecycle (start / pause / reset / step),
* stimulus injection (adding an external current to a chosen region),
* the aggregate statistics a frame needs (per-region rates, synchrony),
* the running expectations the readouts consume (rate EMA, arousal EMA),
* wall-clock throughput measurement.

Threading
---------
The runner is guarded by a lock. FastAPI's WebSocket handler runs in the event
loop while the stepping happens in a worker thread, so every mutation of engine
state goes through :meth:`RunRunner.locked`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..readout.proxy import ReadoutInputs, compute_all, synchrony_index, to_dicts
from ..scenarios.demo_network import DemoNetwork, build_demo_network
from ..simulation.backend import VectorizedEngine

#: How many recent steps the synchrony window keeps.
SYNCHRONY_WINDOW = 40

#: How many recent steps the per-region rate window keeps. At dt = 0.1 ms this
#: is 20 ms of simulated time. A shorter window makes most regions read 0 Hz on
#: any given frame (spikes are sparse per step), which drives the cross-region
#: coefficient of variation to its ceiling and pins the confusion readout at 1.
REGION_WINDOW_STEPS = 200

#: Steps advanced before the first frame is served, so the network is already
#: firing when a viewer connects. The demo network needs ~300 steps to reach its
#: first spike, and the per-region rate window is REGION_WINDOW_STEPS long, so
#: the warm-up must exceed both for the first frame to show steady-state rates
#: rather than a window that still contains the silent start-up. This is real
#: simulation, not fabricated state, and it is counted in the step counter.
WARMUP_STEPS = 1000

#: EMA time constant for the rate expectation, in steps.
EXPECTATION_TAU_STEPS = 60.0

#: EMA time constant for "sustained" arousal, in steps.
SUSTAINED_TAU_STEPS = 120.0

#: Default stimulus strength, in the placeholder current units the engine uses.
DEFAULT_STIMULUS_STRENGTH = 6.0


@dataclass
class Stimulus:
    """An external current injected into one region for a bounded time."""

    region: str
    strength: float
    remaining_steps: int
    total_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "strength": self.strength,
            "remaining_steps": self.remaining_steps,
            "total_steps": self.total_steps,
        }


@dataclass
class RunConfig:
    """Mutable run parameters, settable over the API."""

    dt_ms: float = 0.1
    #: Steps advanced per streamed frame. >1 means the stream is decimated.
    steps_per_frame: int = 4
    #: Target frames per second for the stream loop.
    target_fps: float = 30.0
    #: Reference rate the arousal readout divides by, in Hz. Chosen to sit near
    #: the demo network's spontaneous rate at the default tonic drive, so the
    #: gauge has room to move in both directions. A display calibration, not a
    #: measured constant.
    baseline_rate_hz: float = 12.0
    #: Region the fear_like readout uses for its synchrony term.
    threat_region: str = "threat"
    #: Seed for the scenario generator.
    seed: int = 20261101
    #: Tonic (constant) current applied to every neuron, in the engine's
    #: placeholder current units. Without it the network is silent: the LIF
    #: defaults sit at rest and nothing depolarises them, so no spike ever
    #: occurs and no readout can move. This is a drive parameter, not a
    #: physiological constant. 2.0 is the smallest value that produces reliable
    #: spontaneous firing in the demo network; below ~1.9 it is silent.
    tonic_current: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dt_ms": self.dt_ms,
            "steps_per_frame": self.steps_per_frame,
            "target_fps": self.target_fps,
            "baseline_rate_hz": self.baseline_rate_hz,
            "threat_region": self.threat_region,
            "seed": self.seed,
            "tonic_current": self.tonic_current,
        }


@dataclass
class RunStats:
    """Throughput and progress counters."""

    steps: int = 0
    total_spikes: int = 0
    steps_per_sec: float = 0.0
    frames_emitted: int = 0
    _last_t: float = field(default=0.0, repr=False)
    _last_steps: int = field(default=0, repr=False)

    def observe(self, steps: int) -> None:
        now = time.perf_counter()
        if self._last_t > 0.0:
            elapsed = now - self._last_t
            if elapsed > 1e-6:
                self.steps_per_sec = (steps - self._last_steps) / elapsed
        self._last_t = now
        self._last_steps = steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "total_spikes": self.total_spikes,
            "steps_per_sec": round(self.steps_per_sec, 2),
            "frames_emitted": self.frames_emitted,
        }


class RunRunner:
    """Owns one simulation run and produces frames from it."""

    def __init__(self, config: RunConfig | None = None) -> None:
        self._lock = threading.RLock()
        self.config = config or RunConfig()
        self.network: DemoNetwork = self._build_network()
        self.engine: VectorizedEngine = self._build_engine()
        self.stats = RunStats()
        self.stimuli: list[Stimulus] = []
        self.running = False
        self._rate_ema = 0.0
        self._arousal_ema = 0.0
        self._spike_count_window: list[int] = []
        self._threat_window: list[int] = []
        self._region_window: list[list[int]] = []
        self._started_at = time.time()
        self._warm_up()

    # -- construction ------------------------------------------------------
    def _build_network(self) -> DemoNetwork:
        return build_demo_network(
            seed=self.config.seed,
            baseline_rate_hz=self.config.baseline_rate_hz,
            threat_region=self.config.threat_region,
        )

    def _build_engine(self) -> VectorizedEngine:
        return VectorizedEngine(
            state=self.network.state,
            connectome=self.network.connectome,
            dt=self.config.dt_ms,
        )

    # -- lifecycle ---------------------------------------------------------
    def locked(self) -> threading.RLock:
        """The lock guarding engine state. Use as a context manager."""
        return self._lock

    def reset(self) -> None:
        """Rebuild the network and engine from the configured seed, then warm up."""
        with self._lock:
            self.network = self._build_network()
            self.engine = self._build_engine()
            self.stats = RunStats()
            self.stimuli = []
            self.running = False
            self._rate_ema = 0.0
            self._arousal_ema = 0.0
            self._spike_count_window = []
            self._threat_window = []
            self._region_window = []
            self._started_at = time.time()
            self._warm_up()

    def _warm_up(self) -> None:
        """Advance past the network's silent start-up before serving frames.

        The demo network takes ~300 steps (30 ms of simulated time) to reach its
        first spike, because the tonic drive has to charge each membrane to
        threshold. Serving that from a cold start means a viewer opens the page
        and watches 2.5 s of nothing, which reads as a broken app rather than as
        a network warming up.

        The warm-up is real simulation, not fabricated state: it runs the same
        engine with the same drive. It is counted in ``stats.steps`` so the step
        counter stays honest about how much simulated time has elapsed.
        """
        for _ in range(WARMUP_STEPS):
            spikes = self.engine.step(self._external_current())
            self.stats.steps += 1
            self.stats.total_spikes += len(spikes)
            self._spike_count_window.append(len(spikes))
            if len(self._spike_count_window) > SYNCHRONY_WINDOW:
                self._spike_count_window.pop(0)
            self._region_window.append(spikes)
            if len(self._region_window) > REGION_WINDOW_STEPS:
                self._region_window.pop(0)
        # Seed the rate expectation from the warmed-up state, so the surprise
        # readout does not report a huge error on the very first frame.
        dt_s = self.config.dt_ms / 1000.0
        if self._spike_count_window:
            mean_spikes = float(np.mean(self._spike_count_window))
            self._rate_ema = mean_spikes / max(self.network.n_neurons, 1) / max(dt_s, 1e-9)

    def start(self) -> None:
        with self._lock:
            self.running = True

    def pause(self) -> None:
        with self._lock:
            self.running = False

    def set_config(self, **changes: Any) -> RunConfig:
        """Update run parameters. Rebuilds the engine if ``dt_ms`` changed."""
        with self._lock:
            allowed = set(RunConfig().to_dict())
            unknown = set(changes) - allowed
            if unknown:
                raise KeyError(f"unknown config key(s): {sorted(unknown)}")
            for key, value in changes.items():
                if value is None:
                    continue
                setattr(self.config, key, value)
            if "dt_ms" in changes and changes["dt_ms"] is not None:
                self.engine.dt = self.config.dt_ms
            if "baseline_rate_hz" in changes and changes["baseline_rate_hz"] is not None:
                self.network.baseline_rate_hz = self.config.baseline_rate_hz
            return self.config

    def inject(self, region: str, strength: float, duration_ms: float) -> Stimulus:
        """Inject an external current into ``region`` for ``duration_ms``."""
        with self._lock:
            if region not in self.network.region_names():
                raise KeyError(f"unknown region {region!r}")
            if duration_ms <= 0:
                raise ValueError("duration_ms must be positive")
            steps = max(1, round(duration_ms / self.config.dt_ms))
            stim = Stimulus(
                region=region,
                strength=float(strength),
                remaining_steps=steps,
                total_steps=steps,
            )
            self.stimuli.append(stim)
            return stim

    # -- stepping ----------------------------------------------------------
    def _external_current(self) -> np.ndarray:
        """Tonic drive plus the currents from all active stimuli."""
        current = np.full(
            self.network.n_neurons, float(self.config.tonic_current), dtype=np.float64
        )
        if not self.stimuli:
            return current
        active: list[Stimulus] = []
        for stim in self.stimuli:
            if stim.remaining_steps <= 0:
                continue
            current[self.network.region_indices(stim.region)] += stim.strength
            stim.remaining_steps -= 1
            active.append(stim)
        self.stimuli = active
        return current

    def advance(self, steps: int | None = None) -> tuple[list[int], list[int]]:
        """Advance ``steps`` timesteps.

        Returns ``(last_step_spikes, all_spikes_in_window)``. The second value is
        what per-region rates are computed from: a rate taken from a single
        0.1 ms step is an instantaneous count, not a rate, and dividing by that
        dt produces meaningless hundreds-of-Hz numbers.
        """
        with self._lock:
            n = self.config.steps_per_frame if steps is None else steps
            spikes: list[int] = []
            window: list[int] = []
            threat_idx = self.network.region_indices(self.network.threat_region)
            for _ in range(max(1, n)):
                spikes = self.engine.step(self._external_current())
                window.extend(spikes)
                self.stats.steps += 1
                self.stats.total_spikes += len(spikes)
                self._spike_count_window.append(len(spikes))
                if len(self._spike_count_window) > SYNCHRONY_WINDOW:
                    self._spike_count_window.pop(0)
                threat_spikes = int(np.isin(spikes, threat_idx).sum())
                self._threat_window.append(threat_spikes)
                if len(self._threat_window) > SYNCHRONY_WINDOW:
                    self._threat_window.pop(0)
                self._region_window.append(spikes)
                if len(self._region_window) > REGION_WINDOW_STEPS:
                    self._region_window.pop(0)
            self.stats.observe(self.stats.steps)
            return spikes, window

    # -- aggregation -------------------------------------------------------
    def _region_activity(self, window_spikes: list[int]) -> list[Any]:
        """Per-region aggregates over a rolling window of recent steps.

        ``rate_hz = spikes_in_window / n_neurons / window_duration_s``, which is
        a genuine rate. ``active_fraction`` is the fraction of the region's
        neurons that fired at least once in the window.

        The window is :data:`REGION_WINDOW_STEPS` steps, not the frame's own
        ``steps_per_frame``: at 4 steps per frame most regions contain no spike
        at all, so a per-frame rate would be 0 Hz almost everywhere and the
        cross-region dispersion would be an artefact of the sampling rate rather
        than a property of the network.
        """
        from .frames import RegionActivity

        # Flatten the rolling window into one spike list.
        flat: list[int] = []
        for step_spikes in self._region_window:
            flat.extend(step_spikes)
        spike_arr = np.asarray(flat, dtype=np.int32)
        v = self.network.state.v
        n_steps = max(len(self._region_window), 1)
        window_s = max(n_steps * self.config.dt_ms / 1000.0, 1e-9)
        out: list[RegionActivity] = []
        for region in self.network.regions:
            idx = region.indices
            n = region.count
            if spike_arr.size:
                in_region_mask = np.isin(spike_arr, idx)
                n_spikes = int(in_region_mask.sum())
                n_active = int(np.unique(spike_arr[in_region_mask]).size)
            else:
                n_spikes = 0
                n_active = 0
            out.append(
                RegionActivity(
                    name=region.name,
                    n_neurons=n,
                    rate_hz=n_spikes / max(n, 1) / window_s,
                    active_fraction=n_active / max(n, 1),
                    mean_v_mv=float(v[idx].mean()),
                )
            )
        return out

    def _readout_inputs(
        self, region_activity: list[Any], window_spikes: list[int]
    ) -> ReadoutInputs:
        """Assemble the measured quantities the readouts consume."""
        dt_s = self.config.dt_ms / 1000.0
        n = self.network.n_neurons
        # Population rate over the synchrony window, in Hz.
        window = self._spike_count_window or [len(window_spikes)]
        mean_spikes_per_step = float(np.mean(window))
        rate_hz = mean_spikes_per_step / max(n, 1) / max(dt_s, 1e-9)

        # Excitatory vs inhibitory drive arriving over the region window.
        # Taken over the whole rolling window rather than one step: a single
        # 0.1 ms step usually contains no spikes at all, so a per-step drive
        # would be zero most of the time and the valence gauge would flicker
        # between 0 and 1. The window is the same one the region rates use, so
        # the two readouts describe the same slice of time.
        exc = inh = 0.0
        flat: list[int] = []
        for step_spikes in self._region_window:
            flat.extend(step_spikes)
        if flat:
            mask = np.isin(self.network.connectome.pre, flat)
            if mask.any():
                w = self.network.connectome.weight[mask]
                exc = float(np.abs(w[w > 0]).sum())
                inh = float(np.abs(w[w < 0]).sum())

        # Running expectation of the rate.
        alpha = 1.0 / EXPECTATION_TAU_STEPS
        self._rate_ema = (1.0 - alpha) * self._rate_ema + alpha * rate_hz

        # Synchrony is computed on the region window, not the 40-step window.
        # At dt = 0.1 ms a 40-step window is 4 ms, which is shorter than the
        # 2 ms refractory period: almost every step has 0 or 1 spikes, so the
        # Fano factor is dominated by refractory spacing rather than by
        # population co-fluctuation, and the index reads ~0 regardless of what
        # the network is doing. The region window is long enough to contain
        # genuine co-fluctuation.
        counts = [len(s) for s in self._region_window]
        threat_idx = self.network.region_indices(self.network.threat_region)
        threat_counts = [int(np.isin(s, threat_idx).sum()) for s in self._region_window]
        synchrony = synchrony_index(counts, n)
        threat_synchrony = synchrony_index(threat_counts, threat_idx.size)

        return ReadoutInputs(
            rate_hz=rate_hz,
            baseline_rate_hz=self.network.baseline_rate_hz,
            exc_drive=exc,
            inh_drive=inh,
            expected_rate_hz=self._rate_ema,
            region_rates_hz=[r.rate_hz for r in region_activity],
            synchrony=synchrony,
            threat_synchrony=threat_synchrony,
            sustained_arousal=self._arousal_ema,
        )

    def _status(self) -> dict[str, Any]:
        return {
            "backend": self.engine.backend.name,
            "engine": "VectorizedEngine",
            "n_neurons": self.network.n_neurons,
            "n_synapses": self.network.n_synapses,
            "running": self.running,
            "data_kind": self.network.provenance.get("kind", "unknown"),
            "neuron_model_tier": self.network.provenance.get("neuron_model_tier", "unknown"),
            "uptime_s": round(time.time() - self._started_at, 2),
            **self.stats.to_dict(),
        }

    def frame(self) -> Any:
        """Advance one frame's worth of steps and return the packed frame."""
        from .frames import build_frame

        with self._lock:
            spikes, window = self.advance()
            region_activity = self._region_activity(window)
            inputs = self._readout_inputs(region_activity, window)
            readouts = compute_all(inputs)
            # Update the sustained-arousal EMA from this frame's arousal value.
            arousal_value = next(r.value for r in readouts if r.name == "arousal")
            beta = 1.0 / SUSTAINED_TAU_STEPS
            self._arousal_ema = (1.0 - beta) * self._arousal_ema + beta * arousal_value
            self.stats.frames_emitted += 1

            return build_frame(
                step=self.stats.steps,
                t_ms=self.engine.t,
                dt_ms=self.config.dt_ms,
                v=self.network.state.v,
                spikes=spikes,
                region_activity=region_activity,
                readouts=to_dicts(readouts),
                status=self._status(),
                provenance=self.network.provenance,
            )

    def snapshot(self) -> dict[str, Any]:
        """Static description of the run, for the initial handshake."""
        with self._lock:
            return {
                "schema_version": 1,
                "config": self.config.to_dict(),
                "status": self._status(),
                "provenance": self.network.provenance,
                "regions": [
                    {
                        "name": r.name,
                        "n_neurons": r.count,
                        "kind": r.kind,
                        "center": list(r.center),
                    }
                    for r in self.network.regions
                ],
                "positions": self.network.positions.round(5).tolist(),
                "readout_provenance": "derived_proxy",
                "readout_disclaimer": (
                    "Affective readouts are engineering proxies over simulation state. "
                    "The model has no feelings and no claim of subjective experience is made."
                ),
            }
