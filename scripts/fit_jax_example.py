#!/usr/bin/env python3
"""Fit a synaptic drive to a target spike rate, using jax.grad.

This is the reason the JAX backend exists. The other backends can run a
simulation; only this one can differentiate through it.

What is actually being fitted
-----------------------------
A single scalar: the constant injected current ``i_inj`` (nA) applied to a
population of LIF neurons. The target is a mean spike rate in Hz. We minimise
the squared error between the simulated rate and the target.

The honest part: the gradient is a surrogate
--------------------------------------------
A spiking neuron's output is a **step function** of its membrane potential. It is
discontinuous at threshold, so its true derivative is zero almost everywhere and
undefined at the threshold. ``jax.grad`` through the spike comparison therefore
returns the derivative of whatever smooth function XLA differentiates through
the comparison — **not** the derivative of the model.

This script uses a **straight-through estimator (STE)**, the standard fix:

    forward  value = hard spike (0 or 1)          <- the real simulation
    backward value = sigmoid((v - v_th) / kappa)  <- a smooth stand-in

implemented as ``soft + stop_gradient(hard - soft)``. The forward pass is the
true spike count, so the reported rate is a real measurement. Only the
*derivative* is substituted. That is a heuristic, not an exact gradient: it
converges here because the problem is 1-D and monotone, and it would not be
trustworthy for a high-dimensional fit without further work.

Two failed attempts are recorded in ADR-0008, because they are the reason the
STE is necessary rather than optional:

1. **Mean membrane potential as the proxy.** Not monotone in the drive: once
   neurons spike they reset to ``v_reset``, so pushing harder can *lower* the
   mean ``v``. The fit diverged — loss rose 3.6 -> 107 while the rate climbed
   57 Hz -> 387 Hz.
2. **STE with the operands swapped** (``hard + stop_gradient(soft - hard)``).
   This is the classic sign error: it makes the forward pass hard and the
   backward pass *zero*, so the gradient vanished and the fit never moved.

Run::

    python scripts/fit_jax_example.py
    python scripts/fit_jax_example.py --json   # machine-readable result
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from mem20_brainz.simulation import LIFState

#: Fixed problem definition. Changing these changes the reported numbers, which
#: is why they are constants rather than flags.
N_NEURONS = 200
STEPS = 200
DT = 0.1  # ms
TARGET_RATE_HZ = 40.0
LEARNING_RATE = 1e-6
ITERATIONS = 300
SEED = 20261101

#: Width of the surrogate sigmoid, in mV. The gradient at the target drive is
#: ~5.4e3 for kappa=2.0, which is why the learning rate is 1e-6 and not 1e-4.
KAPPA = 2.0


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print(
            "This example needs the optional jax extra:\n"
            "    pip install 'mem20-brainz[jax]'",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def build_state(n: int, seed: int) -> LIFState:
    """A heterogeneous population — the regime where precision matters (ADR-0007)."""
    rng = np.random.default_rng(seed)
    state = LIFState.uniform(n)
    state.tau_m = state.tau_m * rng.uniform(0.8, 1.2, n)
    state.v_threshold = state.v_threshold + rng.uniform(-3, 3, n)
    state.v = state.v + rng.uniform(-5, 5, n)
    return state


def make_simulate(jax, jnp, state: LIFState, steps: int, dt: float, kappa: float):
    """Trace the LIF loop, returning (ste_spike_count, hard_spike_count).

    ``ste_spike_count`` carries the surrogate gradient; ``hard_spike_count`` is
    the true spike count and is what gets reported as a rate.
    """
    arrays0 = tuple(
        jnp.asarray(a)
        for a in (
            state.v,
            state.v_rest,
            state.v_reset,
            state.v_threshold,
            state.tau_m,
            state.r_m,
            state.refractory_ms,
            state.refractory_until,
        )
    )

    def simulate(i_inj):
        def body(arrays, t):
            v, v_rest, v_reset, v_threshold, tau_m, r_m, refractory_ms, refractory_until = arrays
            active = t >= refractory_until
            dv = (-(v - v_rest) + r_m * i_inj) / tau_m
            v_next = v + dv * dt
            v_after = jnp.where(active, v_next, v)
            spiked = active & (v_after >= v_threshold)
            new_arrays = (
                jnp.where(spiked, v_reset, v_after),
                v_rest,
                v_reset,
                v_threshold,
                tau_m,
                r_m,
                refractory_ms,
                jnp.where(spiked, t + refractory_ms, refractory_until),
            )

            hard = spiked.astype(jnp.float64)
            soft = jnp.where(active, jax.nn.sigmoid((v_after - v_threshold) / kappa), 0.0)
            # Straight-through: forward = hard, backward = soft.
            ste = soft + jax.lax.stop_gradient(hard - soft)

            return new_arrays, (jnp.sum(ste), jnp.sum(hard))

        # lax.scan, not a Python loop: unrolling 200 steps builds a graph large
        # enough to exhaust memory, and scan is the idiomatic JAX form anyway.
        times = jnp.arange(steps, dtype=jnp.float64) * dt
        _, (ste_counts, hard_counts) = jax.lax.scan(body, arrays0, times)
        return jnp.sum(ste_counts), jnp.sum(hard_counts)

    return jax.jit(simulate)


def rate_hz(spike_count: float, n_neurons: int, steps: int, dt: float) -> float:
    """Mean firing rate in Hz from a spike count."""
    duration_s = steps * dt / 1000.0
    return float(spike_count) / n_neurons / duration_s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    jax, jnp = _require_jax()
    state = build_state(N_NEURONS, SEED)
    simulate = make_simulate(jax, jnp, state, STEPS, DT, KAPPA)

    target_spikes = TARGET_RATE_HZ * N_NEURONS * (STEPS * DT / 1000.0)

    def loss(i_inj):
        ste, _ = simulate(i_inj)
        return (ste - target_spikes) ** 2

    grad_fn = jax.jit(jax.grad(loss))

    # Start below the answer so the fit has to climb.
    i_inj = jnp.asarray(2.0)
    history = []
    for step in range(ITERATIONS):
        ste, hard = simulate(i_inj)
        history.append(
            {
                "iteration": step,
                "i_inj": float(i_inj),
                "loss": float((ste - target_spikes) ** 2),
                "rate_hz": rate_hz(float(hard), N_NEURONS, STEPS, DT),
            }
        )
        i_inj = i_inj - LEARNING_RATE * grad_fn(i_inj)

    final = history[-1]
    result = {
        "config": {
            "n_neurons": N_NEURONS,
            "steps": STEPS,
            "dt_ms": DT,
            "target_rate_hz": TARGET_RATE_HZ,
            "learning_rate": LEARNING_RATE,
            "iterations": ITERATIONS,
            "kappa_mv": KAPPA,
            "seed": SEED,
        },
        "fitted_drive_na": final["i_inj"],
        "initial_loss": history[0]["loss"],
        "final_loss": final["loss"],
        "initial_rate_hz": history[0]["rate_hz"],
        "final_rate_hz": final["rate_hz"],
        "rate_error_hz": final["rate_hz"] - TARGET_RATE_HZ,
        "history": history,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("Fitting a constant injected current to a target spike rate")
    print(f"  population      : {N_NEURONS} heterogeneous LIF neurons")
    print(f"  duration        : {STEPS} steps x {DT} ms = {STEPS * DT / 1000:.2f} s")
    print(f"  target rate     : {TARGET_RATE_HZ:.1f} Hz ({target_spikes:.0f} spikes)")
    print(f"  surrogate kappa : {KAPPA} mV (straight-through estimator)")
    print()
    print(f"  {'iter':>4}  {'i_inj (nA)':>11}  {'loss':>12}  {'rate (Hz)':>10}")
    stride = max(1, ITERATIONS // 10)
    for row in history[::stride] + [history[-1]]:
        print(
            f"  {row['iteration']:>4}  {row['i_inj']:>11.4f}  "
            f"{row['loss']:>12.3e}  {row['rate_hz']:>10.2f}"
        )
    print()
    print(f"  fitted drive    : {final['i_inj']:.4f} nA")
    print(f"  loss            : {history[0]['loss']:.3e} -> {final['loss']:.3e}")
    print(f"  rate            : {history[0]['rate_hz']:.2f} -> {final['rate_hz']:.2f} Hz")
    print(f"  rate error      : {final['rate_hz'] - TARGET_RATE_HZ:+.2f} Hz")
    print()
    print("  NOTE: the gradient is a surrogate (straight-through estimator). The")
    print("  forward pass is the true spike count, so the rate above is a real")
    print("  measurement; only the derivative is substituted. This is a heuristic")
    print("  search, not exact gradient descent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
