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

So this script does not differentiate the spike train. It optimises a **smooth
proxy**: the mean membrane potential, which is continuous in ``i_inj``. The
proxy is chosen because it is monotone in the drive, so driving it to a target
drives the spike rate in the same direction. The spike rate is then *measured*
at each step and reported — as an outcome, never as a differentiated quantity.

That distinction matters: a surrogate-gradient result is a heuristic search, not
an exact gradient descent. It converges here because the problem is 1-D and
monotone. It would not be trustworthy for a high-dimensional fit without a
proper surrogate-gradient method (e.g. SuperSpike), which this project does not
yet implement.

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
LEARNING_RATE = 0.5
ITERATIONS = 60
SEED = 20261101


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print(
            "This example needs the optional jax extra:\n    pip install 'mem20-brainz[jax]'",
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


def make_simulate(jax, jnp, state: LIFState, steps: int, dt: float, kappa: float = 2.0):
    """Trace the LIF loop, returning (surrogate_rate, spike_count).

    The surrogate is a **sigmoid pseudo-spike** ``sigmoid((v - v_th) / kappa)``
    accumulated over neurons and steps. It is the standard trick (SuperSpike and
    relatives): a smooth, monotone stand-in for the step function at threshold,
    with ``kappa`` setting how sharply it approximates the step.

    Why not the mean membrane potential? Because it is **not monotone** in the
    drive: once neurons start spiking they reset to ``v_reset``, so pushing
    harder can *lower* the mean ``v``. An earlier version of this script used
    mean ``v`` as the proxy and the fit diverged — loss rose from 3.6 to 107
    while the rate climbed from 57 Hz to 387 Hz. That failure is recorded in
    ADR-0008 because it is the concrete reason a naive proxy is not enough.

    The spike count is returned from the same trace, so it is a genuine
    measurement of the simulation the gradient was taken through.
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
            # Smooth pseudo-spike, evaluated on the pre-reset potential.
            pseudo = jax.nn.sigmoid((v_after - v_threshold) / kappa)
            return new_arrays, (jnp.sum(pseudo), jnp.sum(spiked.astype(jnp.float64)))

        # lax.scan, not a Python loop: unrolling 200 steps builds a graph large
        # enough to exhaust memory, and scan is the idiomatic JAX form anyway.
        times = jnp.arange(steps, dtype=jnp.float64) * dt
        _, (pseudo_sums, spike_counts) = jax.lax.scan(body, arrays0, times)
        return jnp.sum(pseudo_sums), jnp.sum(spike_counts)

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
    simulate = make_simulate(jax, jnp, state, STEPS, DT)

    # --- calibrate the surrogate target ---------------------------------
    # The surrogate is a pseudo-spike count, but the *goal* is a real spike
    # rate. So first find the drive that produces the target rate, by bisection,
    # and read off the surrogate value there. That surrogate value becomes the
    # target the gradient descends toward.
    def rate_at(i_inj: float) -> float:
        _, spikes = simulate(jnp.asarray(float(i_inj)))
        return rate_hz(float(spikes), N_NEURONS, STEPS, DT)

    lo, hi = 0.0, 200.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if rate_at(mid) < TARGET_RATE_HZ:
            lo = mid
        else:
            hi = mid
    calibrated_drive = 0.5 * (lo + hi)
    target_surrogate = float(simulate(jnp.asarray(calibrated_drive))[0])

    # --- gradient fit on the surrogate ----------------------------------
    def surrogate_loss(i_inj):
        surrogate, _ = simulate(i_inj)
        return (surrogate - target_surrogate) ** 2

    grad_fn = jax.jit(jax.grad(surrogate_loss))

    # Start deliberately far from the answer so the fit has work to do.
    i_inj = jnp.asarray(5.0)
    history = []
    for step in range(ITERATIONS):
        loss = float(surrogate_loss(i_inj))
        _, spikes = simulate(i_inj)
        history.append(
            {
                "iteration": step,
                "i_inj": float(i_inj),
                "loss": loss,
                "rate_hz": rate_hz(float(spikes), N_NEURONS, STEPS, DT),
            }
        )
        g = grad_fn(i_inj)
        i_inj = i_inj - LEARNING_RATE * g

    final = history[-1]
    result = {
        "config": {
            "n_neurons": N_NEURONS,
            "steps": STEPS,
            "dt_ms": DT,
            "target_rate_hz": TARGET_RATE_HZ,
            "learning_rate": LEARNING_RATE,
            "iterations": ITERATIONS,
            "seed": SEED,
        },
        "calibrated_drive_na": calibrated_drive,
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
    print(f"  duration        : {STEPS} steps x {DT} ms = {STEPS * DT / 1000:.1f} s")
    print(f"  target rate     : {TARGET_RATE_HZ:.1f} Hz")
    print(f"  calibrated drive: {calibrated_drive:.4f} nA (bisection, for the surrogate target)")
    print()
    print(f"  {'iter':>4}  {'i_inj (nA)':>11}  {'loss':>12}  {'rate (Hz)':>10}")
    for row in [*history[:: max(1, ITERATIONS // 10)], history[-1]]:
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
    print("  NOTE: the gradient is a surrogate. It is taken through the mean")
    print("  membrane potential (a smooth proxy), not through the spike train,")
    print("  which is discontinuous at threshold. The spike rate above is a")
    print("  measured outcome of the fitted drive, not a differentiated quantity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
