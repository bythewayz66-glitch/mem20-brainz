"""Smoke tests: the package imports, the primitives behave, and a tiny
simulation runs end to end. These assert scaffold correctness only — they say
nothing about biological fidelity."""

from __future__ import annotations

import numpy as np
import pytest

import mem20_brainz
from mem20_brainz.connectome import SparseConnectome
from mem20_brainz.io import load_config, load_connectome, save_connectome
from mem20_brainz.neurons import LIFNeuron
from mem20_brainz.regions import Region
from mem20_brainz.simulation import SimulationEngine
from mem20_brainz.synapses import Synapse


def test_version_string():
    assert isinstance(mem20_brainz.__version__, str)
    assert mem20_brainz.__version__.count(".") >= 1


def test_placeholder_neuron_instantiates_and_spikes():
    neuron = LIFNeuron(id=0)
    assert neuron.v == pytest.approx(-70.0)
    # Strong sustained current must drive it over threshold.
    spiked = any(neuron.step(i_inj=50.0, dt=0.1, t=k * 0.1) for k in range(200))
    assert spiked


def test_neuron_stays_quiet_without_input():
    neuron = LIFNeuron(id=1)
    assert not any(neuron.step(0.0, 0.1, k * 0.1) for k in range(200))


def test_synapse_sign():
    assert Synapse(pre=0, post=1, weight=2.0, sign=1).effective_weight == pytest.approx(2.0)
    assert Synapse(pre=0, post=1, weight=2.0, sign=-1).effective_weight == pytest.approx(-2.0)


def test_region_validates_fractions():
    Region(name="V1", neuron_count=10, cell_types={"exc": 0.8, "inh": 0.2})
    with pytest.raises(ValueError):
        Region(name="bad", neuron_count=1, cell_types={"exc": 0.5})


def test_connectome_roundtrip(tmp_path):
    conn = SparseConnectome.from_arrays(
        n_neurons=3, pre=[0, 0, 1], post=[1, 2, 2], weight=[1.0, 0.5, 2.0]
    )
    assert conn.n_synapses == 3
    assert conn.in_degree().tolist() == [0, 1, 2]

    path = tmp_path / "conn.json"
    save_connectome(conn, path)
    loaded = load_connectome(path)
    assert loaded.n_synapses == 3
    assert loaded.pre.tolist() == [0, 0, 1]


def test_connectome_rejects_mismatched_arrays():
    with pytest.raises(ValueError):
        SparseConnectome(n_neurons=2, pre=np.array([0]), post=np.array([1, 1]))


def test_engine_runs_and_records_spikes():
    neurons = [LIFNeuron(id=i) for i in range(4)]
    conn = SparseConnectome.from_arrays(n_neurons=4, pre=[0, 1], post=[2, 3])
    engine = SimulationEngine(neurons=neurons, connectome=conn, dt=0.1)

    drive = np.full(4, 50.0)
    total = engine.run(steps=100, external_current=drive)

    assert total > 0
    assert len(engine.spike_history) == 100
    assert engine.t == pytest.approx(10.0)


def test_engine_rejects_population_mismatch():
    with pytest.raises(ValueError):
        SimulationEngine(
            neurons=[LIFNeuron(id=0)],
            connectome=SparseConnectome.from_arrays(n_neurons=5, pre=[], post=[]),
        )


def test_load_example_config():
    cfg = load_config("configs/simulation_default.yaml")
    assert cfg["dt_ms"] == pytest.approx(0.1)
    assert cfg["n_neurons"] == 1000
