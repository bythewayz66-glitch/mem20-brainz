"""mem20_brainz — scaffold for a whole-brain neuron replication project.

Mission: build a biologically grounded model of the human brain with one
modeled unit per neuron (~86 billion neurons, ~100 trillion synapses).

STATUS: pre-alpha scaffold (v0.0.1).

What exists today: data containers for neurons, synapses, regions and a sparse
connectome, plus a minimal leaky integrate-and-fire stepping loop that can run a
few thousand synthetic neurons. What does NOT exist: any real connectome data,
any biologically validated neuron model, any distributed/GPU execution, and any
claim of brain-scale simulation. See README.md for the honest status table.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
