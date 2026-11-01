# mem20-brainz

**A public, open, biologically grounded attempt to model the human brain at the
scale of individual neurons — one modeled unit per neuron, ~86 billion of them,
connected by ~100 trillion synapses.**

> **Status: pre-alpha scaffold (v0.0.1).** This repository currently contains a
> project skeleton, data containers, a reference simulation loop, three engines
> (NumPy, Numba, and a CSR event-driven path) checked against that reference, and
> a live observability layer (streaming server + web frontend) for watching a
> small synthetic network run. It does **not** contain a brain model, real
> connectome data, or any brain-scale capability. Every claim below is labelled
> as *implemented* or *aspirational*. Please read
> [Current status](#current-status) before anything else.

---

## The scientific goal

The human brain is the only known system that produces a mind, and it does so
with roughly **86 billion neurons** and on the order of **100 trillion
synapses**. Every attempt to understand it computationally has had to make a
choice: model a small circuit faithfully, or model the whole system coarsely.

This project takes the long way around. The target is a model in which **each
neuron in the human brain is represented by its own modeled unit** — with its
own state, its own dynamics, and its own set of incoming and outgoing synapses —
and in which the connectivity is grounded in measured connectome data wherever
such data exists, and explicitly labelled as synthetic where it does not.

That is not a claim that such a model would be conscious, or that it would
reproduce human behaviour. It is a claim about *resolution*: we want to stop
asking "what does the average neuron in this region do" and start asking "what
does this neuron do, given who it is connected to".
