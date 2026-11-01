# Data model

How neurons, synapses, regions, connectomes and datasets are represented, and
what each field means. Field names here are the contract; the JSON Schemas in
[`data/schemas/`](../data/schemas/) are the machine-checkable version.

## Neuron

```json
{
  "id": 0,
  "model": "lif",
  "region": "cortical_column_toy",
  "cell_type": "exc",
  "v_rest_mv": -70.0,
  "v_threshold_mv": -50.0,
  "v_reset_mv": -75.0,
  "tau_m_ms": 20.0,
  "r_m_mohm": 10.0
}
```

| Field | Type | Meaning |
|---|---|---|
| `id` | int ≥ 0 | Global neuron index. Stable within a run. |
| `model` | enum | Model class. Today only `lif`. |
| `region` | string | Owning region name. |
| `cell_type` | string | Cell type label within the region. |
| `v_rest_mv` | float | Resting potential, mV. |
| `v_threshold_mv` | float | Spike threshold, mV. |
| `v_reset_mv` | float | Post-spike reset, mV. |
| `tau_m_ms` | float > 0 | Membrane time constant, ms. |
| `r_m_mohm` | float > 0 | Membrane resistance, MΩ. |

**Caveat:** all parameter values in this repository are textbook placeholders.
None has been fitted to a measurement.

## Synapse

```json
{ "pre": 0, "post": 1, "weight": 1.0, "delay_ms": 1.0, "sign": 1 }
```

| Field | Type | Meaning |
|---|---|---|
| `pre` | int ≥ 0 | Presynaptic neuron id. |
| `post` | int ≥ 0 | Postsynaptic neuron id. |
| `weight` | float | Unitless placeholder weight. |
| `delay_ms` | float ≥ 0 | Conduction delay, ms. |
| `sign` | +1 / −1 | Excitatory / inhibitory. |

`effective_weight = weight × sign`. Weights are not calibrated against measured
postsynaptic potentials.

## Region

```json
{
  "name": "cortical_column_toy",
  "neuron_count": 1000,
  "cell_types": { "exc": 0.8, "inh": 0.2 },
  "notes": "Toy composition for interface testing only."
}
```

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Unique region name. |
| `neuron_count` | int ≥ 0 | Neuron budget for the region. |
| `cell_types` | map<string, float> | Fractions; **must sum to 1.0**. |
| `notes` | string | Provenance / caveats. |

A region is a *declaration*. It does not load an atlas, and its composition is
not anatomically derived unless `notes` says so.

## Connectome (fixture format)

```json
{
  "n_neurons": 4,
  "pre":    [0, 0, 1, 2, 2, 3],
  "post":   [1, 2, 2, 3, 0, 1],
  "weight": [1.0, 0.5, 1.5, 1.0, -1.0, 0.75],
  "delay_ms": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0]
}
```

Coordinate-list (COO) form. All four arrays must be equal length. This format is
for **small fixtures only** — a real connectome is never stored this way.

### Storage budget

| Representation | Bytes / synapse | 10¹⁴ synapses |
|---|---|---|
| COO, int32/int32/float32/float32 | 16 | ~1.6 TB |
| CSR, int32 indices + float32 weights | 8 | ~0.8 TB |
| CSR + int8 quantised weight | 5 | ~0.5 TB |

Even the most compact form is far beyond single-machine RAM, which is why
sharding and streaming are Phase 4 prerequisites rather than optimisations.

## Dataset manifest

```json
{
  "name": "example-mesoscale-tracer",
  "source": "Publication / provider",
  "url": "https://… or a DOI",
  "checksum": "sha256:…",
  "license": "CC-BY-4.0",
  "measured": true,
  "notes": "Preprocessing applied, caveats."
}
```

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Dataset identifier. |
| `source` | string | Publication or provider. |
| `url` | string | External location or stable DOI. |
| `checksum` | string | sha256 of the archive. |
| `license` | string | Usage terms. |
| `measured` | **bool, required** | `true` = measured, `false` = synthetic/statistical. |
| `notes` | string | Caveats, preprocessing. |

The `measured` flag is mandatory and is the single most important field in the
project. It is what stops a synthetic connectome from being mistaken for a
measured one.

## Run record (planned)

Not implemented yet. Intended shape:

```json
{
  "config": { "…": "…" },
  "seed": 0,
  "code_version": "0.0.1",
  "datasets": [ { "name": "…", "measured": true } ],
  "fidelity_map": { "region": "lif" },
  "measured_synapse_fraction": 0.0
}
```

The `measured_synapse_fraction` field is the honesty check: it states, for a
finished run, what proportion of synapses came from measured data.
