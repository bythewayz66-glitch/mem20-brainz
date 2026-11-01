# data/

**This directory holds schemas and tiny fixtures only.**

## The rule

Large connectome datasets are **never** committed to this repository. Not in a
branch, not in a release asset, not in Git LFS. A single electron-microscopy
volume or a full human connectome is orders of magnitude larger than a Git
repository should ever be, and Git cannot diff or garbage-collect it usefully.

`.gitignore` already blocks the common binary formats (`*.h5`, `*.nwb`,
`*.npy`, `*.npz`, `*.parquet`, `*.zarr/`, `*.sqlite`, …) and the `data/raw/`,
`data/external/` and `data/cache/` directories.

## Where real data lives

External object storage, referenced by a manifest committed here. A manifest is
a small JSON file recording, for each dataset:

| field | meaning |
|---|---|
| `name` | dataset identifier |
| `source` | publication / provider |
| `url` | external location (or a stable DOI) |
| `checksum` | sha256 of the archive |
| `license` | usage terms |
| `measured` | `true` if measured, `false` if synthetic/statistical |
| `notes` | caveats, preprocessing applied |

The `measured` flag is mandatory. Every dataset in this project must be
labelled as measured or synthetic, and any figure derived from it must carry
that label forward.

## Layout

```
data/
├── README.md          # this file
├── schemas/           # JSON Schema definitions for our internal formats
└── fixtures/          # tiny hand-written examples used by tests
```

## Fixtures

Fixtures are hand-written, a few dozen synapses at most, and exist purely so
tests can exercise the I/O path. They are not samples of any real connectome.
