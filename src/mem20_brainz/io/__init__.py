"""I/O layer.

Readers and writers for configuration and small connectome fixtures. Large
datasets are expected to live in external object storage and be streamed; no
such reader exists yet.
"""

from .readers import load_config, load_connectome, save_connectome

__all__ = ["load_config", "load_connectome", "save_connectome"]
