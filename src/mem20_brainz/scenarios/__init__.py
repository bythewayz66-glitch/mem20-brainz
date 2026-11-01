"""Scenario builders.

A scenario is a fully specified, reproducible network plus the metadata needed
to describe it honestly (provenance, baseline parameters, region layout).

Everything here is synthetic. See :mod:`mem20_brainz.scenarios.demo_network`.
"""

from .demo_network import (
    DEFAULT_REGION_SIZES,
    DEFAULT_THREAT_REGION,
    EXCITATORY_REGIONS,
    DemoNetwork,
    RegionSpec,
    build_demo_network,
)

__all__ = [
    "DEFAULT_REGION_SIZES",
    "DEFAULT_THREAT_REGION",
    "EXCITATORY_REGIONS",
    "DemoNetwork",
    "RegionSpec",
    "build_demo_network",
]
