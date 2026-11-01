"""Visualization helpers.

Optional: requires the `viz` extra (matplotlib). Nothing here is imported at
package import time, so the core package has no plotting dependency.
"""

from .plots import plot_raster

__all__ = ["plot_raster"]
