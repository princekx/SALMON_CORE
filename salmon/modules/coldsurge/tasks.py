"""Compatibility shim for Cold Surge task imports.

This module re-exports model-specific task classes so existing recipes using
`salmon.modules.coldsurge.tasks` continue to work.
"""

from salmon.modules.coldsurge.gpm_tasks import (
    DisplayGPMColdSurgeMaps,
    RetrieveGPMColdSurgeData,
)
from salmon.modules.coldsurge.mogreps_tasks import (
    ComputeMogrepsColdSurgeIndices,
    DisplayMogrepsColdSurgeMaps,
    RetrieveMogrepsColdSurgeData,
)

__all__ = [
    "RetrieveMogrepsColdSurgeData",
    "ComputeMogrepsColdSurgeIndices",
    "DisplayMogrepsColdSurgeMaps",
    "RetrieveGPMColdSurgeData",
    "DisplayGPMColdSurgeMaps",
]
