"""Dataset ingestion: resolve, download, verify, and register datasets."""

from signal_engine.ingestion.base import (
    DatasetHandle,
    DatasetSource,
    DownloadResult,
    fingerprint_file,
    register_local_dataset,
)
from signal_engine.ingestion.tlc import TLCSource, TLCVehicle

__all__ = [
    "DatasetHandle",
    "DatasetSource",
    "DownloadResult",
    "TLCSource",
    "TLCVehicle",
    "fingerprint_file",
    "register_local_dataset",
]
