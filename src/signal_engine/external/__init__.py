"""Optional external services: Brave Search grounding and context compression."""

from signal_engine.external.brave import BraveResult, BraveSearchClient
from signal_engine.external.compression import (
    CompressionResult,
    ContextCompressor,
    NoOpContextCompressor,
    TokenCompanyCompressor,
    build_compressor,
)

__all__ = [
    "BraveResult",
    "BraveSearchClient",
    "CompressionResult",
    "ContextCompressor",
    "NoOpContextCompressor",
    "TokenCompanyCompressor",
    "build_compressor",
]
