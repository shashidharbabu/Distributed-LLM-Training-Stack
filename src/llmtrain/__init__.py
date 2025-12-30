"""Distributed LLM training stack."""

__all__ = [
    "cli",
    "launcher",
    "train",
    "distributed",
    "checkpointing",
    "faults",
    "profiler",
    "metrics",
    "deepspeed_utils",
]

__version__ = "0.1.0"

# Marker for type checkers
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
