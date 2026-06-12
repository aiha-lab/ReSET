"""ReSET — Reasoning Step Entropy-based Temperature Scaling.

Online, step-aware decoding-temperature control that recovers reasoning
accuracy under NVFP4 quantization. See :mod:`reset.logits_processor`.
"""

from __future__ import annotations

from .logits_processor import (
    ReSETAdapter,
    ReSETRequest,
    entropy_of,
    get_newline_token_ids,
    scale_by_temperature,
)

__all__ = [
    "ReSETAdapter",
    "ReSETRequest",
    "get_newline_token_ids",
    "entropy_of",
    "scale_by_temperature",
]
__version__ = "0.1.0"
