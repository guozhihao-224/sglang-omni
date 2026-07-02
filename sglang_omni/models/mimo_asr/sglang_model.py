# SPDX-License-Identifier: Apache-2.0
"""SGLang model entry for MiMo-V2.5-ASR.

This file intentionally starts with a minimal importable model class so the
SGLang registry path can be validated before the full MiMo prefill/decode port.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
from sglang.srt.layers.quantization.base_config import QuantizationConfig

from .configuration_mimo_asr import MiMoV2ASRConfig


class MiMoV2ASRForCausalLM(nn.Module):
    """Placeholder for the native MiMo-ASR SGLang model implementation."""

    def __init__(
        self,
        config: MiMoV2ASRConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix

    def pad_input_ids(self, input_ids: list[int], mm_inputs: Any):
        raise NotImplementedError("MiMo-ASR pad_input_ids is not implemented yet")

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("MiMo-ASR forward/decode is not implemented yet")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        raise NotImplementedError("MiMo-ASR weight loading is not implemented yet")


EntryClass = MiMoV2ASRForCausalLM


__all__ = ["MiMoV2ASRForCausalLM"]
