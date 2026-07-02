# SPDX-License-Identifier: Apache-2.0
"""Configuration helpers for MiMo-V2.5-ASR.

The released MiMo-V2.5-ASR checkpoint uses ``model_type: qwen2`` with a flat
Qwen2 text-backbone config plus MiMo-specific audio-code fields.  We keep this
class local for type/default centralization and do not override HuggingFace's
global ``qwen2`` AutoConfig registration.
"""

from __future__ import annotations

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class MiMoV2ASRConfig(Qwen2Config):
    """Qwen2-compatible MiMo-ASR config with audio-code defaults."""

    model_type = "qwen2"

    def __init__(
        self,
        *,
        audio_channels: int = 8,
        group_size: int = 4,
        input_local_layers: int = 6,
        input_local_dim: int = 1024,
        input_full_attention: bool = True,
        speech_vocab_size: str = "1025-1025-129-129-129-129-129-129",
        speech_zeroemb_idx: str = "1024-1024-128-128-128-128-128-128",
        delay_pattern: str = "0-1-2-3-4-5-6-7",
        empty_token_id: int = 151667,
        stop_token_id: int = 151645,
        rope_theta: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if rope_theta is not None:
            self.rope_theta = rope_theta
        self.audio_channels = audio_channels
        self.group_size = group_size
        self.input_local_layers = input_local_layers
        self.input_local_dim = input_local_dim
        self.input_full_attention = input_full_attention
        self.speech_vocab_size = speech_vocab_size
        self.speech_zeroemb_idx = speech_zeroemb_idx
        self.delay_pattern = delay_pattern
        self.empty_token_id = empty_token_id
        self.stop_token_id = stop_token_id

    @property
    def speech_vocab_sizes(self) -> list[int]:
        return _parse_dash_ints(self.speech_vocab_size)

    @property
    def speech_zeroemb_indices(self) -> list[int]:
        return _parse_dash_ints(self.speech_zeroemb_idx)

    @property
    def delay_pattern_values(self) -> list[int]:
        return _parse_dash_ints(self.delay_pattern)

    def get_text_config(self, decoder: bool = False):
        return self


def _parse_dash_ints(value: str) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split("-")]


__all__ = ["MiMoV2ASRConfig"]
