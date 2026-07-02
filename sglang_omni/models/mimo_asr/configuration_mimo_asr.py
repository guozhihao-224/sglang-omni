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
        local_dim: int = 1024,
        local_layers: int = 16,
        local_attn_heads: int = 64,
        local_ffn_dim: int = 4096,
        local_attn_dropout: float = 0.1,
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
        self.local_dim = local_dim
        self.local_layers = local_layers
        self.local_attn_heads = local_attn_heads
        self.local_ffn_dim = local_ffn_dim
        self.local_attn_dropout = local_attn_dropout
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

    def local_config(self):
        config = self._copy_for_local_transformer()
        config.hidden_size = int(self.local_dim)
        config.num_hidden_layers = int(self.local_layers)
        config.num_attention_heads = int(self.local_attn_heads)
        config.num_key_value_heads = int(self.local_attn_heads)
        config.intermediate_size = int(self.local_ffn_dim)
        config.attention_dropout = float(self.local_attn_dropout)
        config.vocab_size = 1
        config.head_dim = config.hidden_size // config.num_attention_heads
        return config

    def input_local_config(self):
        config = self._copy_for_local_transformer()
        config.hidden_size = int(self.input_local_dim)
        config.num_hidden_layers = int(self.input_local_layers)
        config.num_attention_heads = int(self.local_attn_heads)
        config.num_key_value_heads = int(self.local_attn_heads)
        config.intermediate_size = config.hidden_size * 4
        config.attention_dropout = float(self.local_attn_dropout)
        config.vocab_size = 1
        config.head_dim = config.hidden_size // config.num_attention_heads
        return config

    def _copy_for_local_transformer(self):
        return self.__class__(**self.to_dict())


def _parse_dash_ints(value: str) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split("-")]


def coerce_mimo_asr_config(config) -> MiMoV2ASRConfig:
    """Return a MiMo config even when SGLang loaded a plain Qwen2Config."""

    if isinstance(config, MiMoV2ASRConfig):
        return config
    if hasattr(config, "to_dict"):
        return MiMoV2ASRConfig(**config.to_dict())
    if isinstance(config, dict):
        return MiMoV2ASRConfig(**config)
    raise TypeError(f"unsupported MiMo config type: {type(config).__name__}")


__all__ = ["MiMoV2ASRConfig", "coerce_mimo_asr_config"]
