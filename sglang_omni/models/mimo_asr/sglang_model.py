# SPDX-License-Identifier: Apache-2.0
"""SGLang model entry for MiMo-V2.5-ASR.

This file intentionally starts with a minimal importable model class so the
SGLang registry path can be validated before the full MiMo prefill/decode port.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

from .configuration_mimo_asr import MiMoV2ASRConfig, coerce_mimo_asr_config

_LANGUAGE_MODEL_PREFIX = "model."
_SUPPORTED_DIRECT_PREFIXES = (
    "lm_head.",
    "speech_embeddings.",
    "speech_group_downcast.",
    "hidden_states_downcast.",
)
_PENDING_MIMO_PREFIXES = (
    "input_local_transformer.",
    "local_transformer.",
    "local_transformer_lm_heads.",
)


def validate_mimo_speech_config(config: MiMoV2ASRConfig) -> None:
    """Validate MiMo audio-code channel metadata."""

    if int(config.audio_channels) < 1:
        raise ValueError(f"audio_channels must be >= 1, got {config.audio_channels}")
    if int(config.group_size) < 1:
        raise ValueError(f"group_size must be >= 1, got {config.group_size}")
    if int(config.local_attn_heads) < 1:
        raise ValueError(
            f"local_attn_heads must be >= 1, got {config.local_attn_heads}"
        )
    if int(config.input_local_dim) % int(config.local_attn_heads) != 0:
        raise ValueError(
            "input_local_dim must be divisible by local_attn_heads "
            f"({config.input_local_dim} % {config.local_attn_heads})"
        )
    if int(config.local_dim) % int(config.local_attn_heads) != 0:
        raise ValueError(
            "local_dim must be divisible by local_attn_heads "
            f"({config.local_dim} % {config.local_attn_heads})"
        )

    speech_vocab_sizes = config.speech_vocab_sizes
    speech_zeroemb_indices = config.speech_zeroemb_indices
    delay_pattern_values = config.delay_pattern_values
    audio_channels = int(config.audio_channels)

    if len(speech_vocab_sizes) != audio_channels:
        raise ValueError(
            "speech_vocab_size channel count must match audio_channels "
            f"({len(speech_vocab_sizes)} != {audio_channels})"
        )
    if len(speech_zeroemb_indices) != audio_channels:
        raise ValueError(
            "speech_zeroemb_idx channel count must match audio_channels "
            f"({len(speech_zeroemb_indices)} != {audio_channels})"
        )
    if len(delay_pattern_values) != audio_channels:
        raise ValueError(
            "delay_pattern channel count must match audio_channels "
            f"({len(delay_pattern_values)} != {audio_channels})"
        )

    for channel, (vocab_size, zeroemb_idx) in enumerate(
        zip(speech_vocab_sizes, speech_zeroemb_indices, strict=True)
    ):
        if vocab_size < 1:
            raise ValueError(
                f"speech vocab size for channel {channel} must be >= 1, got {vocab_size}"
            )
        if zeroemb_idx < 0 or zeroemb_idx >= vocab_size:
            raise ValueError(
                f"speech zeroemb index for channel {channel} must be in "
                f"[0, {vocab_size}), got {zeroemb_idx}"
            )


def normalize_mimo_audio_codes(
    codes: Any,
    *,
    audio_channels: int = 8,
) -> torch.Tensor:
    """Normalize audio codes to contiguous ``[frames, channels]`` int64."""

    audio_codes = torch.as_tensor(codes, dtype=torch.long)
    if audio_codes.ndim == 3 and audio_codes.shape[0] == 1:
        audio_codes = audio_codes.squeeze(0)
    if audio_codes.ndim != 2:
        raise ValueError(
            f"MiMo audio codes must be 2-D, got shape {tuple(audio_codes.shape)}"
        )
    if audio_codes.shape[1] == audio_channels:
        return audio_codes.contiguous()
    if audio_codes.shape[0] == audio_channels:
        return audio_codes.transpose(0, 1).contiguous()
    raise ValueError(
        "MiMo audio codes must have audio_channels in shape [T, C] or [C, T], "
        f"got {tuple(audio_codes.shape)} with audio_channels={audio_channels}"
    )


def pad_mimo_audio_codes_to_group(
    codes: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Pad ``[frames, channels]`` codes by repeating the last frame."""

    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if codes.ndim != 2:
        raise ValueError(f"MiMo audio codes must be 2-D, got shape {tuple(codes.shape)}")
    num_frames = int(codes.shape[0])
    if num_frames < 1:
        raise ValueError("MiMo audio codes must contain at least one frame")
    remainder = num_frames % group_size
    if remainder == 0:
        return codes.contiguous()
    pad_frames = group_size - remainder
    return torch.cat([codes, codes[-1:].expand(pad_frames, -1)], dim=0).contiguous()


def group_mimo_audio_codes(
    codes: Any,
    *,
    audio_channels: int = 8,
    group_size: int = 4,
) -> torch.Tensor:
    """Return grouped codes as ``[groups, channels, group_size]``."""

    normalized_codes = normalize_mimo_audio_codes(codes, audio_channels=audio_channels)
    padded_codes = pad_mimo_audio_codes_to_group(
        normalized_codes,
        group_size=group_size,
    )
    num_groups = int(padded_codes.shape[0]) // group_size
    return (
        padded_codes.reshape(num_groups, group_size, audio_channels)
        .transpose(1, 2)
        .contiguous()
    )


class MiMoInputLocalTransformer(nn.Module):
    """Replaceable wrapper for MiMo's input-local transformer.

    This wrapper gives the prefill path a stable module boundary: Qwen2 by
    default for real configs, identity for zero-layer tests, or any injected
    ``nn.Module`` with the same tensor contract.
    """

    def __init__(self, module: nn.Module | None = None) -> None:
        super().__init__()
        self.module = module

    def forward(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        if self.module is None:
            return speech_embeddings
        return _run_local_transformer_module(self.module, speech_embeddings)


class MiMoLocalTransformer(nn.Module):
    """Replaceable wrapper for MiMo's decode-time local transformer."""

    def __init__(self, module: nn.Module | None = None) -> None:
        super().__init__()
        self.module = module

    def forward(self, local_hidden_states: torch.Tensor) -> torch.Tensor:
        if self.module is None:
            return local_hidden_states
        return _run_local_transformer_module(self.module, local_hidden_states)


def _run_local_transformer_module(
    module: nn.Module,
    inputs_embeds: torch.Tensor,
) -> torch.Tensor:
    if isinstance(module, Qwen2Model):
        outputs = module(inputs_embeds=inputs_embeds)
    else:
        outputs = module(inputs_embeds)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs[0] if isinstance(outputs, tuple) else outputs


def _build_local_qwen2_model(
    config: MiMoV2ASRConfig,
    *,
    input_local: bool,
) -> Qwen2Model:
    local_config = config.input_local_config() if input_local else config.local_config()
    model = Qwen2Model(local_config)
    model.embed_tokens = None
    return model


class MiMoV2ASRForCausalLM(nn.Module):
    """Placeholder for the native MiMo-ASR SGLang model implementation."""

    def __init__(
        self,
        config: MiMoV2ASRConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = coerce_mimo_asr_config(config)
        validate_mimo_speech_config(config)
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix
        self.language_model = None
        self.return_hidden_states_output = False
        self.audio_channels = int(config.audio_channels)
        self.group_size = int(config.group_size)
        self.speech_vocab_sizes = list(config.speech_vocab_sizes)
        self.speech_zeroemb_indices = list(config.speech_zeroemb_indices)
        self.delay_pattern_values = list(config.delay_pattern_values)

        input_local_dim = int(config.input_local_dim)
        hidden_size = int(config.hidden_size)
        self.speech_embeddings = nn.ModuleList(
            nn.Embedding(
                vocab_size,
                input_local_dim,
                padding_idx=zeroemb_idx,
            )
            for vocab_size, zeroemb_idx in zip(
                self.speech_vocab_sizes,
                self.speech_zeroemb_indices,
                strict=True,
            )
        )
        self.speech_group_downcast = nn.Linear(
            self.group_size * input_local_dim,
            hidden_size,
            bias=False,
        )
        self.input_local_transformer = MiMoInputLocalTransformer(
            _build_local_qwen2_model(config, input_local=True)
            if int(config.input_local_layers) > 0
            else None
        )
        self.hidden_states_downcast = nn.Linear(
            hidden_size,
            int(config.local_dim),
            bias=False,
        )
        self.local_transformer = MiMoLocalTransformer(
            _build_local_qwen2_model(config, input_local=False)
            if int(config.local_layers) > 0
            else None
        )
        self.local_transformer_lm_heads = nn.ModuleList(
            nn.Linear(int(config.local_dim), vocab_size, bias=False)
            for vocab_size in self.speech_vocab_sizes
        )

    def build_language_model(self) -> nn.Module:
        """Build and attach the SGLang Qwen2 text backbone.

        Kept explicit for now so helper/unit tests can instantiate this class
        without constructing a large Qwen2 model.  The full native forward path
        should call this during model initialization once weight loading and
        prefill/decode integration are complete.
        """

        if self.language_model is None:
            self.language_model = self._build_language_model()
        return self.language_model

    def _build_language_model(self) -> nn.Module:
        from sglang.srt.models.qwen2 import Qwen2ForCausalLM

        return Qwen2ForCausalLM(
            self.config,
            self.quant_config,
            prefix=add_prefix("model", self.prefix),
        )

    def embed_grouped_audio_codes(self, codes: Any) -> torch.Tensor:
        """Embed MiMo audio codes as ``[groups, group_size, input_local_dim]``.

        This covers the deterministic front half of official prefill embedding:
        normalize/group code frames, lookup one embedding table per RVQ channel,
        mask each channel's zeroemb/padding token, then sum channels.
        """

        grouped_codes = group_mimo_audio_codes(
            codes,
            audio_channels=self.audio_channels,
            group_size=self.group_size,
        ).to(next(self.speech_embeddings.parameters()).device)
        embeddings: torch.Tensor | None = None
        for channel_idx, embedding in enumerate(self.speech_embeddings):
            channel_codes = grouped_codes[:, channel_idx, :]
            channel_embeddings = embedding(channel_codes)
            zeroemb_idx = self.speech_zeroemb_indices[channel_idx]
            channel_embeddings = channel_embeddings.masked_fill(
                (channel_codes == zeroemb_idx).unsqueeze(-1),
                0.0,
            )
            embeddings = (
                channel_embeddings
                if embeddings is None
                else embeddings + channel_embeddings
            )
        assert embeddings is not None
        return embeddings

    def apply_input_local_transformer(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply MiMo's input local transformer.

        The wrapper is identity until the official input-local transformer is
        ported, but tests and future code can replace ``self.input_local_transformer``
        without changing the prefill projection flow.
        """

        return self.input_local_transformer(speech_embeddings)

    def project_grouped_audio_embeds(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        """Project grouped speech embeddings to text hidden size."""

        if speech_embeddings.ndim != 3:
            raise ValueError(
                "speech_embeddings must be [groups, group_size, dim], got "
                f"shape {tuple(speech_embeddings.shape)}"
            )
        if speech_embeddings.shape[1] != self.group_size:
            raise ValueError(
                f"speech group dimension must be {self.group_size}, got "
                f"{speech_embeddings.shape[1]}"
            )
        transformed = self.apply_input_local_transformer(speech_embeddings)
        flattened = transformed.reshape(transformed.shape[0], -1)
        return self.speech_group_downcast(flattened)

    def encode_audio_codes_to_hidden(self, codes: Any) -> torch.Tensor:
        """Encode MiMo audio codes into ``[groups, hidden_size]`` embeddings."""

        return self.project_grouped_audio_embeds(self.embed_grouped_audio_codes(codes))

    def project_hidden_states_to_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project LM hidden states to MiMo local-transformer dimension."""

        if hidden_states.ndim < 2:
            raise ValueError(
                "hidden_states must end with hidden_size, got "
                f"shape {tuple(hidden_states.shape)}"
            )
        if int(hidden_states.shape[-1]) != int(self.config.hidden_size):
            raise ValueError(
                "hidden_states last dimension must match hidden_size "
                f"({hidden_states.shape[-1]} != {self.config.hidden_size})"
            )
        return self.hidden_states_downcast(hidden_states)

    def compute_local_code_logits(self, local_hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """Compute per-channel MiMo RVQ logits from local hidden states."""

        if local_hidden_states.ndim < 2:
            raise ValueError(
                "local_hidden_states must end with local_dim, got "
                f"shape {tuple(local_hidden_states.shape)}"
            )
        if int(local_hidden_states.shape[-1]) != int(self.config.local_dim):
            raise ValueError(
                "local_hidden_states last dimension must match local_dim "
                f"({local_hidden_states.shape[-1]} != {self.config.local_dim})"
            )
        transformed = self.local_transformer(local_hidden_states)
        return [head(transformed) for head in self.local_transformer_lm_heads]

    def sample_local_code_ids(
        self,
        logits: list[torch.Tensor],
        *,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_p: float = 0.95,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample MiMo RVQ code ids from per-channel local logits."""

        if len(logits) != self.audio_channels:
            raise ValueError(
                "local logits channel count must match audio_channels "
                f"({len(logits)} != {self.audio_channels})"
            )
        sampled_channels = [
            self._sample_logits(
                channel_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            for channel_logits in logits
        ]
        return torch.stack(sampled_channels, dim=-1)

    def local_forward(
        self,
        hidden_states: torch.Tensor,
        *,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_p: float = 0.95,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate per-channel MiMo local code ids from LM hidden states."""

        local_hidden_states = self.project_hidden_states_to_local(hidden_states)
        logits = self.compute_local_code_logits(local_hidden_states)
        return self.sample_local_code_ids(
            logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )

    def build_decode_token_group(
        self,
        text_token_id: int,
        speech_codes: Any | None = None,
        *,
        text_tail_token_id: int | None = None,
    ) -> torch.Tensor:
        """Build one flattened MiMo decode group.

        The group is represented as ``group_size`` columns of
        ``[text, rvq0, ..., rvqN]`` and flattened column-major, yielding
        ``(audio_channels + 1) * group_size`` token ids.
        """

        if speech_codes is None:
            codes = self._zero_speech_codes_for_decode_group()
        else:
            codes = self._normalize_decode_speech_codes(speech_codes)

        tail_token_id = (
            int(self.config.empty_token_id)
            if text_tail_token_id is None
            else int(text_tail_token_id)
        )
        rows = torch.empty(
            self.audio_channels + 1,
            self.group_size,
            dtype=torch.long,
            device=codes.device,
        )
        rows[0].fill_(tail_token_id)
        rows[0, 0] = int(text_token_id)
        rows[1:] = codes.transpose(0, 1)
        return rows.transpose(0, 1).reshape(-1).contiguous()

    def build_empty_decode_token_group(
        self,
        hidden_states: torch.Tensor,
        *,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_p: float = 0.95,
        generator: torch.Generator | None = None,
        text_tail_token_id: int | None = None,
    ) -> torch.Tensor:
        """Build a flattened decode group for a generated ``<|empty|>`` token."""

        speech_codes = self.local_forward(
            hidden_states,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        expected = (self.group_size, self.audio_channels)
        transposed = (self.audio_channels, self.group_size)
        if speech_codes.ndim != 2 or tuple(speech_codes.shape) not in {
            expected,
            transposed,
        }:
            raise ValueError(
                "local_forward for one decode group must return "
                "[group_size, audio_channels] or [audio_channels, group_size], "
                f"got {tuple(speech_codes.shape)}"
            )
        return self.build_decode_token_group(
            int(self.config.empty_token_id),
            speech_codes,
            text_tail_token_id=text_tail_token_id,
        )

    def build_decode_step(
        self,
        text_token_id: int,
        hidden_states: torch.Tensor | None = None,
        *,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_p: float = 0.95,
        generator: torch.Generator | None = None,
        text_tail_token_id: int | None = None,
    ) -> tuple[torch.Tensor, bool]:
        """Build one MiMo decode step and report whether it stops generation."""

        text_token_id = int(text_token_id)
        if text_token_id == int(self.config.empty_token_id):
            if hidden_states is None:
                raise ValueError("hidden_states are required for <|empty|> decode")
            group = self.build_empty_decode_token_group(
                hidden_states,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
                text_tail_token_id=text_tail_token_id,
            )
        else:
            group = self.build_decode_token_group(
                text_token_id,
                text_tail_token_id=text_tail_token_id,
            )
        return group, text_token_id == int(self.config.stop_token_id)

    def _zero_speech_codes_for_decode_group(self) -> torch.Tensor:
        return torch.tensor(
            self.speech_zeroemb_indices,
            dtype=torch.long,
        ).unsqueeze(0).expand(self.group_size, -1).contiguous()

    def _normalize_decode_speech_codes(self, speech_codes: Any) -> torch.Tensor:
        codes = torch.as_tensor(speech_codes, dtype=torch.long)
        expected = (self.group_size, self.audio_channels)
        transposed = (self.audio_channels, self.group_size)
        if tuple(codes.shape) == expected:
            return codes.contiguous()
        if tuple(codes.shape) == transposed:
            return codes.transpose(0, 1).contiguous()
        raise ValueError(
            "speech_codes must be [group_size, audio_channels] or "
            "[audio_channels, group_size], got "
            f"shape {tuple(codes.shape)}; expected {expected}"
        )

    @staticmethod
    def _sample_logits(
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if logits.ndim < 1:
            raise ValueError("logits must have at least one dimension")
        if not do_sample:
            return torch.argmax(logits, dim=-1)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if top_p <= 0 or top_p > 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        original_shape = logits.shape[:-1]
        vocab_size = int(logits.shape[-1])
        flat_logits = logits.reshape(-1, vocab_size) / float(temperature)
        probs = torch.softmax(flat_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative_probs > float(top_p)
        remove_mask[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_sorted = torch.multinomial(
            sorted_probs,
            1,
            generator=generator,
        ).squeeze(-1)
        sampled = sorted_indices.gather(1, sampled_sorted.unsqueeze(-1)).squeeze(-1)
        return sampled.reshape(original_shape)

    def get_audio_feature(self, items: list[Any]) -> torch.Tensor:
        """Encode multimodal audio items into hidden-size embeddings.

        ``request_builders`` stores MiMo RVQ codes under
        ``item.model_specific_data["audio_codes"]`` and also mirrors them in
        ``item.feature``.  Prefer model-specific data so later metadata can
        evolve without changing the tensor field contract.
        """

        if not items:
            raise ValueError("MiMo-ASR get_audio_feature requires at least one item")
        encoded_items: list[torch.Tensor] = []
        for item in items:
            codes = self._audio_codes_from_item(item)
            encoded_items.append(self.encode_audio_codes_to_hidden(codes))
        return torch.cat(encoded_items, dim=0)

    @staticmethod
    def _audio_codes_from_item(item: Any) -> Any:
        model_specific_data = getattr(item, "model_specific_data", None) or {}
        codes = model_specific_data.get("audio_codes", getattr(item, "feature", None))
        if codes is None:
            raise ValueError("MiMo-ASR audio item is missing audio_codes/feature")
        return codes

    def pad_input_ids(self, input_ids: list[int], mm_inputs: Any):
        """Replace MiMo ``<|empty|>`` placeholders with item pad values.

        SGLang's multimodal embedding scatter uses each item's ``pad_value`` and
        inclusive ``offsets`` to locate positions.  Request builders normally
        precompute those fields, but this hook keeps the model robust when it is
        invoked through SGLang's native multimodal padding path.
        """

        mm_items = list(getattr(mm_inputs, "mm_items", None) or [])
        if not mm_items:
            return input_ids

        padded_ids = list(input_ids)
        empty_positions = [
            idx
            for idx, token_id in enumerate(padded_ids)
            if token_id == self.config.empty_token_id
        ]
        consumed_empty_positions = 0

        for item in mm_items:
            expected_tokens = int(
                self.encode_audio_codes_to_hidden(
                    self._audio_codes_from_item(item)
                ).shape[0]
            )
            self._ensure_item_pad_value(item)
            pad_value = item.pad_value
            offsets = list(getattr(item, "offsets", None) or [])
            if offsets:
                positions = self._positions_from_offsets(offsets)
                if len(positions) != expected_tokens:
                    raise ValueError(
                        "MiMo-ASR offset span length must match audio hidden groups "
                        f"({len(positions)} != {expected_tokens})"
                    )
            else:
                positions = empty_positions[
                    consumed_empty_positions : consumed_empty_positions + expected_tokens
                ]
                if len(positions) != expected_tokens:
                    raise ValueError(
                        "MiMo-ASR input_ids do not contain enough <|empty|> placeholders "
                        f"({len(positions)} != {expected_tokens})"
                    )
                self._set_item_offsets(item, positions)
                consumed_empty_positions += expected_tokens

            for position in positions:
                if position < 0 or position >= len(padded_ids):
                    raise ValueError(
                        f"MiMo-ASR offset position {position} is outside "
                        f"input length {len(padded_ids)}"
                    )
                token_id = padded_ids[position]
                if token_id not in {self.config.empty_token_id, pad_value}:
                    raise ValueError(
                        "MiMo-ASR offset points to non-placeholder token "
                        f"at position {position}: {token_id}"
                    )
                padded_ids[position] = pad_value

        return padded_ids

    def merge_audio_embeds_into_token_embeds(
        self,
        token_embeds: torch.Tensor,
        items: list[Any],
    ) -> torch.Tensor:
        """Scatter encoded audio embeddings into token embeddings.

        This is the prefill merge primitive used before calling the text
        backbone with ``input_embeds``.  It expects item offsets to be inclusive
        and already aligned with hidden groups.
        """

        if token_embeds.ndim != 2:
            raise ValueError(
                "token_embeds must be [seq_len, hidden_size], got "
                f"shape {tuple(token_embeds.shape)}"
            )
        merged = token_embeds.clone()
        for item in items:
            offsets = list(getattr(item, "offsets", None) or [])
            if not offsets:
                raise ValueError("MiMo-ASR audio item must have offsets before scatter")
            positions = self._positions_from_offsets(offsets)
            audio_hidden = self.encode_audio_codes_to_hidden(self._audio_codes_from_item(item)).to(
                device=merged.device,
                dtype=merged.dtype,
            )
            if len(positions) != int(audio_hidden.shape[0]):
                raise ValueError(
                    "MiMo-ASR scatter positions must match audio hidden groups "
                    f"({len(positions)} != {audio_hidden.shape[0]})"
                )
            if int(audio_hidden.shape[1]) != int(merged.shape[1]):
                raise ValueError(
                    "MiMo-ASR audio hidden size must match token embedding size "
                    f"({audio_hidden.shape[1]} != {merged.shape[1]})"
                )
            for row_idx, position in enumerate(positions):
                if position < 0 or position >= int(merged.shape[0]):
                    raise ValueError(
                        f"MiMo-ASR scatter position {position} is outside "
                        f"sequence length {merged.shape[0]}"
                    )
                merged[position] = audio_hidden[row_idx]
        return merged

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        token_embedding: nn.Module,
        items: list[Any] | None = None,
    ) -> torch.Tensor:
        """Embed text ids and optionally scatter MiMo audio embeddings."""

        if input_ids.ndim != 1:
            raise ValueError(
                f"input_ids must be 1-D for MiMo-ASR prefill, got shape {tuple(input_ids.shape)}"
            )
        token_embeds = token_embedding(input_ids.to(dtype=torch.long))
        if token_embeds.ndim != 2:
            raise ValueError(
                "token_embedding must return [seq_len, hidden_size], got "
                f"shape {tuple(token_embeds.shape)}"
            )
        if not items:
            return token_embeds
        return self.merge_audio_embeds_into_token_embeds(token_embeds, items)

    def prepare_prefill_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        items: list[Any] | None = None,
    ) -> torch.Tensor:
        """Prepare text/audio ``input_embeds`` for the language backbone."""

        token_embedding = self._get_token_embedding_module()
        safe_input_ids = self._restore_placeholder_ids_for_embedding(
            input_ids,
            items,
            token_embedding,
        )
        return self.embed_input_ids(
            safe_input_ids,
            token_embedding,
            items,
        )

    def _restore_placeholder_ids_for_embedding(
        self,
        input_ids: torch.Tensor,
        items: list[Any] | None,
        token_embedding: nn.Module,
    ) -> torch.Tensor:
        if not items:
            return input_ids
        safe_input_ids = input_ids.clone()
        placeholder_id = self._safe_placeholder_id_for_embedding(token_embedding)
        for item in items:
            for position in self._positions_from_offsets(
                list(getattr(item, "offsets", None) or [])
            ):
                if position < 0 or position >= int(safe_input_ids.shape[0]):
                    raise ValueError(
                        f"MiMo-ASR offset position {position} is outside "
                        f"input length {safe_input_ids.shape[0]}"
                    )
                safe_input_ids[position] = placeholder_id
        return safe_input_ids

    def _safe_placeholder_id_for_embedding(self, token_embedding: nn.Module) -> int:
        placeholder_id = int(self.config.empty_token_id)
        num_embeddings = getattr(token_embedding, "num_embeddings", None)
        if num_embeddings is None:
            weight = getattr(token_embedding, "weight", None)
            num_embeddings = int(weight.shape[0]) if weight is not None else None
        if num_embeddings is not None and placeholder_id >= int(num_embeddings):
            return 0
        return placeholder_id

    def _get_token_embedding_module(self) -> nn.Module:
        language_model = self.build_language_model()
        get_input_embeddings = getattr(language_model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            return get_input_embeddings()
        for attr_path in (
            ("model", "embed_tokens"),
            ("language_model", "embed_tokens"),
            ("embed_tokens",),
        ):
            module: Any = language_model
            for attr in attr_path:
                module = getattr(module, attr, None)
                if module is None:
                    break
            if isinstance(module, nn.Module):
                return module
        raise AttributeError("MiMo-ASR language model does not expose token embeddings")

    @staticmethod
    def _extract_mm_items_from_forward_batch(forward_batch: Any) -> list[Any]:
        for attr in ("multimodal_inputs", "mm_inputs"):
            mm_inputs = getattr(forward_batch, attr, None)
            if mm_inputs is not None:
                return list(getattr(mm_inputs, "mm_items", None) or [])
        batch = getattr(forward_batch, "batch", None)
        if batch is not None:
            for attr in ("multimodal_inputs", "mm_inputs"):
                mm_inputs = getattr(batch, attr, None)
                if mm_inputs is not None:
                    return list(getattr(mm_inputs, "mm_items", None) or [])
        return []

    @staticmethod
    def _ensure_item_pad_value(item: Any) -> None:
        if getattr(item, "pad_value", None) is not None:
            return
        set_pad_value = getattr(item, "set_pad_value", None)
        if callable(set_pad_value):
            set_pad_value()
        if getattr(item, "pad_value", None) is None:
            raise ValueError("MiMo-ASR multimodal item is missing pad_value")

    @staticmethod
    def _positions_from_offsets(offsets: list[tuple[int, int]]) -> list[int]:
        positions: list[int] = []
        for start, end in offsets:
            if end < start:
                raise ValueError(f"MiMo-ASR invalid offset span ({start}, {end})")
            positions.extend(range(int(start), int(end) + 1))
        return positions

    @staticmethod
    def _set_item_offsets(item: Any, positions: list[int]) -> None:
        if not positions:
            item.offsets = []
            return
        expected = list(range(positions[0], positions[0] + len(positions)))
        if positions != expected:
            raise ValueError(
                "MiMo-ASR inferred placeholder positions must be contiguous"
            )
        item.offsets = [(positions[0], positions[-1])]


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        **kwargs: Any,
    ) -> torch.Tensor | LogitsProcessorOutput:
        language_model = self.build_language_model()
        input_embeds = kwargs.pop("input_embeds", None)
        if input_embeds is None:
            input_embeds = kwargs.pop("inputs_embeds", None)
        mm_items = kwargs.pop("mimo_mm_items", None)
        if input_embeds is None:
            if mm_items is None:
                mm_items = self._extract_mm_items_from_forward_batch(forward_batch)
            if mm_items:
                input_embeds = self.prepare_prefill_inputs_embeds(input_ids, mm_items)

        lm_kwargs = dict(kwargs)
        if input_embeds is not None:
            lm_kwargs["input_embeds"] = input_embeds
        lm_output = language_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            **lm_kwargs,
        )
        if not self.return_hidden_states_output:
            return lm_output
        return self._finalize_forward_logits_output(
            lm_output,
            input_ids=input_ids,
            forward_batch=forward_batch,
        )

    def _finalize_forward_logits_output(
        self,
        lm_output: torch.Tensor | LogitsProcessorOutput,
        *,
        input_ids: torch.Tensor,
        forward_batch: Any,
    ) -> LogitsProcessorOutput:
        """Return real text logits plus hidden states for MiMo decode hooks."""

        if isinstance(lm_output, LogitsProcessorOutput):
            return lm_output

        hidden_states = lm_output
        sample_hidden_states = self._select_sample_hidden_states(
            hidden_states,
            forward_batch,
        )
        next_token_logits = self._compute_text_logits(
            sample_hidden_states,
            input_ids=input_ids,
            forward_batch=forward_batch,
        )
        return LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=sample_hidden_states,
        )

    def _select_sample_hidden_states(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_extend = (
            forward_mode is not None
            and hasattr(forward_mode, "is_extend")
            and bool(forward_mode.is_extend())
        )
        if not is_extend:
            return hidden_states
        last_index = self._extend_last_index(forward_batch, hidden_states.device)
        return hidden_states[last_index]

    @staticmethod
    def _extend_last_index(forward_batch: Any, device: torch.device) -> torch.Tensor:
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return torch.tensor(
                [max(int(forward_batch.input_ids.shape[0]) - 1, 0)],
                device=device,
            )
        return torch.cumsum(extend_seq_lens.to(device=device), dim=0) - 1

    def _compute_text_logits(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        language_model = self.build_language_model()
        logits_processor = getattr(language_model, "logits_processor", None)
        lm_head = getattr(language_model, "lm_head", None)
        if logits_processor is None or lm_head is None:
            raise RuntimeError(
                "MiMo-ASR requires the language model to expose logits_processor "
                "and lm_head when return_hidden_states_output is enabled"
            )
        return logits_processor(
            input_ids,
            hidden_states,
            lm_head,
            forward_batch,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load currently implemented MiMo-ASR weights.

        This intentionally supports only the modules already present in this
        skeleton.  Qwen2 backbone and local-transformer weights are routed and
        guarded so missing implementation fails loudly instead of silently
        dropping required checkpoint tensors.
        """

        pending_language_weights: list[tuple[str, torch.Tensor]] = []
        pending_mimo_weights: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            route = route_mimo_weight_name(name)
            if route == "language_model":
                pending_language_weights.append((name, loaded_weight))
                continue
            if route == "pending_mimo":
                pending_mimo_weights.append((name, loaded_weight))
                continue
            if route == "unknown":
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)

        if pending_language_weights:
            if self.language_model is None:
                self.build_language_model()
            self.language_model.load_weights(pending_language_weights)
            loaded.update(name for name, _ in pending_language_weights)

        missing_mimo_prefixes: set[str] = set()
        for name, loaded_weight in pending_mimo_weights:
            if self._load_pending_mimo_weight(name, loaded_weight):
                loaded.add(name)
            else:
                missing_mimo_prefixes.add(name.split(".", 1)[0])

        if missing_mimo_prefixes:
            raise NotImplementedError(
                "MiMo-ASR local transformer weight loading is not implemented yet for prefixes: "
                + ", ".join(sorted(missing_mimo_prefixes))
            )

        return loaded

    def _load_pending_mimo_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        for prefix in _PENDING_MIMO_PREFIXES:
            if not name.startswith(prefix):
                continue
            module_name = prefix[:-1]
            module = getattr(self, module_name, None)
            if module is None:
                return False
            local_name = name.removeprefix(prefix)
            if _should_skip_local_transformer_weight(local_name):
                return True
            if isinstance(
                module,
                (MiMoInputLocalTransformer, MiMoLocalTransformer),
            ) and module.module is not None:
                module = module.module
            params_dict = dict(module.named_parameters(remove_duplicate=False))
            param = params_dict.get(local_name)
            if param is None:
                return False
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            return True
        return False


def _should_skip_local_transformer_weight(local_name: str) -> bool:
    return local_name == "embed_tokens.weight" or "rotary_emb." in local_name


def route_mimo_weight_name(name: str) -> str:
    """Classify MiMo checkpoint weights for staged loading."""

    if name.startswith(_LANGUAGE_MODEL_PREFIX):
        return "language_model"
    if name.startswith("lm_head."):
        return "language_model"
    if name.startswith(_SUPPORTED_DIRECT_PREFIXES):
        return "direct"
    if name.startswith(_PENDING_MIMO_PREFIXES):
        return "pending_mimo"
    return "unknown"


EntryClass = MiMoV2ASRForCausalLM


__all__ = [
    "MiMoV2ASRForCausalLM",
    "MiMoInputLocalTransformer",
    "MiMoLocalTransformer",
    "group_mimo_audio_codes",
    "normalize_mimo_audio_codes",
    "pad_mimo_audio_codes_to_group",
    "route_mimo_weight_name",
    "validate_mimo_speech_config",
]
