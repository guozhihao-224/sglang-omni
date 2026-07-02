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
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

from .configuration_mimo_asr import MiMoV2ASRConfig

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


class MiMoV2ASRForCausalLM(nn.Module):
    """Placeholder for the native MiMo-ASR SGLang model implementation."""

    def __init__(
        self,
        config: MiMoV2ASRConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        validate_mimo_speech_config(config)
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix
        self.language_model = None
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
        )
        self.hidden_states_downcast = nn.Linear(
            hidden_size,
            input_local_dim,
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

        The real transformer port is a later phase.  Keeping this as a separate
        hook makes the current embedding/downcast path testable and easy to
        replace with the official module.
        """

        return speech_embeddings

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
        backbone with ``inputs_embeds``.  It expects item offsets to be inclusive
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


    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("MiMo-ASR forward/decode is not implemented yet")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load currently implemented MiMo-ASR weights.

        This intentionally supports only the modules already present in this
        skeleton.  Qwen2 backbone and local-transformer weights are routed and
        guarded so missing implementation fails loudly instead of silently
        dropping required checkpoint tensors.
        """

        pending_language_weights: list[tuple[str, torch.Tensor]] = []
        pending_mimo_prefixes: set[str] = set()
        loaded: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            route = route_mimo_weight_name(name)
            if route == "language_model":
                pending_language_weights.append((name.removeprefix(_LANGUAGE_MODEL_PREFIX), loaded_weight))
                continue
            if route == "pending_mimo":
                pending_mimo_prefixes.add(name.split(".", 1)[0])
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
                raise NotImplementedError(
                    "MiMo-ASR Qwen2 language_model is not built yet; cannot load model.* weights"
                )
            self.language_model.load_weights(pending_language_weights)
            loaded.update(f"{_LANGUAGE_MODEL_PREFIX}{name}" for name, _ in pending_language_weights)

        if pending_mimo_prefixes:
            raise NotImplementedError(
                "MiMo-ASR local transformer weight loading is not implemented yet for prefixes: "
                + ", ".join(sorted(pending_mimo_prefixes))
            )

        return loaded


def route_mimo_weight_name(name: str) -> str:
    """Classify MiMo checkpoint weights for staged loading."""

    if name.startswith(_LANGUAGE_MODEL_PREFIX):
        return "language_model"
    if name.startswith(_SUPPORTED_DIRECT_PREFIXES):
        return "direct"
    if name.startswith(_PENDING_MIMO_PREFIXES):
        return "pending_mimo"
    return "unknown"


EntryClass = MiMoV2ASRForCausalLM


__all__ = [
    "MiMoV2ASRForCausalLM",
    "group_mimo_audio_codes",
    "normalize_mimo_audio_codes",
    "pad_mimo_audio_codes_to_group",
    "route_mimo_weight_name",
    "validate_mimo_speech_config",
]
