# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-Transcribe-Diarize model."""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.whisper import WhisperEncoder
from sglang.srt.utils import add_prefix

from sglang_omni.models.moss_transcribe_diarize.hf_config import (
    MossTranscribeDiarizeConfig,
)

logger = logging.getLogger(__name__)

# Upper bound on mel chunks per Whisper encoder forward. Bounds peak
# activation memory when many requests' audio windows are batched together
# (each chunk costs ~seq_len x ffn_dim of transient activations per layer).
_ENCODER_MAX_BATCH_CHUNKS = 128


def _validate_audio_item(
    item: MultimodalDataItem,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check one audio item and return (features, lengths, chunk_mapping)."""
    if item.feature is None:
        raise ValueError(
            "MOSS-Transcribe-Diarize audio item is missing input_features."
        )
    input_features = item.feature

    audio_feature_lengths = getattr(item, "audio_feature_lengths", None)
    if audio_feature_lengths is None:
        raise ValueError(
            "MOSS-Transcribe-Diarize audio item is missing audio_feature_lengths."
        )
    audio_feature_lengths = audio_feature_lengths.to(device="cpu", dtype=torch.long)
    if audio_feature_lengths.numel() != input_features.shape[0]:
        raise ValueError(
            "audio_feature_lengths must contain one length per input_features "
            f"chunk: got {audio_feature_lengths.numel()} lengths for "
            f"{input_features.shape[0]} chunks."
        )

    audio_chunk_mapping = getattr(item, "audio_chunk_mapping", None)
    if audio_chunk_mapping is None:
        audio_chunk_mapping = torch.zeros(
            input_features.shape[0], dtype=torch.long, device="cpu"
        )
    else:
        audio_chunk_mapping = audio_chunk_mapping.to(device="cpu", dtype=torch.long)
    if audio_chunk_mapping.numel() != input_features.shape[0]:
        raise ValueError(
            "audio_chunk_mapping must contain one sample index per input_features "
            f"chunk: got {audio_chunk_mapping.numel()} indices for "
            f"{input_features.shape[0]} chunks."
        )

    return input_features, audio_feature_lengths, audio_chunk_mapping


class VQAdaptor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, norm_eps: float = 1e-6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size, eps=norm_eps, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MossTranscribeDiarizeForConditionalGeneration(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: MossTranscribeDiarizeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.whisper_encoder = WhisperEncoder(config.audio_config, quant_config)
        self.vq_adaptor = VQAdaptor(
            input_dim=config.adaptor_input_dim,
            hidden_size=config.text_config.hidden_size,
            norm_eps=config.text_config.rms_norm_eps,
        )
        self.language_model = Qwen3ForCausalLM(
            config.text_config,
            quant_config,
            prefix=add_prefix("model.language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def time_merge(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = features.shape
        merge_size = int(self.config.audio_merge_size)
        trimmed_len = (seq_len // merge_size) * merge_size
        return features[:, :trimmed_len, :].reshape(
            batch_size, trimmed_len // merge_size, hidden_size * merge_size
        )

    def _run_whisper_encoder(
        self,
        input_features: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        encoder_len = (input_features.shape[-1] - 1) // 2 + 1
        encoder_position_ids = torch.arange(
            encoder_len,
            device=input_features.device,
            dtype=torch.long,
        )
        if input_features.shape[0] <= _ENCODER_MAX_BATCH_CHUNKS:
            return self.whisper_encoder(
                input_features, encoder_position_ids, forward_batch
            )
        outputs = [
            self.whisper_encoder(sub_batch, encoder_position_ids, forward_batch)
            for sub_batch in input_features.split(_ENCODER_MAX_BATCH_CHUNKS, dim=0)
        ]
        return torch.cat(outputs, dim=0)

    def get_audio_feature(
        self,
        items: List[MultimodalDataItem],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_size = self.config.text_config.hidden_size
        adaptor_param = next(self.vq_adaptor.parameters())
        if not items:
            return torch.empty(
                (0, hidden_size),
                device=adaptor_param.device,
                dtype=adaptor_param.dtype,
            )

        device = next(self.whisper_encoder.parameters()).device
        encoder_dtype = next(self.whisper_encoder.parameters()).dtype
        merge_size = int(self.config.audio_merge_size)

        validated = [_validate_audio_item(item) for item in items]

        # Batch chunks across items through one encoder call. Whisper
        # attends over the whole mel window, so zero-padding narrower
        # windows would change results; instead only items with identical
        # mel widths are batched together (the feature extractor pads all
        # windows to 30s, so in practice this is one group).
        groups: dict[int, list[int]] = {}
        for item_idx, (features, _, _) in enumerate(validated):
            groups.setdefault(int(features.shape[-1]), []).append(item_idx)

        encoded: dict[int, torch.Tensor] = {}
        for item_indices in groups.values():
            batch = torch.cat(
                [
                    validated[i][0].to(device=device, dtype=encoder_dtype)
                    for i in item_indices
                ],
                dim=0,
            )
            whisper_features = self._run_whisper_encoder(batch, forward_batch)
            chunk_counts = [validated[i][0].shape[0] for i in item_indices]
            for item_idx, item_features in zip(
                item_indices, whisper_features.split(chunk_counts, dim=0)
            ):
                encoded[item_idx] = item_features

        # Assemble per-audio features (items in input order, audios in
        # sample order within each item), then run the token-wise adaptor
        # once over all audios.
        merged_audios: list[torch.Tensor] = []
        for item_idx, (_, lengths, mapping) in enumerate(validated):
            whisper_features = encoded[item_idx]
            lengths_list = lengths.tolist()
            mapping_list = mapping.tolist()
            num_audios = max(mapping_list) + 1 if mapping_list else 0
            per_audio_chunks: list[list[torch.Tensor]] = [
                [] for _ in range(num_audios)
            ]
            for chunk_idx, token_len in enumerate(lengths_list):
                per_audio_chunks[mapping_list[chunk_idx]].append(
                    whisper_features[
                        chunk_idx : chunk_idx + 1, : int(token_len) * merge_size
                    ]
                )
            for parts in per_audio_chunks:
                if not parts:
                    continue
                feat = torch.cat(parts, dim=1).to(dtype=adaptor_param.dtype)
                merged_audios.append(self.time_merge(feat))

        if not merged_audios:
            return torch.empty(
                (0, hidden_size),
                device=adaptor_param.device,
                dtype=adaptor_param.dtype,
            )

        merged = torch.cat(merged_audios, dim=1)
        return self.vq_adaptor(merged).squeeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: lambda items: self.get_audio_feature(
                    items,
                    forward_batch,
                ),
            },
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        whisper_stacked_params_mapping = [
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        def load_one(name: str, loaded_weight: torch.Tensor):
            original_name = name
            if "rotary_emb.inv_freq" in name:
                return
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                return

            if name == "lm_head.weight":
                name = "language_model.lm_head.weight"
            elif name.startswith("model.language_model."):
                name = "language_model.model." + name[len("model.language_model.") :]
            elif name.startswith("model.whisper_encoder."):
                name = "whisper_encoder." + name[len("model.whisper_encoder.") :]
            elif name.startswith("model.vq_adaptor."):
                name = "vq_adaptor." + name[len("model.vq_adaptor.") :]

            if (
                name == "language_model.model.embed_tokens.weight"
                and self.config.text_config.tie_word_embeddings
                and "language_model.lm_head.weight" in params_dict
            ):
                param = params_dict["language_model.lm_head.weight"]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            handled = False
            if name.startswith("whisper_encoder."):
                for param_name, weight_name, shard_id in whisper_stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                        handled = True
                        break
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        param.weight_loader(param, loaded_weight, shard_id)
                        handled = True
                    break

            if name.startswith("language_model."):
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                        handled = True
                        break
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        param.weight_loader(param, loaded_weight, shard_id)
                        handled = True
                    break

            if handled:
                return

            if name.endswith(".bias") and name not in params_dict:
                return

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.debug(f"Skipping weight: {original_name} -> {name}")

        for name, loaded_weight in weights:
            load_one(name, loaded_weight)
            if (
                name.startswith("model.whisper_encoder.layers.")
                and ".self_attn.k_proj.weight" in name
            ):
                load_one(
                    name.replace(".k_proj.weight", ".k_proj.bias"),
                    torch.zeros(
                        loaded_weight.shape[0],
                        dtype=loaded_weight.dtype,
                        device=loaded_weight.device,
                    ),
                )


EntryClass = MossTranscribeDiarizeForConditionalGeneration
