# SPDX-License-Identifier: Apache-2.0
"""Tests for the batched MOSS-Transcribe-Diarize audio feature path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import sglang_omni.models.moss_transcribe_diarize.sglang_model as sglang_model
from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
    MossTranscribeDiarizeForConditionalGeneration,
    VQAdaptor,
)

_ENCODER_HIDDEN = 4
_MERGE_SIZE = 2
_LLM_HIDDEN = _ENCODER_HIDDEN * _MERGE_SIZE


class FakeWhisperEncoder(nn.Module):
    """Deterministic per-chunk stand-in for the Whisper encoder.

    Downsamples the mel time axis by 2 (like the real conv stem) and expands
    to a fixed hidden size, so identical chunks produce identical outputs
    regardless of how they are batched.
    """

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls: list[int] = []

    def forward(self, input_features, position_ids, forward_batch):
        del position_ids, forward_batch
        self.calls.append(int(input_features.shape[0]))
        batch_size, _, mel_len = input_features.shape
        out_len = mel_len // 2
        base = (
            input_features[:, 0, : out_len * 2]
            .reshape(batch_size, out_len, 2)
            .mean(dim=-1)
        )
        scales = torch.arange(
            1, _ENCODER_HIDDEN + 1, device=base.device, dtype=base.dtype
        )
        return base.unsqueeze(-1) * scales


def _make_model() -> MossTranscribeDiarizeForConditionalGeneration:
    model = MossTranscribeDiarizeForConditionalGeneration.__new__(
        MossTranscribeDiarizeForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        audio_merge_size=_MERGE_SIZE,
        text_config=SimpleNamespace(hidden_size=_LLM_HIDDEN),
    )
    model.whisper_encoder = FakeWhisperEncoder()
    torch.manual_seed(0)
    model.vq_adaptor = VQAdaptor(input_dim=_LLM_HIDDEN, hidden_size=_LLM_HIDDEN)
    return model


def _make_item(
    *,
    num_chunks: int,
    mel_len: int,
    token_lengths: list[int],
    chunk_mapping: list[int] | None = None,
    seed: int = 0,
) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        feature=torch.randn((num_chunks, 3, mel_len), generator=generator),
        audio_feature_lengths=torch.tensor(token_lengths, dtype=torch.long),
        audio_chunk_mapping=(
            torch.tensor(chunk_mapping, dtype=torch.long)
            if chunk_mapping is not None
            else None
        ),
    )


def _reference_get_audio_feature(model, items) -> torch.Tensor:
    """Pre-batching implementation: encode and adapt one item at a time."""
    merge_size = int(model.config.audio_merge_size)
    audio_embeds = []
    for item in items:
        features = item.feature
        whisper_features = model.whisper_encoder(features, None, None)
        mapping = item.audio_chunk_mapping
        if mapping is None:
            mapping = torch.zeros(features.shape[0], dtype=torch.long)
        lengths_list = item.audio_feature_lengths.tolist()
        mapping_list = mapping.tolist()
        num_audios = max(mapping_list) + 1 if mapping_list else 0
        per_audio_chunks = [[] for _ in range(num_audios)]
        for chunk_idx, token_len in enumerate(lengths_list):
            per_audio_chunks[mapping_list[chunk_idx]].append(
                whisper_features[
                    chunk_idx : chunk_idx + 1, : int(token_len) * merge_size
                ]
            )
        for parts in per_audio_chunks:
            if not parts:
                continue
            feat = torch.cat(parts, dim=1)
            merged = model.time_merge(feat)
            audio_embeds.append(model.vq_adaptor(merged).squeeze(0))
    return torch.cat(audio_embeds, dim=0)


def test_batched_path_matches_per_item_reference() -> None:
    model = _make_model()
    items = [
        _make_item(num_chunks=3, mel_len=20, token_lengths=[5, 5, 2], seed=1),
        _make_item(
            num_chunks=4,
            mel_len=20,
            token_lengths=[5, 3, 5, 1],
            chunk_mapping=[0, 0, 1, 1],
            seed=2,
        ),
        _make_item(num_chunks=1, mel_len=20, token_lengths=[4], seed=3),
    ]

    expected = _reference_get_audio_feature(model, items)
    model.whisper_encoder.calls.clear()
    actual = model.get_audio_feature(items, forward_batch=None)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)
    # All 8 chunks from the 3 items went through one encoder call.
    assert model.whisper_encoder.calls == [8]


def test_items_with_different_mel_widths_are_grouped_not_padded() -> None:
    model = _make_model()
    items = [
        _make_item(num_chunks=2, mel_len=20, token_lengths=[5, 3], seed=4),
        _make_item(num_chunks=1, mel_len=12, token_lengths=[3], seed=5),
        _make_item(num_chunks=1, mel_len=20, token_lengths=[5], seed=6),
    ]

    expected = _reference_get_audio_feature(model, items)
    model.whisper_encoder.calls.clear()
    actual = model.get_audio_feature(items, forward_batch=None)

    torch.testing.assert_close(actual, expected)
    # One call per distinct mel width: [20-wide chunks, 12-wide chunk].
    assert sorted(model.whisper_encoder.calls) == [1, 3]


def test_encoder_batch_is_capped_by_max_chunks(monkeypatch) -> None:
    model = _make_model()
    monkeypatch.setattr(sglang_model, "_ENCODER_MAX_BATCH_CHUNKS", 2)
    items = [
        _make_item(num_chunks=5, mel_len=20, token_lengths=[5, 5, 5, 5, 2], seed=7),
    ]

    expected = _reference_get_audio_feature(model, items)
    model.whisper_encoder.calls.clear()
    actual = model.get_audio_feature(items, forward_batch=None)

    torch.testing.assert_close(actual, expected)
    assert model.whisper_encoder.calls == [2, 2, 1]


def test_empty_items_return_empty_embedding() -> None:
    model = _make_model()

    result = model.get_audio_feature([], forward_batch=None)

    assert result.shape == (0, _LLM_HIDDEN)
    assert result.dtype == next(model.vq_adaptor.parameters()).dtype


def test_missing_feature_raises() -> None:
    model = _make_model()
    item = SimpleNamespace(
        feature=None,
        audio_feature_lengths=torch.tensor([1]),
        audio_chunk_mapping=None,
    )

    with pytest.raises(ValueError, match="missing input_features"):
        model.get_audio_feature([item], forward_batch=None)


def test_mismatched_lengths_raise() -> None:
    model = _make_model()
    item = _make_item(num_chunks=2, mel_len=20, token_lengths=[5], seed=8)

    with pytest.raises(ValueError, match="one length per input_features"):
        model.get_audio_feature([item], forward_batch=None)
