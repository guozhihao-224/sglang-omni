# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
    MossTranscribeDiarizeForConditionalGeneration,
)


class _FakeConfig:
    audio_merge_size = 2

    class text_config:
        hidden_size = 3


class _FakeWhisperEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.empty(0, dtype=torch.float32))
        self.calls = 0
        self.batch_shapes = []

    def forward(self, input_features, encoder_position_ids, forward_batch):
        del encoder_position_ids, forward_batch
        self.calls += 1
        self.batch_shapes.append(tuple(input_features.shape))
        batch_size = input_features.shape[0]
        seq_len = (input_features.shape[-1] - 1) // 2 + 1
        values = torch.arange(
            batch_size * seq_len,
            device=input_features.device,
            dtype=input_features.dtype,
        )
        return values.reshape(batch_size, seq_len, 1).expand(batch_size, seq_len, 4)


class _FakeAdaptor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.empty(0, dtype=torch.float32))

    def forward(self, features):
        return features[..., :3]


def _item(features, lengths, mapping=None):
    item = MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=0,
        feature=torch.tensor(features, dtype=torch.float32),
    )
    item.audio_feature_lengths = torch.tensor(lengths, dtype=torch.long)
    if mapping is not None:
        item.audio_chunk_mapping = torch.tensor(mapping, dtype=torch.long)
    return item


def test_get_audio_feature_batches_encoder_across_items_and_preserves_order() -> None:
    model = MossTranscribeDiarizeForConditionalGeneration.__new__(
        MossTranscribeDiarizeForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = _FakeConfig()
    model.whisper_encoder = _FakeWhisperEncoder()
    model.vq_adaptor = _FakeAdaptor()

    first = _item(
        [
            [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]],
            [[3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4]],
        ],
        lengths=[1, 1],
        mapping=[0, 0],
    )
    second = _item(
        [
            [[5, 5, 5, 5], [6, 6, 6, 6]],
        ],
        lengths=[1],
    )

    output = model.get_audio_feature([first, second], forward_batch=None)

    assert model.whisper_encoder.calls == 1
    assert model.whisper_encoder.batch_shapes == [(3, 2, 6)]
    assert output.tolist() == [
        [0.0, 0.0, 1.0],
        [3.0, 3.0, 4.0],
        [6.0, 6.0, 7.0],
    ]


def test_get_audio_feature_returns_empty_hidden_tensor_for_no_items() -> None:
    model = MossTranscribeDiarizeForConditionalGeneration.__new__(
        MossTranscribeDiarizeForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = _FakeConfig()
    model.whisper_encoder = _FakeWhisperEncoder()
    model.vq_adaptor = _FakeAdaptor()

    output = model.get_audio_feature([], forward_batch=None)

    assert tuple(output.shape) == (0, 3)
    assert output.dtype == torch.float32
