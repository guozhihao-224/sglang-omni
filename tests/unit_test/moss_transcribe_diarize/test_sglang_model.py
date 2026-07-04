# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang_omni.models.moss_transcribe_diarize.model_runner import (
    MossTranscribeDiarizeModelRunner,
)
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
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
        [6.0, 6.0, 6.0],
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


def test_prefill_runner_batches_audio_items_across_requests() -> None:
    class FakeEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros((20, 3), dtype=torch.float32))

        def forward(self, input_ids):
            return input_ids.to(torch.float32).unsqueeze(-1).expand(-1, 3).clone()

    class FakeLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.probe = nn.Parameter(torch.empty(0, dtype=torch.float32))

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = FakeLanguageModel()
            self.embedding = FakeEmbedding()
            self.encoded_item_count = None
            self.forward_input_embeds = None

        def get_input_embeddings(self):
            return self.embedding

        def _encode_audio_items_batched(self, items, forward_batch):
            del forward_batch
            self.encoded_item_count = len(items)
            return [
                torch.tensor([[100.0, 101.0, 102.0]], dtype=torch.float32),
                torch.tensor([[200.0, 201.0, 202.0]], dtype=torch.float32),
            ]

        def forward(self, **kwargs):
            self.forward_input_embeds = kwargs["input_embeds"].detach().cpu()
            return SimpleNamespace()

    runner = MossTranscribeDiarizeModelRunner.__new__(MossTranscribeDiarizeModelRunner)
    runner.model = FakeModel()
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(init_forward_metadata=lambda fb: None)
        )
    )

    first_item = _item([[[1, 1], [2, 2]]], lengths=[1])
    first_item.pad_value = -101
    first_item.offsets = [(1, 1)]
    second_item = _item([[[3, 3], [4, 4]]], lengths=[1])
    second_item.pad_value = -202
    second_item.offsets = [(1, 1)]

    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    prefix_indices=[],
                    extend_input_len=3,
                    multimodal_inputs=SimpleNamespace(mm_items=[first_item]),
                )
            )
        ),
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    prefix_indices=[],
                    extend_input_len=2,
                    multimodal_inputs=SimpleNamespace(mm_items=[second_item]),
                )
            )
        ),
    ]
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([10, -101, 11, 12, -202], dtype=torch.long),
        positions=torch.arange(5, dtype=torch.long),
        mrope_positions=None,
    )

    runner.custom_prefill_forward(forward_batch, object(), requests)

    assert runner.model.encoded_item_count == 2
    assert runner.model.forward_input_embeds.tolist() == [
        [10.0, 10.0, 10.0],
        [100.0, 101.0, 102.0],
        [11.0, 11.0, 11.0],
        [12.0, 12.0, 12.0],
        [200.0, 201.0, 202.0],
    ]


def test_prefill_runner_audio_indices_respect_prefix_slice() -> None:
    item = _item([[[1, 1], [2, 2]]], lengths=[3])
    item.pad_value = -101
    item.offsets = [(1, 3)]
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([-101, -101], dtype=torch.long),
    )
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    prefix_indices=[0, 1],
                    extend_input_len=2,
                    multimodal_inputs=SimpleNamespace(mm_items=[item]),
                )
            )
        )
    ]

    row_indices, embed_indices = MossTranscribeDiarizeModelRunner._audio_indices(
        forward_batch,
        requests,
    )

    assert row_indices.tolist() == [0, 1]
    assert embed_indices.tolist() == [1, 2]
