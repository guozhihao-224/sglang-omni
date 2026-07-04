# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.moss_transcribe_diarize.encoder_cuda_graph import (
    MossTranscribeDiarizeEncoderCudaGraphRunner,
)


class FakeWhisperEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(num_mel_bins=80)

    def forward(self, input_features, encoder_position_ids, forward_batch):
        del forward_batch
        pos = encoder_position_ids.to(input_features.device, input_features.dtype).sum()
        return input_features.transpose(1, 2) * self.weight + pos


def test_encoder_cuda_graph_runner_skips_cpu_inputs() -> None:
    runner = MossTranscribeDiarizeEncoderCudaGraphRunner(FakeWhisperEncoder())
    input_features = torch.zeros((1, 80, 8), dtype=torch.float32)
    encoder_position_ids = torch.arange(4, dtype=torch.long)

    assert (
        runner.encode(input_features, encoder_position_ids, forward_batch=None) is None
    )
    assert runner.captured_shapes() == []


def test_encoder_cuda_graph_runner_rejects_oversized_batch_bucket() -> None:
    runner = MossTranscribeDiarizeEncoderCudaGraphRunner(
        FakeWhisperEncoder(), batch_sizes=(1, 2)
    )
    input_features = torch.zeros((3, 80, 8), dtype=torch.float32)
    encoder_position_ids = torch.arange(4, dtype=torch.long)

    assert (
        runner.encode(input_features, encoder_position_ids, forward_batch=None) is None
    )
    assert runner.captured_shapes() == []
