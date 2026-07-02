# SPDX-License-Identifier: Apache-2.0
"""MiMo audio tokenizer adapter placeholder."""

from __future__ import annotations

from typing import Any


class MiMoAudioTokenizerAdapter:
    """Adapter for XiaomiMiMo/MiMo-Audio-Tokenizer.

    The real tokenizer port is a later phase.  The class exists now so the
    stage factory can wire dependencies and tests can monkeypatch it.
    """

    def __init__(self, audio_tokenizer_path: str, *, device: str = "cuda:0") -> None:
        self.audio_tokenizer_path = audio_tokenizer_path
        self.device = device

    def encode(self, audio: Any, *, sample_rate: int):
        raise NotImplementedError("MiMo audio tokenizer encode is not implemented yet")


__all__ = ["MiMoAudioTokenizerAdapter"]
