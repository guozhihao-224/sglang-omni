# SPDX-License-Identifier: Apache-2.0
"""MiMo audio tokenizer adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .tool_funcs.audio_lengths import MIMO_ASR_AUDIO_CHANNELS, MIMO_ASR_GROUP_SIZE

MIMO_ASR_SAMPLE_RATE = 24000


class MiMoAudioTokenizerAdapter:
    """Adapter around XiaomiMiMo/MiMo-Audio-Tokenizer.

    The adapter keeps the public request-builder contract stable: ``encode``
    always returns ``[T, 8]`` int64 codes padded to a multiple of group size.
    Tests can inject a lightweight backend; production uses lazy imports.
    """

    def __init__(
        self,
        audio_tokenizer_path: str,
        *,
        device: str = "cuda:0",
        backend: Any | None = None,
        group_size: int = MIMO_ASR_GROUP_SIZE,
    ) -> None:
        self.audio_tokenizer_path = audio_tokenizer_path
        self.device = device
        self.group_size = group_size
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        self._backend = backend

    def encode(self, audio: Any, *, sample_rate: int) -> torch.Tensor:
        """Encode audio into padded ``[T, 8]`` MiMo RVQ code frames."""

        waveform = self._normalize_audio(audio, sample_rate=sample_rate)
        backend = self._backend or self._load_backend()
        raw_codes = backend.encode(audio=(waveform, MIMO_ASR_SAMPLE_RATE))
        codes = self._normalize_codes(raw_codes)
        return self._pad_codes(codes)

    def _load_backend(self) -> Any:
        """Load the real tokenizer backend lazily.

        Supported import paths intentionally cover the public vLLM-Omni layout
        and the likely standalone MiMo package layout.  If neither is installed,
        fail with an actionable error instead of at module import time.
        """

        import importlib

        candidates = (
            "mimo_audio.mimo_audio_code2wav",
            "vllm_omni.model_executor.models.mimo_audio.mimo_audio_code2wav",
        )
        last_error: BaseException | None = None
        for module_name in candidates:
            try:
                module = importlib.import_module(module_name)
                get_tokenizer_worker = getattr(module, "get_tokenizer_worker")
                self._backend = get_tokenizer_worker(
                    device=self.device,
                    config_path=self.audio_tokenizer_path,
                    audio_tokenizer_path=self.audio_tokenizer_path,
                )
                return self._backend
            except Exception as exc:  # pragma: no cover - environment dependent
                last_error = exc
        raise RuntimeError(
            "Unable to load MiMo audio tokenizer backend. Install the official "
            "MiMo audio tokenizer package or vLLM-Omni, and set "
            "audio_tokenizer_path to XiaomiMiMo/MiMo-Audio-Tokenizer or a local path."
        ) from last_error

    @staticmethod
    def _normalize_audio(audio: Any, *, sample_rate: int) -> torch.Tensor:
        waveform = _as_float_tensor(audio)
        if waveform.ndim == 2:
            if waveform.shape[0] == 1:
                waveform = waveform[0]
            elif waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            elif waveform.shape[0] <= waveform.shape[1]:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.mean(dim=1)
        elif waveform.ndim != 1:
            raise ValueError(
                f"MiMo audio must be 1-D or 2-D, got shape {tuple(waveform.shape)}"
            )

        if sample_rate != MIMO_ASR_SAMPLE_RATE:
            try:
                import torchaudio
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "torchaudio is required to resample MiMo-ASR audio"
                ) from exc
            waveform = torchaudio.functional.resample(
                waveform,
                int(sample_rate),
                MIMO_ASR_SAMPLE_RATE,
            )
        return waveform.contiguous().to(torch.float32)

    @staticmethod
    def _normalize_codes(raw_codes: Any) -> torch.Tensor:
        codes = torch.as_tensor(raw_codes, dtype=torch.long)
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes.squeeze(0)
        if codes.ndim != 2:
            raise ValueError(
                f"MiMo audio codes must be 2-D, got shape {tuple(codes.shape)}"
            )
        if codes.shape[1] == MIMO_ASR_AUDIO_CHANNELS:
            return codes.contiguous()
        if codes.shape[0] == MIMO_ASR_AUDIO_CHANNELS:
            return codes.transpose(0, 1).contiguous()
        raise ValueError(
            "MiMo audio codes must have 8 channels in shape [T, 8] or [8, T], "
            f"got {tuple(codes.shape)}"
        )

    def _pad_codes(self, codes: torch.Tensor) -> torch.Tensor:
        num_frames = int(codes.shape[0])
        remainder = num_frames % self.group_size
        if remainder == 0:
            return codes
        if num_frames == 0:
            raise ValueError("MiMo audio tokenizer returned zero code frames")
        pad_frames = self.group_size - remainder
        tail = codes[-1:].expand(pad_frames, -1)
        return torch.cat([codes, tail], dim=0).contiguous()


def _as_float_tensor(audio: Any) -> torch.Tensor:
    if isinstance(audio, torch.Tensor):
        return audio.detach().to(torch.float32).cpu()
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if isinstance(audio, (list, tuple)):
        return torch.tensor(audio, dtype=torch.float32)
    raise ValueError(f"Unsupported MiMo-ASR audio value: {type(audio).__name__}")


__all__ = ["MIMO_ASR_SAMPLE_RATE", "MiMoAudioTokenizerAdapter"]
