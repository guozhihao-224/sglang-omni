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

        worker_candidates = (
            "mimo_audio.mimo_audio_code2wav",
            "vllm_omni.model_executor.models.mimo_audio.mimo_audio_code2wav",
        )
        last_error: BaseException | None = None
        for module_name in worker_candidates:
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
        try:
            module = importlib.import_module("mimo_audio_tokenizer")
            tokenizer_cls = getattr(module, "MiMoAudioTokenizer")
            self._backend = _OfficialMiMoAudioTokenizerBackend(
                tokenizer_cls=tokenizer_cls,
                audio_tokenizer_path=self.audio_tokenizer_path,
                device=self.device,
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


class _OfficialMiMoAudioTokenizerBackend:
    """Backend wrapper for XiaomiMiMo's official MiMoAudioTokenizer."""

    def __init__(
        self,
        *,
        tokenizer_cls: Any,
        audio_tokenizer_path: str,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.tokenizer = tokenizer_cls.from_pretrained(audio_tokenizer_path)
        self.tokenizer.eval().bfloat16().to(self.device)
        self.config = self.tokenizer.config
        self._mel_transform = None

    def encode(self, *, audio: tuple[torch.Tensor, int]) -> torch.Tensor:
        waveform, sample_rate = audio
        waveform = waveform.to(device=self.device, dtype=torch.float32)
        if int(sample_rate) != int(self.config.sampling_rate):
            try:
                import torchaudio
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "torchaudio is required to resample MiMo-ASR audio"
                ) from exc
            waveform = torchaudio.functional.resample(
                waveform,
                int(sample_rate),
                int(self.config.sampling_rate),
            )
        return self._encode_waveform(waveform).transpose(0, 1).detach().cpu()

    def _encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        target_sr = int(self.config.sampling_rate)
        chunk_samples = 30 * target_sr
        n_fft = int(self.config.nfft)
        total_samples = int(waveform.shape[-1])
        code_parts: list[torch.Tensor] = []
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            if 0 < total_samples - end < n_fft:
                end = total_samples
            chunk = waveform[start:end]
            if int(chunk.shape[-1]) < n_fft:
                chunk = torch.nn.functional.pad(chunk, (0, n_fft - int(chunk.shape[-1])))
            mel = self._wav_to_mel(chunk).transpose(0, 1)
            code_parts.append(self._encode_features(mel, torch.tensor([mel.size(0)])))
            start = end
        if not code_parts:
            raise ValueError("MiMo audio tokenizer received empty waveform")
        return torch.cat(code_parts, dim=-1)

    def _wav_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._mel_transform is None:
            from torchaudio.transforms import MelSpectrogram

            self._mel_transform = MelSpectrogram(
                sample_rate=int(self.config.sampling_rate),
                n_fft=int(self.config.nfft),
                hop_length=int(self.config.hop_length),
                win_length=int(self.config.window_size),
                f_min=float(self.config.fmin),
                f_max=float(self.config.fmax),
                n_mels=int(self.config.n_mels),
                power=1.0,
                center=True,
            ).to(self.device)
        spec = self._mel_transform(waveform[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze()

    def _encode_features(
        self,
        input_features: torch.Tensor,
        input_lens: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            codes, _ = self.tokenizer.encoder.encode(
                input_features=input_features.to(self.device),
                input_lens=input_lens.to(self.device),
                return_codes_only=True,
            )
        return codes


def _as_float_tensor(audio: Any) -> torch.Tensor:
    if isinstance(audio, torch.Tensor):
        return audio.detach().to(torch.float32).cpu()
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if isinstance(audio, (list, tuple)):
        return torch.tensor(audio, dtype=torch.float32)
    raise ValueError(f"Unsupported MiMo-ASR audio value: {type(audio).__name__}")


__all__ = ["MIMO_ASR_SAMPLE_RATE", "MiMoAudioTokenizerAdapter"]
