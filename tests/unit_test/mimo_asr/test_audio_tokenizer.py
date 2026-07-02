# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.mimo_asr.audio_tokenizer import (
    MIMO_ASR_SAMPLE_RATE,
    MiMoAudioTokenizerAdapter,
    _install_flash_attn_varlen_compat,
)


class _FakeBackend:
    def __init__(self, codes) -> None:
        self.codes = codes
        self.calls: list[dict] = []

    def encode(self, *, audio):
        waveform, sample_rate = audio
        self.calls.append(
            {
                "shape": tuple(waveform.shape),
                "sample_rate": sample_rate,
                "dtype": waveform.dtype,
            }
        )
        return self.codes


def test_mimo_audio_tokenizer_returns_padded_t_by_8_codes() -> None:
    backend = _FakeBackend(torch.arange(5 * 8).reshape(5, 8))
    adapter = MiMoAudioTokenizerAdapter("tok", backend=backend)

    codes = adapter.encode(np.zeros(2400, dtype=np.float32), sample_rate=24000)

    assert backend.calls == [
        {"shape": (2400,), "sample_rate": MIMO_ASR_SAMPLE_RATE, "dtype": torch.float32}
    ]
    assert codes.shape == (8, 8)
    assert codes.dtype == torch.long
    assert torch.equal(codes[:5], torch.arange(5 * 8).reshape(5, 8))
    assert torch.equal(codes[5:], codes[4:5].expand(3, -1))


def test_mimo_audio_tokenizer_transposes_channel_first_codes() -> None:
    backend = _FakeBackend(torch.arange(8 * 5).reshape(8, 5))
    adapter = MiMoAudioTokenizerAdapter("tok", backend=backend)

    codes = adapter.encode(torch.zeros(100), sample_rate=24000)

    assert codes.shape == (8, 8)
    assert torch.equal(codes[:5], torch.arange(8 * 5).reshape(8, 5).transpose(0, 1))


def test_mimo_audio_tokenizer_accepts_batched_single_codes() -> None:
    backend = _FakeBackend(torch.arange(4 * 8).reshape(1, 4, 8))
    adapter = MiMoAudioTokenizerAdapter("tok", backend=backend)

    codes = adapter.encode(torch.zeros(100), sample_rate=24000)

    assert codes.shape == (4, 8)
    assert torch.equal(codes, torch.arange(4 * 8).reshape(4, 8))


def test_mimo_audio_tokenizer_mixes_stereo_to_mono() -> None:
    backend = _FakeBackend(torch.arange(4 * 8).reshape(4, 8))
    adapter = MiMoAudioTokenizerAdapter("tok", backend=backend)
    stereo = np.stack(
        [np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float32)],
        axis=0,
    )

    adapter.encode(stereo, sample_rate=24000)

    assert backend.calls[0]["shape"] == (10,)


def test_mimo_audio_tokenizer_rejects_bad_code_shape() -> None:
    backend = _FakeBackend(torch.zeros(5, 7, dtype=torch.long))
    adapter = MiMoAudioTokenizerAdapter("tok", backend=backend)

    try:
        adapter.encode(torch.zeros(100), sample_rate=24000)
    except ValueError as exc:
        assert "8 channels" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad code shape should fail")


def test_mimo_audio_tokenizer_rejects_bad_audio_shape() -> None:
    adapter = MiMoAudioTokenizerAdapter("tok", backend=_FakeBackend(torch.zeros(4, 8)))

    try:
        adapter.encode(torch.zeros(1, 2, 3), sample_rate=24000)
    except ValueError as exc:
        assert "1-D or 2-D" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad audio shape should fail")


def test_mimo_audio_tokenizer_rejects_bad_group_size() -> None:
    try:
        MiMoAudioTokenizerAdapter("tok", backend=_FakeBackend(torch.zeros(4, 8)), group_size=0)
    except ValueError as exc:
        assert "group_size" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad group size should fail")


def test_mimo_audio_tokenizer_loads_official_backend(monkeypatch) -> None:
    class _FakeTokenizer:
        config = SimpleNamespace(
            sampling_rate=24000,
            nfft=400,
            hop_length=160,
            window_size=400,
            fmin=0,
            fmax=8000,
            n_mels=128,
        )

        @classmethod
        def from_pretrained(cls, path):
            assert path == "tok"
            return cls()

        def eval(self):
            return self

        def bfloat16(self):
            return self

        def to(self, device):
            return self

    def _fake_import_module(name):
        if name == "mimo_audio_tokenizer":
            return SimpleNamespace(MiMoAudioTokenizer=_FakeTokenizer)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)

    backend = MiMoAudioTokenizerAdapter("tok", device="cpu")._load_backend()

    assert backend.tokenizer.__class__ is _FakeTokenizer


def test_mimo_audio_tokenizer_does_not_import_vllm_worker(monkeypatch) -> None:
    imported_modules: list[str] = []

    class _FakeTokenizer:
        config = SimpleNamespace(
            sampling_rate=24000,
            nfft=400,
            hop_length=160,
            window_size=400,
            fmin=0,
            fmax=8000,
            n_mels=128,
        )

        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def eval(self):
            return self

        def bfloat16(self):
            return self

        def to(self, device):
            return self

    def _fake_import_module(name):
        imported_modules.append(name)
        if name == "mimo_audio_tokenizer":
            return SimpleNamespace(MiMoAudioTokenizer=_FakeTokenizer)
        if name.startswith("vllm_omni") or name.startswith("mimo_audio"):
            raise AssertionError(f"unexpected worker import: {name}")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)

    MiMoAudioTokenizerAdapter("tok", device="cpu")._load_backend()

    assert "mimo_audio_tokenizer" in imported_modules


def test_official_backend_matches_vllm_encode_shape_and_channel_slice() -> None:
    from sglang_omni.models.mimo_asr.audio_tokenizer import (
        _OfficialMiMoAudioTokenizerBackend,
    )

    class _FakeEncoder:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def encode(self, *, input_features, input_lens, return_codes_only):
            assert return_codes_only is True
            self.calls.append((tuple(input_features.shape), tuple(input_lens.tolist())))
            total_len = int(input_lens.sum().item())
            codes = torch.arange(12 * total_len, dtype=torch.long).reshape(12, total_len)
            return codes, None

    class _FakeTokenizer:
        config = SimpleNamespace(
            sampling_rate=24000,
            nfft=400,
            hop_length=160,
            window_size=400,
            fmin=0,
            fmax=8000,
            n_mels=80,
        )

        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def __init__(self) -> None:
            self.encoder = _FakeEncoder()

        def eval(self):
            return self

        def to(self, device):
            return self

        def bfloat16(self):
            raise AssertionError("CPU backend should not cast to bfloat16")

    backend = _OfficialMiMoAudioTokenizerBackend(
        tokenizer_cls=_FakeTokenizer,
        audio_tokenizer_path="tok",
        device="cpu",
        audio_channels=8,
    )
    backend._wav_to_mel = lambda waveform: torch.zeros(80, 6501)

    codes = backend.encode(audio=(torch.zeros(24000), 24000), max_length=6000)

    assert codes.shape == (8, 6501)
    assert backend.tokenizer.encoder.calls == [
        ((6000, 80), (6000,)),
        ((501, 80), (501,)),
    ]
    expected = torch.arange(12 * 6501, dtype=torch.long).reshape(12, 6501)[:8]
    assert torch.equal(codes, expected)


def test_mimo_audio_tokenizer_installs_flash_attn_varlen_compat(monkeypatch) -> None:
    flash_attn = SimpleNamespace()
    sentinel = object()

    def _fake_import_module(name):
        if name == "flash_attn":
            return flash_attn
        if name == "flash_attn.flash_attn_interface":
            return SimpleNamespace(flash_attn_varlen_func=sentinel)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)

    _install_flash_attn_varlen_compat()

    assert flash_attn.flash_attn_varlen_func is sentinel
