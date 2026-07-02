# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.models.mimo_asr.configuration_mimo_asr import MiMoV2ASRConfig
from sglang_omni.models.mimo_asr.sglang_model import (
    MiMoV2ASRForCausalLM,
    group_mimo_audio_codes,
    normalize_mimo_audio_codes,
    pad_mimo_audio_codes_to_group,
    validate_mimo_speech_config,
)


def _tiny_config(**overrides) -> MiMoV2ASRConfig:
    defaults = {
        "vocab_size": 128,
        "hidden_size": 7,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "intermediate_size": 8,
        "audio_channels": 2,
        "group_size": 2,
        "input_local_dim": 3,
        "speech_vocab_size": "5-6",
        "speech_zeroemb_idx": "4-5",
        "delay_pattern": "0-1",
    }
    defaults.update(overrides)
    return MiMoV2ASRConfig(**defaults)


def test_normalize_mimo_audio_codes_accepts_t_by_c_and_c_by_t() -> None:
    t_by_c = torch.arange(6).reshape(3, 2)
    c_by_t = t_by_c.transpose(0, 1)

    assert torch.equal(normalize_mimo_audio_codes(t_by_c, audio_channels=2), t_by_c)
    assert torch.equal(normalize_mimo_audio_codes(c_by_t, audio_channels=2), t_by_c)
    assert torch.equal(
        normalize_mimo_audio_codes(t_by_c.unsqueeze(0), audio_channels=2),
        t_by_c,
    )


def test_pad_and_group_mimo_audio_codes_repeat_tail_frame() -> None:
    codes = torch.tensor([[1, 10], [2, 20], [3, 30]], dtype=torch.long)

    padded = pad_mimo_audio_codes_to_group(codes, group_size=2)
    grouped = group_mimo_audio_codes(codes, audio_channels=2, group_size=2)

    assert torch.equal(padded, torch.tensor([[1, 10], [2, 20], [3, 30], [3, 30]]))
    assert grouped.shape == (2, 2, 2)
    assert torch.equal(grouped[0], torch.tensor([[1, 2], [10, 20]]))
    assert torch.equal(grouped[1], torch.tensor([[3, 3], [30, 30]]))


def test_mimo_audio_code_helpers_reject_bad_shapes() -> None:
    try:
        normalize_mimo_audio_codes(torch.zeros(3, 3), audio_channels=2)
    except ValueError as exc:
        assert "audio_channels" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad channel count should fail")

    try:
        pad_mimo_audio_codes_to_group(torch.zeros(0, 2, dtype=torch.long), group_size=2)
    except ValueError as exc:
        assert "at least one frame" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("empty codes should fail")


def test_validate_mimo_speech_config_rejects_mismatched_channels() -> None:
    config = _tiny_config(speech_vocab_size="5-6-7")

    try:
        validate_mimo_speech_config(config)
    except ValueError as exc:
        assert "speech_vocab_size" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("mismatched speech vocab channels should fail")


def test_validate_mimo_speech_config_rejects_bad_zeroemb_idx() -> None:
    config = _tiny_config(speech_zeroemb_idx="5-5")

    try:
        validate_mimo_speech_config(config)
    except ValueError as exc:
        assert "zeroemb" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("out-of-range zeroemb index should fail")


def test_mimo_sglang_model_initializes_speech_embedding_modules() -> None:
    config = _tiny_config()

    model = MiMoV2ASRForCausalLM(config)

    assert model.audio_channels == 2
    assert model.group_size == 2
    assert model.speech_vocab_sizes == [5, 6]
    assert model.speech_zeroemb_indices == [4, 5]
    assert model.delay_pattern_values == [0, 1]
    assert len(model.speech_embeddings) == 2
    assert model.speech_embeddings[0].num_embeddings == 5
    assert model.speech_embeddings[0].embedding_dim == 3
    assert model.speech_embeddings[0].padding_idx == 4
    assert model.speech_embeddings[1].num_embeddings == 6
    assert model.speech_embeddings[1].padding_idx == 5
    assert model.speech_group_downcast.in_features == 6
    assert model.speech_group_downcast.out_features == 7
