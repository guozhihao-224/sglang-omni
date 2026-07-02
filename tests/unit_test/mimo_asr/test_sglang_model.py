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


def test_mimo_model_embeds_grouped_audio_codes_with_zeroemb_mask() -> None:
    config = _tiny_config(hidden_size=2)
    model = MiMoV2ASRForCausalLM(config)
    with torch.no_grad():
        model.speech_embeddings[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [99.0, 99.0, 99.0],
                ]
            )
        )
        model.speech_embeddings[1].weight.copy_(
            torch.tensor(
                [
                    [0.0, 10.0, 0.0],
                    [0.0, 20.0, 0.0],
                    [0.0, 30.0, 0.0],
                    [0.0, 40.0, 0.0],
                    [0.0, 50.0, 0.0],
                    [99.0, 99.0, 99.0],
                ]
            )
        )

    codes = torch.tensor([[1, 2], [4, 3], [2, 5]], dtype=torch.long)

    embeddings = model.embed_grouped_audio_codes(codes)

    assert embeddings.shape == (2, 2, 3)
    assert torch.equal(
        embeddings,
        torch.tensor(
            [
                [[2.0, 30.0, 0.0], [0.0, 40.0, 0.0]],
                [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ]
        ),
    )


def test_mimo_model_projects_grouped_audio_embeds_to_hidden_size() -> None:
    config = _tiny_config(hidden_size=2)
    model = MiMoV2ASRForCausalLM(config)
    with torch.no_grad():
        model.speech_group_downcast.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ]
            )
        )
        model.speech_group_downcast.bias.zero_()
    embeddings = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    projected = model.project_grouped_audio_embeds(embeddings)

    assert torch.equal(projected, torch.tensor([[1.0, 4.0], [7.0, 10.0]]))


def test_mimo_model_encode_audio_codes_to_hidden_combines_embedding_and_projection() -> None:
    config = _tiny_config(hidden_size=1)
    model = MiMoV2ASRForCausalLM(config)
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
        model.speech_group_downcast.bias.zero_()

    hidden = model.encode_audio_codes_to_hidden(torch.tensor([[0, 0], [1, 1]]))

    assert hidden.shape == (1, 1)
    assert torch.equal(hidden, torch.tensor([[12.0]]))


def test_mimo_model_project_grouped_audio_embeds_validates_shape() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())

    try:
        model.project_grouped_audio_embeds(torch.zeros(2, 3, 3))
    except ValueError as exc:
        assert "speech group dimension" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad group dimension should fail")
