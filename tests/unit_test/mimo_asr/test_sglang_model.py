# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from sglang_omni.models.mimo_asr.configuration_mimo_asr import MiMoV2ASRConfig
from sglang_omni.models.mimo_asr.model_runner import (
    MiMoASRModelRunner,
    MiMoASROutputProcessor,
    build_mimo_decode_groups,
    commit_mimo_decode_groups_after_sglang,
    commit_mimo_decode_groups_to_reqs,
)
from sglang_omni.models.mimo_asr.sglang_model import (
    MiMoInputLocalTransformer,
    MiMoLocalTransformer,
    MiMoV2ASRForCausalLM,
    group_mimo_audio_codes,
    normalize_mimo_audio_codes,
    pad_mimo_audio_codes_to_group,
    route_mimo_weight_name,
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
        "input_local_layers": 0,
        "input_local_dim": 3,
        "local_dim": 3,
        "local_layers": 0,
        "local_attn_heads": 1,
        "local_ffn_dim": 4,
        "local_attn_dropout": 0.0,
        "speech_vocab_size": "5-6",
        "speech_zeroemb_idx": "4-5",
        "delay_pattern": "0-1",
    }
    defaults.update(overrides)
    return MiMoV2ASRConfig(**defaults)


def test_mimo_model_accepts_plain_qwen2_config_with_mimo_fields() -> None:
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=7,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=8,
        audio_channels=2,
        group_size=2,
        input_local_layers=0,
        input_local_dim=3,
        local_dim=3,
        local_layers=0,
        local_attn_heads=1,
        local_ffn_dim=4,
        speech_vocab_size="5-6",
        speech_zeroemb_idx="4-5",
        delay_pattern="0-1",
        empty_token_id=99,
        stop_token_id=42,
    )

    model = MiMoV2ASRForCausalLM(config)

    assert isinstance(model.config, MiMoV2ASRConfig)
    assert model.speech_vocab_sizes == [5, 6]
    assert model.speech_zeroemb_indices == [4, 5]
    assert model.config.empty_token_id == 99


class _FakeMMItem:
    def __init__(self, codes, *, pad_value: int | None = None, offsets=None) -> None:
        self.feature = codes
        self.model_specific_data = {}
        self.pad_value = pad_value
        self.offsets = offsets

    def set_pad_value(self) -> None:
        self.pad_value = -1000 - int(torch.as_tensor(self.feature).numel())


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


def test_mimo_config_builds_local_qwen2_configs() -> None:
    config = _tiny_config(
        input_local_layers=2,
        input_local_dim=4,
        local_dim=6,
        local_layers=3,
        local_attn_heads=2,
        local_ffn_dim=12,
        local_attn_dropout=0.25,
    )

    input_config = config.input_local_config()
    local_config = config.local_config()

    assert input_config.hidden_size == 4
    assert input_config.num_hidden_layers == 2
    assert input_config.num_attention_heads == 2
    assert input_config.intermediate_size == 16
    assert input_config.vocab_size == 1
    assert local_config.hidden_size == 6
    assert local_config.num_hidden_layers == 3
    assert local_config.num_attention_heads == 2
    assert local_config.intermediate_size == 12
    assert local_config.attention_dropout == 0.25
    assert local_config.vocab_size == 1


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
    assert isinstance(model.input_local_transformer, MiMoInputLocalTransformer)
    assert model.input_local_transformer.module is None
    assert model.hidden_states_downcast.in_features == 7
    assert model.hidden_states_downcast.out_features == 3
    assert isinstance(model.local_transformer, MiMoLocalTransformer)
    assert model.local_transformer.module is None
    assert len(model.local_transformer_lm_heads) == 2
    assert model.local_transformer_lm_heads[0].in_features == 3
    assert model.local_transformer_lm_heads[0].out_features == 5
    assert model.local_transformer_lm_heads[1].out_features == 6


def test_mimo_sglang_model_builds_qwen2_local_transformers() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_layers=1, local_layers=1))

    assert model.input_local_transformer.module.embed_tokens is None
    assert len(model.input_local_transformer.module.layers) == 1
    assert model.local_transformer.module.embed_tokens is None
    assert len(model.local_transformer.module.layers) == 1


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
    embeddings = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    projected = model.project_grouped_audio_embeds(embeddings)

    assert torch.equal(projected, torch.tensor([[1.0, 4.0], [7.0, 10.0]]))


def test_mimo_input_local_transformer_defaults_to_identity() -> None:
    transformer = MiMoInputLocalTransformer()
    values = torch.randn(2, 3, 4)

    assert transformer(values) is values


def test_mimo_model_project_uses_replaceable_input_local_transformer() -> None:
    class _Scale(torch.nn.Module):
        def forward(self, values):
            return values * 2

    config = _tiny_config(hidden_size=1)
    model = MiMoV2ASRForCausalLM(config)
    model.input_local_transformer = MiMoInputLocalTransformer(_Scale())
    with torch.no_grad():
        model.speech_group_downcast.weight.fill_(1.0)
    embeddings = torch.ones(1, 2, 3)

    projected = model.project_grouped_audio_embeds(embeddings)

    assert torch.equal(projected, torch.tensor([[12.0]]))


def test_mimo_model_encode_audio_codes_to_hidden_combines_embedding_and_projection() -> None:
    config = _tiny_config(hidden_size=1)
    model = MiMoV2ASRForCausalLM(config)
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)

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


def test_mimo_model_projects_hidden_states_to_local_dim() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=2, input_local_dim=3))
    with torch.no_grad():
        model.hidden_states_downcast.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            )
        )

    local_hidden = model.project_hidden_states_to_local(torch.tensor([[2.0, 3.0]]))

    assert torch.equal(local_hidden, torch.tensor([[2.0, 3.0, 5.0]]))


def test_mimo_model_project_hidden_states_validates_hidden_size() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=2))

    try:
        model.project_hidden_states_to_local(torch.zeros(1, 3))
    except ValueError as exc:
        assert "hidden_size" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad hidden size should fail")


def test_mimo_model_computes_local_code_logits_per_channel() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2, local_dim=2))
    with torch.no_grad():
        model.local_transformer_lm_heads[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ]
            )
        )
        model.local_transformer_lm_heads[1].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [2.0, 2.0],
                ]
            )
        )

    logits = model.compute_local_code_logits(torch.tensor([[2.0, 3.0]]))

    assert len(logits) == 2
    assert torch.equal(logits[0], torch.tensor([[2.0, 3.0, 5.0, 4.0, 6.0]]))
    assert torch.equal(
        logits[1],
        torch.tensor([[2.0, 3.0, 5.0, 4.0, 6.0, 10.0]]),
    )


def test_mimo_model_compute_local_code_logits_validates_local_dim() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2, local_dim=2))

    try:
        model.compute_local_code_logits(torch.zeros(1, 3))
    except ValueError as exc:
        assert "local_dim" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad local hidden size should fail")


def test_mimo_local_transformer_defaults_to_identity() -> None:
    transformer = MiMoLocalTransformer()
    values = torch.randn(2, 3)

    assert transformer(values) is values


def test_mimo_model_compute_local_code_logits_uses_replaceable_local_transformer() -> None:
    class _Shift(torch.nn.Module):
        def forward(self, values):
            return values + 1

    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2, local_dim=2))
    model.local_transformer = MiMoLocalTransformer(_Shift())
    with torch.no_grad():
        model.local_transformer_lm_heads[0].weight.fill_(1.0)
        model.local_transformer_lm_heads[1].weight.fill_(1.0)

    logits = model.compute_local_code_logits(torch.tensor([[2.0, 3.0]]))

    assert torch.equal(logits[0], torch.full((1, 5), 7.0))
    assert torch.equal(logits[1], torch.full((1, 6), 7.0))


def test_mimo_model_sample_local_code_ids_greedy() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2, local_dim=2))
    logits = [
        torch.tensor([[1.0, 4.0, 3.0, 2.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 2.0, 5.0, 3.0, 4.0]]),
    ]

    codes = model.sample_local_code_ids(logits, do_sample=False)

    assert torch.equal(codes, torch.tensor([[1, 3]]))


def test_mimo_model_sample_local_code_ids_top_p_keeps_highest_token() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2, local_dim=2))
    logits = [
        torch.tensor([[0.0, 10.0, 1.0, 2.0, 3.0]]),
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 10.0]]),
    ]
    generator = torch.Generator().manual_seed(0)

    codes = model.sample_local_code_ids(
        logits,
        do_sample=True,
        temperature=1.0,
        top_p=0.01,
        generator=generator,
    )

    assert torch.equal(codes, torch.tensor([[1, 5]]))


def test_mimo_model_local_forward_generates_greedy_codes() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=2, input_local_dim=2, local_dim=2))
    with torch.no_grad():
        model.hidden_states_downcast.weight.copy_(torch.eye(2))
        model.local_transformer_lm_heads[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                ]
            )
        )
        model.local_transformer_lm_heads[1].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [2.0, 2.0],
                ]
            )
        )

    codes = model.local_forward(torch.tensor([[2.0, 3.0]]), do_sample=False)

    assert torch.equal(codes, torch.tensor([[4, 5]]))


def test_mimo_model_local_forward_preserves_prefix_shape() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=2, input_local_dim=2, local_dim=2))
    with torch.no_grad():
        model.hidden_states_downcast.weight.copy_(torch.eye(2))
        for head in model.local_transformer_lm_heads:
            head.weight.fill_(1.0)

    codes = model.local_forward(torch.ones(2, 3, 2), do_sample=False)

    assert codes.shape == (2, 3, 2)


def test_mimo_model_sample_local_code_ids_validates_inputs() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2, local_dim=2))

    try:
        model.sample_local_code_ids([torch.zeros(1, 5)], do_sample=False)
    except ValueError as exc:
        assert "audio_channels" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad channel count should fail")

    logits = [torch.zeros(1, 5), torch.zeros(1, 6)]
    try:
        model.sample_local_code_ids(logits, temperature=0.0)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad temperature should fail")

    try:
        model.sample_local_code_ids(logits, top_p=1.5)
    except ValueError as exc:
        assert "top_p" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad top_p should fail")


def test_mimo_model_build_decode_token_group_with_speech_codes() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(empty_token_id=99))
    speech_codes = torch.tensor([[10, 20], [11, 21]])

    group = model.build_decode_token_group(7, speech_codes)

    assert torch.equal(group, torch.tensor([7, 10, 20, 99, 11, 21]))


def test_mimo_model_build_decode_token_group_accepts_channel_first_codes() -> None:
    model = MiMoV2ASRForCausalLM(
        _tiny_config(
            audio_channels=3,
            empty_token_id=99,
            speech_vocab_size="5-6-7",
            speech_zeroemb_idx="4-5-6",
            delay_pattern="0-1-2",
        )
    )
    speech_codes = torch.tensor([[10, 11], [20, 21], [30, 31]])

    group = model.build_decode_token_group(7, speech_codes)

    assert torch.equal(group, torch.tensor([7, 10, 20, 30, 99, 11, 21, 31]))


def test_mimo_model_build_decode_token_group_fills_zeroemb_without_speech() -> None:
    model = MiMoV2ASRForCausalLM(
        _tiny_config(empty_token_id=99, speech_zeroemb_idx="4-5")
    )

    group = model.build_decode_token_group(8, text_tail_token_id=0)

    assert torch.equal(group, torch.tensor([8, 4, 5, 0, 4, 5]))


def test_mimo_model_build_empty_decode_token_group_uses_local_forward() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(empty_token_id=99))
    calls = []

    def _fake_local_forward(hidden_states, **kwargs):
        calls.append((hidden_states, kwargs))
        return torch.tensor([[10, 20], [11, 21]])

    model.local_forward = _fake_local_forward
    hidden_states = torch.ones(1, 7)

    group = model.build_empty_decode_token_group(
        hidden_states,
        do_sample=False,
        text_tail_token_id=0,
    )

    assert torch.equal(group, torch.tensor([99, 10, 20, 0, 11, 21]))
    assert calls[0][0] is hidden_states
    assert calls[0][1]["do_sample"] is False


def test_mimo_model_build_empty_decode_token_group_validates_local_shape() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())
    model.local_forward = lambda hidden_states, **kwargs: torch.tensor([[1, 2]])

    try:
        model.build_empty_decode_token_group(torch.ones(1, 7))
    except ValueError as exc:
        assert "local_forward" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad local_forward shape should fail")


def test_mimo_model_build_decode_step_regular_token_uses_zeroemb() -> None:
    model = MiMoV2ASRForCausalLM(
        _tiny_config(empty_token_id=99, stop_token_id=42, speech_zeroemb_idx="4-5")
    )

    group, stopped = model.build_decode_step(8, text_tail_token_id=0)

    assert stopped is False
    assert torch.equal(group, torch.tensor([8, 4, 5, 0, 4, 5]))


def test_mimo_model_build_decode_step_empty_token_uses_local_forward() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(empty_token_id=99, stop_token_id=42))
    calls = []

    def _fake_local_forward(hidden_states, **kwargs):
        calls.append((hidden_states, kwargs))
        return torch.tensor([[10, 20], [11, 21]])

    model.local_forward = _fake_local_forward
    hidden_states = torch.ones(1, 7)

    group, stopped = model.build_decode_step(
        99,
        hidden_states,
        do_sample=False,
        text_tail_token_id=0,
    )

    assert stopped is False
    assert torch.equal(group, torch.tensor([99, 10, 20, 0, 11, 21]))
    assert calls[0][0] is hidden_states
    assert calls[0][1]["do_sample"] is False


def test_mimo_model_build_decode_step_stop_token_marks_stopped() -> None:
    model = MiMoV2ASRForCausalLM(
        _tiny_config(empty_token_id=99, stop_token_id=42, speech_zeroemb_idx="4-5")
    )

    group, stopped = model.build_decode_step(42, text_tail_token_id=0)

    assert stopped is True
    assert torch.equal(group, torch.tensor([42, 4, 5, 0, 4, 5]))


def test_mimo_model_build_decode_step_requires_hidden_for_empty() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(empty_token_id=99))

    try:
        model.build_decode_step(99)
    except ValueError as exc:
        assert "hidden_states" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("empty decode without hidden_states should fail")


def test_mimo_decode_groups_helper_batches_regular_empty_and_stop() -> None:
    model = MiMoV2ASRForCausalLM(
        _tiny_config(empty_token_id=99, stop_token_id=42, speech_zeroemb_idx="4-5")
    )
    calls = []

    def _fake_local_forward(hidden_states, **kwargs):
        calls.append((hidden_states, kwargs))
        return torch.tensor([[10, 20], [11, 21]])

    model.local_forward = _fake_local_forward

    groups, stopped = build_mimo_decode_groups(
        model,
        torch.tensor([8, 99, 42]),
        torch.ones(3, 7),
        do_sample=False,
        text_tail_token_id=0,
    )

    assert torch.equal(
        groups,
        torch.tensor(
            [
                [8, 4, 5, 0, 4, 5],
                [99, 10, 20, 0, 11, 21],
                [42, 4, 5, 0, 4, 5],
            ]
        ),
    )
    assert torch.equal(stopped, torch.tensor([False, False, True]))
    assert calls[0][0].shape == (1, 7)
    assert calls[0][1]["do_sample"] is False


def test_mimo_decode_groups_helper_uses_last_hidden_from_sequence() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(empty_token_id=99))
    captured = []

    def _fake_local_forward(hidden_states, **kwargs):
        captured.append(hidden_states.clone())
        return torch.tensor([[10, 20], [11, 21]])

    model.local_forward = _fake_local_forward
    hidden_states = torch.arange(42, dtype=torch.float32).reshape(1, 6, 7)

    build_mimo_decode_groups(model, torch.tensor([99]), hidden_states)

    assert torch.equal(captured[0], hidden_states[:, -1, :])


def test_mimo_decode_groups_helper_validates_shapes() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())

    try:
        build_mimo_decode_groups(model, torch.zeros(1, 1, dtype=torch.long), None)
    except ValueError as exc:
        assert "text_token_ids" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad text ids shape should fail")

    try:
        build_mimo_decode_groups(
            model,
            torch.zeros(2, dtype=torch.long),
            torch.zeros(1, 7),
        )
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad hidden batch size should fail")


def test_mimo_asr_output_processor_preserves_grouped_ids() -> None:
    processor = MiMoASROutputProcessor()
    model_output = SimpleNamespace(
        next_token_ids=torch.tensor([1, 4]),
        mimo_asr_decode_groups=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        mimo_asr_stopped=torch.tensor([False, True]),
    )
    scheduler_output = SimpleNamespace(
        requests=[
            SimpleNamespace(request_id="r0"),
            SimpleNamespace(request_id="r1"),
        ]
    )

    outputs = processor.process(model_output, scheduler_output)

    assert outputs["r0"].data == [1, 2, 3]
    assert outputs["r1"].data == [4, 5, 6]
    assert outputs["r0"].finished is False
    assert outputs["r1"].finished is True


def test_mimo_runner_post_decode_preserves_scheduler_text_ids() -> None:
    model = MiMoV2ASRForCausalLM(
        _tiny_config(empty_token_id=99, stop_token_id=42, speech_zeroemb_idx="4-5")
    )
    model.local_forward = lambda hidden_states, **kwargs: torch.tensor(
        [[10, 20], [11, 21]]
    )
    runner = object.__new__(MiMoASRModelRunner)
    runner.model = model
    result = SimpleNamespace(
        next_token_ids=torch.tensor([8, 99, 42]),
        logits_output=SimpleNamespace(hidden_states=torch.ones(3, 7)),
    )
    schedule_batch = SimpleNamespace(output_ids=torch.tensor([8, 99, 42]))

    runner.post_decode(result, None, schedule_batch, [])

    assert torch.equal(result.next_token_ids, torch.tensor([8, 99, 42]))
    assert torch.equal(schedule_batch.output_ids, torch.tensor([8, 99, 42]))
    assert torch.equal(
        result.mimo_asr_decode_groups,
        torch.tensor(
            [
                [8, 4, 5, 99, 4, 5],
                [99, 10, 20, 99, 11, 21],
                [42, 4, 5, 99, 4, 5],
            ]
        ),
    )
    assert torch.equal(result.mimo_asr_stopped, torch.tensor([False, False, True]))


def test_mimo_runner_post_prefill_samples_missing_text_ids() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())
    runner = object.__new__(MiMoASRModelRunner)
    runner.model = model
    runner._sample_next_token_ids = lambda *args: torch.tensor([7])
    result = SimpleNamespace(
        next_token_ids=None,
        logits_output=SimpleNamespace(hidden_states=torch.ones(1, 7)),
    )

    runner.post_prefill(result, "forward", "schedule", ["request"])

    assert torch.equal(result.next_token_ids, torch.tensor([7]))
    assert torch.equal(
        result.mimo_asr_decode_groups,
        torch.tensor([[7, 4, 5, model.config.empty_token_id, 4, 5]]),
    )


def test_mimo_runner_post_process_outputs_replaces_text_with_group() -> None:
    runner = object.__new__(MiMoASRModelRunner)
    result = SimpleNamespace(
        mimo_asr_decode_groups=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        mimo_asr_stopped=torch.tensor([False, True]),
    )
    scheduler_output = SimpleNamespace(
        requests=[
            SimpleNamespace(request_id="r0"),
            SimpleNamespace(request_id="r1"),
        ]
    )
    outputs = {
        "r0": SimpleNamespace(data=1, finished=False),
        "r1": SimpleNamespace(data=4, finished=False),
    }

    runner.post_process_outputs(result, scheduler_output, outputs)

    assert outputs["r0"].data == [1, 2, 3]
    assert outputs["r0"].finished is False
    assert outputs["r1"].data == [4, 5, 6]
    assert outputs["r1"].finished is True


def test_mimo_commit_decode_groups_extends_reqs_and_returns_stopped() -> None:
    reqs = [SimpleNamespace(output_ids=[1]), SimpleNamespace(output_ids=[])]
    groups = torch.tensor([[10, 11, 12], [20, 21, 22]])

    stopped = commit_mimo_decode_groups_to_reqs(
        reqs,
        groups,
        torch.tensor([False, True]),
    )

    assert reqs[0].output_ids == [1, 10, 11, 12]
    assert reqs[1].output_ids == [20, 21, 22]
    assert stopped == [reqs[1]]
    assert getattr(reqs[1], "_mimo_asr_stopped") is True


def test_mimo_commit_after_sglang_replaces_last_text_token() -> None:
    reqs = [
        SimpleNamespace(output_ids=[1, 10], finished_reason=None),
        SimpleNamespace(output_ids=[2, 20], finished_reason=None),
    ]
    batch = SimpleNamespace(reqs=reqs)
    result = SimpleNamespace(
        mimo_asr_decode_groups=torch.tensor([[10, 11, 12], [20, 21, 22]]),
        mimo_asr_stopped=torch.tensor([False, True]),
    )

    stopped = commit_mimo_decode_groups_after_sglang(batch, result)

    assert reqs[0].output_ids == [1, 10, 11, 12]
    assert reqs[0].finished_reason is None
    assert reqs[1].output_ids == [2, 20, 21, 22]
    assert stopped == [reqs[1]]
    assert reqs[1].finished_reason is not None


def test_mimo_commit_after_sglang_ignores_non_mimo_results() -> None:
    req = SimpleNamespace(output_ids=[1, 2], finished_reason=None)

    stopped = commit_mimo_decode_groups_after_sglang(
        SimpleNamespace(reqs=[req]),
        SimpleNamespace(next_token_ids=torch.tensor([2])),
    )

    assert stopped == []
    assert req.output_ids == [1, 2]


def test_mimo_commit_decode_groups_validates_shapes() -> None:
    reqs = [SimpleNamespace(output_ids=[])]

    try:
        commit_mimo_decode_groups_to_reqs(reqs, torch.zeros(1, 2, 3), [False])
    except ValueError as exc:
        assert "groups" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad groups shape should fail")

    try:
        commit_mimo_decode_groups_to_reqs(reqs, torch.zeros(2, 3), [False, False])
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad groups batch size should fail")


def test_mimo_model_build_decode_token_group_validates_speech_shape() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())

    try:
        model.build_decode_token_group(7, torch.zeros(3, 2, dtype=torch.long))
    except ValueError as exc:
        assert "speech_codes" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad speech code shape should fail")


def test_mimo_model_get_audio_feature_uses_model_specific_audio_codes() -> None:
    config = _tiny_config(hidden_size=1)
    model = MiMoV2ASRForCausalLM(config)
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
    item = SimpleNamespace(
        feature=torch.tensor([[4, 5], [4, 5]]),
        model_specific_data={"audio_codes": torch.tensor([[0, 0], [1, 1]])},
    )

    hidden = model.get_audio_feature([item])

    assert torch.equal(hidden, torch.tensor([[12.0]]))


def test_mimo_model_get_audio_feature_falls_back_to_feature() -> None:
    config = _tiny_config(hidden_size=1)
    model = MiMoV2ASRForCausalLM(config)
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
    item = SimpleNamespace(
        feature=torch.tensor([[0, 0], [1, 1]]),
        model_specific_data={},
    )

    hidden = model.get_audio_feature([item])

    assert torch.equal(hidden, torch.tensor([[12.0]]))


def test_mimo_model_get_audio_feature_concats_multiple_items() -> None:
    config = _tiny_config(hidden_size=1)
    model = MiMoV2ASRForCausalLM(config)
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
    items = [
        SimpleNamespace(feature=torch.tensor([[0, 0], [1, 1]]), model_specific_data={}),
        SimpleNamespace(feature=torch.tensor([[2, 2], [3, 3]]), model_specific_data={}),
    ]

    hidden = model.get_audio_feature(items)

    assert torch.equal(hidden, torch.tensor([[12.0], [12.0]]))


def test_mimo_model_get_audio_feature_validates_items() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())

    try:
        model.get_audio_feature([])
    except ValueError as exc:
        assert "at least one item" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("empty item list should fail")

    try:
        model.get_audio_feature([SimpleNamespace(model_specific_data={})])
    except ValueError as exc:
        assert "missing audio_codes" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("missing audio codes should fail")


def test_mimo_model_pad_input_ids_uses_existing_offsets() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())
    item = _FakeMMItem(
        torch.tensor([[0, 0], [1, 1]]),
        pad_value=-7,
        offsets=[(1, 1)],
    )
    mm_inputs = SimpleNamespace(mm_items=[item])

    padded = model.pad_input_ids([11, model.config.empty_token_id, 12], mm_inputs)

    assert padded == [11, -7, 12]
    assert item.offsets == [(1, 1)]


def test_mimo_model_pad_input_ids_infers_offsets_from_empty_tokens() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())
    item = _FakeMMItem(torch.tensor([[0, 0], [1, 1], [2, 2]]))
    mm_inputs = SimpleNamespace(mm_items=[item])

    padded = model.pad_input_ids(
        [11, model.config.empty_token_id, model.config.empty_token_id, 12],
        mm_inputs,
    )

    assert padded == [11, item.pad_value, item.pad_value, 12]
    assert item.offsets == [(1, 2)]


def test_mimo_model_pad_input_ids_handles_multiple_items() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())
    first = _FakeMMItem(torch.tensor([[0, 0], [1, 1]]), pad_value=-1)
    second = _FakeMMItem(torch.tensor([[2, 2], [3, 3], [4, 4]]), pad_value=-2)
    mm_inputs = SimpleNamespace(mm_items=[first, second])

    padded = model.pad_input_ids(
        [
            11,
            model.config.empty_token_id,
            12,
            model.config.empty_token_id,
            model.config.empty_token_id,
        ],
        mm_inputs,
    )

    assert padded == [11, -1, 12, -2, -2]
    assert first.offsets == [(1, 1)]
    assert second.offsets == [(3, 4)]


def test_mimo_model_pad_input_ids_rejects_offset_length_mismatch() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())
    item = _FakeMMItem(
        torch.tensor([[0, 0], [1, 1], [2, 2]]),
        pad_value=-1,
        offsets=[(1, 1)],
    )

    try:
        model.pad_input_ids(
            [11, model.config.empty_token_id, 12],
            SimpleNamespace(mm_items=[item]),
        )
    except ValueError as exc:
        assert "offset span length" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("offset mismatch should fail")


def test_mimo_model_pad_input_ids_rejects_non_placeholder_offset() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())
    item = _FakeMMItem(torch.tensor([[0, 0], [1, 1]]), pad_value=-1, offsets=[(1, 1)])

    try:
        model.pad_input_ids(
            [11, 999, 12],
            SimpleNamespace(mm_items=[item]),
        )
    except ValueError as exc:
        assert "non-placeholder" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("non-placeholder offset should fail")


def test_mimo_model_pad_input_ids_rejects_missing_placeholders() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())
    item = _FakeMMItem(torch.tensor([[0, 0], [1, 1], [2, 2]]), pad_value=-1)

    try:
        model.pad_input_ids(
            [11, model.config.empty_token_id, 12],
            SimpleNamespace(mm_items=[item]),
        )
    except ValueError as exc:
        assert "not contain enough" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("missing placeholders should fail")


def test_mimo_model_merge_audio_embeds_into_token_embeds_single_item() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
    token_embeds = torch.tensor([[0.0], [1.0], [2.0]])
    item = _FakeMMItem(
        torch.tensor([[0, 0], [1, 1]]),
        pad_value=-1,
        offsets=[(1, 1)],
    )

    merged = model.merge_audio_embeds_into_token_embeds(token_embeds, [item])

    assert torch.equal(merged, torch.tensor([[0.0], [12.0], [2.0]]))
    assert torch.equal(token_embeds, torch.tensor([[0.0], [1.0], [2.0]]))


def test_mimo_model_merge_audio_embeds_into_token_embeds_multiple_items() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
    token_embeds = torch.zeros(5, 1)
    items = [
        _FakeMMItem(torch.tensor([[0, 0], [1, 1]]), offsets=[(1, 1)]),
        _FakeMMItem(torch.tensor([[2, 2], [3, 3], [4, 4]]), offsets=[(3, 4)]),
    ]

    merged = model.merge_audio_embeds_into_token_embeds(token_embeds, items)

    assert torch.equal(merged, torch.tensor([[0.0], [12.0], [0.0], [12.0], [6.0]]))


def test_mimo_model_merge_audio_embeds_requires_offsets() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    item = _FakeMMItem(torch.tensor([[0, 0], [1, 1]]))

    try:
        model.merge_audio_embeds_into_token_embeds(torch.zeros(3, 1), [item])
    except ValueError as exc:
        assert "must have offsets" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("missing offsets should fail")


def test_mimo_model_merge_audio_embeds_rejects_length_mismatch() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    item = _FakeMMItem(torch.tensor([[0, 0], [1, 1], [2, 2]]), offsets=[(1, 1)])

    try:
        model.merge_audio_embeds_into_token_embeds(torch.zeros(3, 1), [item])
    except ValueError as exc:
        assert "positions must match" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("length mismatch should fail")


def test_mimo_model_merge_audio_embeds_rejects_hidden_size_mismatch() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=2))
    item = _FakeMMItem(torch.tensor([[0, 0], [1, 1]]), offsets=[(1, 1)])

    try:
        model.merge_audio_embeds_into_token_embeds(torch.zeros(3, 1), [item])
    except ValueError as exc:
        assert "hidden size" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("hidden size mismatch should fail")


def test_mimo_model_merge_audio_embeds_rejects_bad_token_embed_shape() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))

    try:
        model.merge_audio_embeds_into_token_embeds(torch.zeros(1, 2, 1), [])
    except ValueError as exc:
        assert "token_embeds" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad token embed shape should fail")


def test_mimo_model_embed_input_ids_returns_text_embeddings_without_items() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=2))
    token_embedding = torch.nn.Embedding(4, 2)
    with torch.no_grad():
        token_embedding.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                ]
            )
        )

    embeds = model.embed_input_ids(torch.tensor([1, 2, 3]), token_embedding)

    assert torch.equal(embeds, torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))


def test_mimo_model_embed_input_ids_scatters_audio_items() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    token_embedding = torch.nn.Embedding(4, 1)
    with torch.no_grad():
        token_embedding.weight.copy_(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
    item = _FakeMMItem(torch.tensor([[0, 0], [1, 1]]), offsets=[(1, 1)])

    embeds = model.embed_input_ids(torch.tensor([1, 2, 3]), token_embedding, [item])

    assert torch.equal(embeds, torch.tensor([[1.0], [12.0], [3.0]]))


def test_mimo_model_embed_input_ids_validates_input_shape() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    token_embedding = torch.nn.Embedding(4, 1)

    try:
        model.embed_input_ids(torch.zeros(1, 2, dtype=torch.long), token_embedding)
    except ValueError as exc:
        assert "input_ids" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad input_ids shape should fail")


def test_mimo_model_embed_input_ids_validates_embedding_shape() -> None:
    class _BadEmbedding(torch.nn.Module):
        def forward(self, input_ids):
            return torch.zeros(input_ids.shape[0], 1, 1)

    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))

    try:
        model.embed_input_ids(torch.tensor([1, 2]), _BadEmbedding())
    except ValueError as exc:
        assert "token_embedding" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("bad token embedding shape should fail")


def test_mimo_model_build_language_model_is_lazy_and_cached(monkeypatch) -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())
    built = object()
    calls = []
    monkeypatch.setattr(
        model,
        "_build_language_model",
        lambda: (calls.append("build") or built),
    )

    assert model.language_model is None
    assert model.build_language_model() is built
    assert model.build_language_model() is built
    assert calls == ["build"]


def test_route_mimo_weight_name_classifies_checkpoint_prefixes() -> None:
    assert route_mimo_weight_name("model.layers.0.self_attn.q_proj.weight") == "language_model"
    assert route_mimo_weight_name("lm_head.weight") == "direct"
    assert route_mimo_weight_name("speech_embeddings.0.weight") == "direct"
    assert route_mimo_weight_name("speech_group_downcast.weight") == "direct"
    assert route_mimo_weight_name("input_local_transformer.layers.0.weight") == "pending_mimo"
    assert route_mimo_weight_name("hidden_states_downcast.weight") == "direct"
    assert route_mimo_weight_name("local_transformer.layers.0.weight") == "pending_mimo"
    assert route_mimo_weight_name("local_transformer_lm_heads.0.weight") == "pending_mimo"
    assert route_mimo_weight_name("unused.weight") == "unknown"


def test_mimo_model_load_weights_loads_supported_direct_modules() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=2))
    weights = [
        ("speech_embeddings.0.weight", torch.full_like(model.speech_embeddings[0].weight, 1.5)),
        ("speech_embeddings.1.weight", torch.full_like(model.speech_embeddings[1].weight, 2.5)),
        (
            "speech_group_downcast.weight",
            torch.full_like(model.speech_group_downcast.weight, 3.5),
        ),
        (
            "hidden_states_downcast.weight",
            torch.full_like(model.hidden_states_downcast.weight, 5.5),
        ),
        ("unknown.weight", torch.tensor([1.0])),
    ]

    loaded = model.load_weights(weights)

    assert loaded == {
        "speech_embeddings.0.weight",
        "speech_embeddings.1.weight",
        "speech_group_downcast.weight",
        "hidden_states_downcast.weight",
    }
    assert torch.equal(model.speech_embeddings[0].weight, torch.full_like(model.speech_embeddings[0].weight, 1.5))
    assert torch.equal(model.speech_embeddings[1].weight, torch.full_like(model.speech_embeddings[1].weight, 2.5))
    assert torch.equal(model.speech_group_downcast.weight, torch.full_like(model.speech_group_downcast.weight, 3.5))
    assert torch.equal(model.hidden_states_downcast.weight, torch.full_like(model.hidden_states_downcast.weight, 5.5))


def test_mimo_model_load_weights_routes_language_model_when_built() -> None:
    class _FakeLanguageModel:
        def __init__(self) -> None:
            self.weights = None

        def load_weights(self, weights):
            self.weights = list(weights)

    model = MiMoV2ASRForCausalLM(_tiny_config())
    language_model = _FakeLanguageModel()
    model.language_model = language_model
    tensor = torch.tensor([1.0])

    loaded = model.load_weights([("model.layers.0.weight", tensor)])

    assert language_model.weights == [("model.layers.0.weight", tensor)]
    assert loaded == {"model.layers.0.weight"}


def test_mimo_model_load_weights_builds_language_model_when_needed() -> None:
    class _FakeLanguageModel:
        def __init__(self) -> None:
            self.weights = None

        def load_weights(self, weights):
            self.weights = list(weights)

    model = MiMoV2ASRForCausalLM(_tiny_config())
    language_model = _FakeLanguageModel()
    model._build_language_model = lambda: language_model
    tensor = torch.tensor([1.0])

    loaded = model.load_weights([("model.layers.0.weight", tensor)])

    assert model.language_model is language_model
    assert language_model.weights == [("model.layers.0.weight", tensor)]
    assert loaded == {"model.layers.0.weight"}


def test_mimo_model_load_weights_loads_injected_input_local_transformer() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2, local_dim=2))
    transformer = torch.nn.Linear(2, 2)
    model.input_local_transformer = MiMoInputLocalTransformer(transformer)
    weights = [
        (
            "input_local_transformer.weight",
            torch.full_like(transformer.weight, 1.25),
        ),
        ("input_local_transformer.bias", torch.full_like(transformer.bias, 2.25)),
    ]

    loaded = model.load_weights(weights)

    assert loaded == {
        "input_local_transformer.weight",
        "input_local_transformer.bias",
    }
    assert torch.equal(
        transformer.weight,
        torch.full_like(transformer.weight, 1.25),
    )
    assert torch.equal(transformer.bias, torch.full_like(transformer.bias, 2.25))


def test_mimo_model_load_weights_loads_default_input_local_transformer() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_layers=1))
    param = model.input_local_transformer.module.layers[0].input_layernorm.weight
    loaded_weight = torch.full_like(param, 1.75)

    loaded = model.load_weights(
        [("input_local_transformer.layers.0.input_layernorm.weight", loaded_weight)]
    )

    assert loaded == {"input_local_transformer.layers.0.input_layernorm.weight"}
    assert torch.equal(param, loaded_weight)


def test_mimo_model_load_weights_loads_local_transformer_lm_heads() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2, local_dim=2))
    head_weight = torch.full_like(model.local_transformer_lm_heads[0].weight, 3.25)

    loaded = model.load_weights(
        [("local_transformer_lm_heads.0.weight", head_weight)]
    )

    assert loaded == {"local_transformer_lm_heads.0.weight"}
    assert torch.equal(model.local_transformer_lm_heads[0].weight, head_weight)


def test_mimo_model_load_weights_loads_injected_local_transformer() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2, local_dim=2))
    transformer = torch.nn.Linear(2, 2)
    model.local_transformer = MiMoLocalTransformer(transformer)
    weights = [
        ("local_transformer.weight", torch.full_like(transformer.weight, 4.25)),
        ("local_transformer.bias", torch.full_like(transformer.bias, 5.25)),
    ]

    loaded = model.load_weights(weights)

    assert loaded == {"local_transformer.weight", "local_transformer.bias"}
    assert torch.equal(
        transformer.weight,
        torch.full_like(transformer.weight, 4.25),
    )
    assert torch.equal(transformer.bias, torch.full_like(transformer.bias, 5.25))


def test_mimo_model_load_weights_loads_default_local_transformer() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(local_layers=1))
    param = model.local_transformer.module.layers[0].input_layernorm.weight
    loaded_weight = torch.full_like(param, 6.25)

    loaded = model.load_weights(
        [("local_transformer.layers.0.input_layernorm.weight", loaded_weight)]
    )

    assert loaded == {"local_transformer.layers.0.input_layernorm.weight"}
    assert torch.equal(param, loaded_weight)


def test_mimo_model_load_weights_skips_local_transformer_non_params() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_layers=1, local_layers=1))

    loaded = model.load_weights(
        [
            ("input_local_transformer.embed_tokens.weight", torch.tensor([1.0])),
            ("local_transformer.rotary_emb.inv_freq", torch.tensor([1.0])),
        ]
    )

    assert loaded == {
        "input_local_transformer.embed_tokens.weight",
        "local_transformer.rotary_emb.inv_freq",
    }


def test_mimo_model_load_weights_rejects_unknown_input_local_transformer_param() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())

    try:
        model.load_weights(
            [("input_local_transformer.layers.0.missing.weight", torch.tensor([1.0]))]
        )
    except NotImplementedError as exc:
        assert "input_local_transformer" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("unknown input local transformer weights should fail")


def test_mimo_model_load_weights_rejects_pending_local_transformer_prefixes() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())

    try:
        model.load_weights(
            [("local_transformer.layers.0.missing.weight", torch.tensor([1.0]))]
        )
    except NotImplementedError as exc:
        assert "local_transformer" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("pending local transformer weights should fail")


class _FakeLanguageModel(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.calls: list[dict] = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("input_embeds") is not None:
            return kwargs["input_embeds"]
        return self.embedding(kwargs["input_ids"])


def _fake_language_model(hidden_size: int = 1) -> _FakeLanguageModel:
    embedding = torch.nn.Embedding(256, hidden_size)
    with torch.no_grad():
        values = torch.arange(256, dtype=torch.float32).unsqueeze(1).expand(-1, hidden_size)
        embedding.weight.copy_(values)
    return _FakeLanguageModel(embedding)


def test_mimo_model_prepare_prefill_inputs_embeds_restores_pad_positions() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    language_model = _fake_language_model(hidden_size=1)
    model.language_model = language_model
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
    item = _FakeMMItem(torch.tensor([[0, 0], [1, 1]]), pad_value=-1, offsets=[(1, 1)])

    embeds = model.prepare_prefill_inputs_embeds(torch.tensor([5, -1, 6]), [item])

    assert torch.equal(embeds, torch.tensor([[5.0], [12.0], [6.0]]))


def test_mimo_model_forward_without_audio_calls_language_model_with_input_ids() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    language_model = _fake_language_model(hidden_size=1)
    model.language_model = language_model
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2])
    forward_batch = SimpleNamespace()

    output = model.forward(input_ids, positions, forward_batch)

    assert torch.equal(output, torch.tensor([[1.0], [2.0], [3.0]]))
    assert language_model.calls[-1]["input_ids"] is input_ids
    assert language_model.calls[-1]["positions"] is positions
    assert language_model.calls[-1]["forward_batch"] is forward_batch
    assert "input_embeds" not in language_model.calls[-1]


def test_mimo_model_forward_with_audio_items_calls_language_model_with_input_embeds() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    language_model = _fake_language_model(hidden_size=1)
    model.language_model = language_model
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
    item = _FakeMMItem(torch.tensor([[0, 0], [1, 1]]), pad_value=-1, offsets=[(1, 1)])
    forward_batch = SimpleNamespace(multimodal_inputs=SimpleNamespace(mm_items=[item]))

    output = model.forward(torch.tensor([5, -1, 6]), torch.tensor([0, 1, 2]), forward_batch)

    assert torch.equal(output, torch.tensor([[5.0], [12.0], [6.0]]))
    assert torch.equal(language_model.calls[-1]["input_embeds"], output)


def test_mimo_model_forward_accepts_explicit_input_embeds() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    language_model = _fake_language_model(hidden_size=1)
    model.language_model = language_model
    input_embeds = torch.tensor([[9.0], [8.0]])

    output = model.forward(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        SimpleNamespace(),
        input_embeds=input_embeds,
    )

    assert output is input_embeds
    assert language_model.calls[-1]["input_embeds"] is input_embeds


def test_mimo_model_forward_accepts_hf_inputs_embeds_alias() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(hidden_size=1))
    language_model = _fake_language_model(hidden_size=1)
    model.language_model = language_model
    inputs_embeds = torch.tensor([[7.0], [6.0]])

    output = model.forward(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        SimpleNamespace(),
        inputs_embeds=inputs_embeds,
    )

    assert output is inputs_embeds
    assert language_model.calls[-1]["input_embeds"] is inputs_embeds
    assert "inputs_embeds" not in language_model.calls[-1]
