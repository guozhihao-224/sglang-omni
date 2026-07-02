# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.mimo_asr.configuration_mimo_asr import MiMoV2ASRConfig
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
        "input_local_dim": 3,
        "speech_vocab_size": "5-6",
        "speech_zeroemb_idx": "4-5",
        "delay_pattern": "0-1",
    }
    defaults.update(overrides)
    return MiMoV2ASRConfig(**defaults)


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
    assert model.hidden_states_downcast.in_features == 7
    assert model.hidden_states_downcast.out_features == 3
    assert isinstance(model.local_transformer, MiMoLocalTransformer)
    assert len(model.local_transformer_lm_heads) == 2
    assert model.local_transformer_lm_heads[0].in_features == 3
    assert model.local_transformer_lm_heads[0].out_features == 5
    assert model.local_transformer_lm_heads[1].out_features == 6


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
        model.speech_group_downcast.bias.zero_()
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
        model.hidden_states_downcast.bias.zero_()

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
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2))
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
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2))

    try:
        model.compute_local_code_logits(torch.zeros(1, 3))
    except ValueError as exc:
        assert "input_local_dim" in str(exc)
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

    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2))
    model.local_transformer = MiMoLocalTransformer(_Shift())
    with torch.no_grad():
        model.local_transformer_lm_heads[0].weight.fill_(1.0)
        model.local_transformer_lm_heads[1].weight.fill_(1.0)

    logits = model.compute_local_code_logits(torch.tensor([[2.0, 3.0]]))

    assert torch.equal(logits[0], torch.full((1, 5), 7.0))
    assert torch.equal(logits[1], torch.full((1, 6), 7.0))


def test_mimo_model_get_audio_feature_uses_model_specific_audio_codes() -> None:
    config = _tiny_config(hidden_size=1)
    model = MiMoV2ASRForCausalLM(config)
    with torch.no_grad():
        for embedding in model.speech_embeddings:
            embedding.weight.fill_(1.0)
        model.speech_group_downcast.weight.fill_(1.0)
        model.speech_group_downcast.bias.zero_()
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
        model.speech_group_downcast.bias.zero_()
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
        model.speech_group_downcast.bias.zero_()
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
        model.speech_group_downcast.bias.zero_()
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
        model.speech_group_downcast.bias.zero_()
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
        model.speech_group_downcast.bias.zero_()
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
            "speech_group_downcast.bias",
            torch.full_like(model.speech_group_downcast.bias, 4.5),
        ),
        (
            "hidden_states_downcast.weight",
            torch.full_like(model.hidden_states_downcast.weight, 5.5),
        ),
        (
            "hidden_states_downcast.bias",
            torch.full_like(model.hidden_states_downcast.bias, 6.5),
        ),
        ("unknown.weight", torch.tensor([1.0])),
    ]

    loaded = model.load_weights(weights)

    assert loaded == {
        "speech_embeddings.0.weight",
        "speech_embeddings.1.weight",
        "speech_group_downcast.weight",
        "speech_group_downcast.bias",
        "hidden_states_downcast.weight",
        "hidden_states_downcast.bias",
    }
    assert torch.equal(model.speech_embeddings[0].weight, torch.full_like(model.speech_embeddings[0].weight, 1.5))
    assert torch.equal(model.speech_embeddings[1].weight, torch.full_like(model.speech_embeddings[1].weight, 2.5))
    assert torch.equal(model.speech_group_downcast.weight, torch.full_like(model.speech_group_downcast.weight, 3.5))
    assert torch.equal(model.speech_group_downcast.bias, torch.full_like(model.speech_group_downcast.bias, 4.5))
    assert torch.equal(model.hidden_states_downcast.weight, torch.full_like(model.hidden_states_downcast.weight, 5.5))
    assert torch.equal(model.hidden_states_downcast.bias, torch.full_like(model.hidden_states_downcast.bias, 6.5))


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

    assert language_model.weights == [("layers.0.weight", tensor)]
    assert loaded == {"model.layers.0.weight"}


def test_mimo_model_load_weights_rejects_language_weights_before_backbone() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())

    try:
        model.load_weights([("model.layers.0.weight", torch.tensor([1.0]))])
    except NotImplementedError as exc:
        assert "language_model" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("language weights should fail before backbone is built")


def test_mimo_model_load_weights_loads_injected_input_local_transformer() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2))
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


def test_mimo_model_load_weights_loads_local_transformer_lm_heads() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2))
    head_weight = torch.full_like(model.local_transformer_lm_heads[0].weight, 3.25)

    loaded = model.load_weights(
        [("local_transformer_lm_heads.0.weight", head_weight)]
    )

    assert loaded == {"local_transformer_lm_heads.0.weight"}
    assert torch.equal(model.local_transformer_lm_heads[0].weight, head_weight)


def test_mimo_model_load_weights_loads_injected_local_transformer() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config(input_local_dim=2))
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


def test_mimo_model_load_weights_rejects_default_input_local_transformer() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())

    try:
        model.load_weights(
            [("input_local_transformer.layers.0.weight", torch.tensor([1.0]))]
        )
    except NotImplementedError as exc:
        assert "input_local_transformer" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("default input local transformer weights should fail")


def test_mimo_model_load_weights_rejects_pending_local_transformer_prefixes() -> None:
    model = MiMoV2ASRForCausalLM(_tiny_config())

    try:
        model.load_weights(
            [("local_transformer.layers.0.weight", torch.tensor([1.0]))]
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
        model.speech_group_downcast.bias.zero_()
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
        model.speech_group_downcast.bias.zero_()
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
