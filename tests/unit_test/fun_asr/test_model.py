# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang_omni.models.fun_asr.sglang_model import (
    EncoderLayerSANM,
    FunAsrNanoAdaptor,
    FunAsrNanoAudioEncoder,
    FunAsrNanoForConditionalGeneration,
    MultiHeadedAttention,
    MultiHeadedAttentionSANM,
)


def test_fun_asr_audio_modules_match_current_checkpoint_parameter_names() -> None:
    encoder = FunAsrNanoAudioEncoder(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=3,
    )
    encoder_names = set(dict(encoder.named_parameters()))

    assert "stem.self_attn.qkv_proj.weight" in encoder_names
    assert "stem.self_attn.qkv_proj.bias" in encoder_names
    assert "stem.self_attn.q_proj.weight" not in encoder_names
    assert "stem.self_attn.k_proj.weight" not in encoder_names
    assert "stem.self_attn.v_proj.weight" not in encoder_names
    assert encoder.stem.self_attn.qkv_proj.weight.shape == (24, 8)
    assert "stem.self_attn.out_proj.weight" in encoder_names
    assert "stem.fsmn.conv.weight" in encoder_names
    assert "stem.fc1.weight" in encoder_names
    assert "layers.0.self_attn_layer_norm.weight" in encoder_names
    assert "layers.0.final_layer_norm.weight" in encoder_names
    assert "layer_norm.weight" in encoder_names
    assert "timestamp_prediction_layers.0.fc2.weight" in encoder_names
    assert "timestamp_prediction_layer_norm.weight" in encoder_names

    projector = FunAsrNanoAdaptor(
        encoder_dim=8,
        llm_dim=8,
        ffn_dim=16,
        num_layers=1,
        attention_heads=2,
    )
    projector_names = set(dict(projector.named_parameters()))

    assert "linear_1.weight" in projector_names
    assert "linear_2.weight" in projector_names
    assert "blocks.0.self_attn.q_proj.weight" in projector_names
    assert "blocks.0.self_attn_layer_norm.weight" in projector_names
    assert "blocks.0.fc1.weight" in projector_names
    assert "blocks.0.final_layer_norm.weight" in projector_names


def _weight_loader_target() -> FunAsrNanoForConditionalGeneration:
    model = FunAsrNanoForConditionalGeneration.__new__(
        FunAsrNanoForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(tie_word_embeddings=False)
    )
    model.audio_tower = nn.Module()
    model.audio_tower.layer_norm = nn.LayerNorm(2)
    model.multi_modal_projector = nn.Module()
    model.multi_modal_projector.linear_1 = nn.Linear(2, 2)
    return model


def test_fun_asr_weight_loader_loads_current_audio_prefixes() -> None:
    model = _weight_loader_target()
    expected = torch.tensor([2.0, 3.0])

    model.load_weights([("model.audio_tower.layer_norm.weight", expected.clone())])

    assert torch.equal(model.audio_tower.layer_norm.weight, expected)


def test_fun_asr_weight_loader_rejects_unknown_audio_weights() -> None:
    model = _weight_loader_target()

    with pytest.raises(ValueError, match=r"model\.audio_tower\.missing\.weight"):
        model.load_weights([("model.audio_tower.missing.weight", torch.ones(2))])


def test_fun_asr_weight_loader_stacks_sanm_qkv_from_hf_shards() -> None:
    model = FunAsrNanoForConditionalGeneration.__new__(
        FunAsrNanoForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(tie_word_embeddings=False)
    )
    model.audio_tower = nn.Module()
    model.audio_tower.stem = nn.Module()
    model.audio_tower.stem.self_attn = MultiHeadedAttentionSANM(
        n_head=2, in_feat=4, n_feat=4, dropout_rate=0.0
    )
    model.multi_modal_projector = nn.Module()

    q_w = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    k_w = torch.arange(16, 32, dtype=torch.float32).reshape(4, 4)
    v_w = torch.arange(32, 48, dtype=torch.float32).reshape(4, 4)
    q_b = torch.tensor([1.0, 2.0, 3.0, 4.0])
    k_b = torch.tensor([5.0, 6.0, 7.0, 8.0])
    v_b = torch.tensor([9.0, 10.0, 11.0, 12.0])

    model.load_weights(
        [
            ("model.audio_tower.stem.self_attn.q_proj.weight", q_w.clone()),
            ("model.audio_tower.stem.self_attn.k_proj.weight", k_w.clone()),
            ("model.audio_tower.stem.self_attn.v_proj.weight", v_w.clone()),
            ("model.audio_tower.stem.self_attn.q_proj.bias", q_b.clone()),
            ("model.audio_tower.stem.self_attn.k_proj.bias", k_b.clone()),
            ("model.audio_tower.stem.self_attn.v_proj.bias", v_b.clone()),
        ]
    )

    qkv = model.audio_tower.stem.self_attn.qkv_proj
    assert torch.equal(qkv.weight[:4], q_w)
    assert torch.equal(qkv.weight[4:8], k_w)
    assert torch.equal(qkv.weight[8:], v_w)
    assert torch.equal(qkv.bias[:4], q_b)
    assert torch.equal(qkv.bias[4:8], k_b)
    assert torch.equal(qkv.bias[8:], v_b)


def test_sanm_attention_shares_v_with_fsmn_path() -> None:
    torch.manual_seed(0)
    layer = EncoderLayerSANM(
        in_size=8,
        size=8,
        attention_heads=2,
        linear_units=16,
        kernel_size=3,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        activation_dropout_rate=0.0,
        activation_function="relu",
    )
    layer.eval()
    x = torch.randn(2, 5, 8)

    # Separate q/k/v path (legacy) vs fused qkv must match when weights align.
    q_w, k_w, v_w = torch.split(layer.self_attn.qkv_proj.weight.detach(), 8, dim=0)
    q_b, k_b, v_b = torch.split(layer.self_attn.qkv_proj.bias.detach(), 8, dim=0)

    with torch.no_grad():
        x_norm = layer.self_attn_layer_norm(x)
        q = torch.nn.functional.linear(x_norm, q_w, q_b)
        k = torch.nn.functional.linear(x_norm, k_w, k_b)
        v = torch.nn.functional.linear(x_norm, v_w, v_b)
        attn_out, v_shared = layer.self_attn(x_norm)

    assert torch.allclose(v_shared, v, atol=1e-5, rtol=1e-5)
    b, t, _ = x_norm.size()
    q_h = q.view(b, t, 2, 4).transpose(1, 2)
    k_h = k.view(b, t, 2, 4).transpose(1, 2)
    v_h = v.view(b, t, 2, 4).transpose(1, 2)
    ref_attn = torch.nn.functional.scaled_dot_product_attention(
        q_h, k_h, v_h, dropout_p=0.0, is_causal=False
    )
    ref_attn = ref_attn.transpose(1, 2).contiguous().view(b, t, 8)
    ref_attn = layer.self_attn.out_proj(ref_attn)
    assert torch.allclose(attn_out, ref_attn, atol=1e-5, rtol=1e-5)

    out = layer(x)
    assert out.shape == x.shape


def test_adaptor_attention_uses_sdpa() -> None:
    torch.manual_seed(0)
    attn = MultiHeadedAttention(n_head=2, n_feat=8, dropout_rate=0.0)
    attn.eval()
    x = torch.randn(2, 5, 8)

    with torch.no_grad():
        out = attn(x)
        q_h = attn.q_proj(x).view(2, 5, 2, 4).transpose(1, 2)
        k_h = attn.k_proj(x).view(2, 5, 2, 4).transpose(1, 2)
        v_h = attn.v_proj(x).view(2, 5, 2, 4).transpose(1, 2)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q_h, k_h, v_h, dropout_p=0.0, is_causal=False
        )
        ref = attn.out_proj(ref.transpose(1, 2).contiguous().view(2, 5, 8))

    assert out.shape == x.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)