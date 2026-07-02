# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

from sglang_omni.models.mimo_asr.config import MiMoASRPipelineConfig
from sglang_omni.models.mimo_asr.stages import create_sglang_mimo_asr_executor
from sglang_omni.models.mimo_asr.tool_funcs.audio_lengths import (
    MIMO_ASR_AUDIO_CHANNELS,
    MIMO_ASR_GROUP_SIZE,
    mimo_asr_flat_prefill_tokens,
    mimo_asr_num_empty_tokens,
    mimo_asr_padded_code_frames,
)
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_mimo_asr_config_uses_single_conservative_asr_stage() -> None:
    config = MiMoASRPipelineConfig(model_path="XiaomiMiMo/MiMo-V2.5-ASR")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_mimo_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert (
        config.stages[0].factory_args["audio_tokenizer_path"]
        == "XiaomiMiMo/MiMo-Audio-Tokenizer"
    )
    assert config.stages[0].factory_args["max_running_requests"] == 8
    assert config.stages[0].factory_args["request_build_max_workers"] == 1
    assert config.stages[0].factory_args["request_build_max_pending"] == 8
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("MiMoV2ASRForCausalLM")
        is MiMoASRPipelineConfig
    )


def test_mimo_asr_stage_signature_matches_initial_native_plan() -> None:
    signature = inspect.signature(create_sglang_mimo_asr_executor)

    assert signature.parameters["audio_tokenizer_path"].default == (
        "XiaomiMiMo/MiMo-Audio-Tokenizer"
    )
    assert signature.parameters["device"].default == "cuda:0"
    assert signature.parameters["dtype"].default == "bfloat16"
    assert signature.parameters["max_running_requests"].default == 8
    assert signature.parameters["max_new_tokens"].default == 8192
    assert signature.parameters["mem_fraction_static"].default is None
    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0
    assert signature.parameters["enable_torch_compile"].default is False
    assert signature.parameters["request_build_max_workers"].default == 1
    assert signature.parameters["request_build_max_pending"].default == 8


def test_mimo_asr_audio_code_length_helpers() -> None:
    assert MIMO_ASR_AUDIO_CHANNELS == 8
    assert MIMO_ASR_GROUP_SIZE == 4

    assert mimo_asr_padded_code_frames(0) == 0
    assert mimo_asr_padded_code_frames(1) == 4
    assert mimo_asr_padded_code_frames(4) == 4
    assert mimo_asr_padded_code_frames(5) == 8

    assert mimo_asr_num_empty_tokens(0) == 0
    assert mimo_asr_num_empty_tokens(1) == 1
    assert mimo_asr_num_empty_tokens(4) == 1
    assert mimo_asr_num_empty_tokens(5) == 2

    assert mimo_asr_flat_prefill_tokens(0) == 0
    assert mimo_asr_flat_prefill_tokens(1) == 36
    assert mimo_asr_flat_prefill_tokens(2) == 72


def test_mimo_asr_length_helpers_validate_inputs() -> None:
    for fn in (mimo_asr_padded_code_frames, mimo_asr_num_empty_tokens):
        try:
            fn(-1)
        except ValueError as exc:
            assert "num_code_frames" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("negative code frames should fail")

    try:
        mimo_asr_padded_code_frames(1, group_size=0)
    except ValueError as exc:
        assert "group_size" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("zero group size should fail")

    try:
        mimo_asr_flat_prefill_tokens(-1)
    except ValueError as exc:
        assert "text_group_count" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("negative text group count should fail")
