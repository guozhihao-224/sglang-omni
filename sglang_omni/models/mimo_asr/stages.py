# SPDX-License-Identifier: Apache-2.0
"""Stage factory for native SGLang-backed MiMo-V2.5-ASR inference."""

from __future__ import annotations

from typing import Any

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoTokenizer

from sglang_omni.models.mimo_asr.audio_tokenizer import MiMoAudioTokenizerAdapter
from sglang_omni.models.mimo_asr.model_runner import (
    MiMoASRModelRunner,
    MiMoASROutputProcessor,
    commit_mimo_decode_groups_after_sglang,
)
from sglang_omni.models.mimo_asr.request_builders import (
    make_mimo_asr_scheduler_adapters,
)
from sglang_omni.scheduling.bootstrap import (
    create_sglang_infrastructure_defer_cuda_graph,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import build_sglang_server_args


def create_sglang_mimo_asr_executor(
    model_path: str,
    *,
    audio_tokenizer_path: str = "XiaomiMiMo/MiMo-Audio-Tokenizer",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 1,
    max_new_tokens: int = 8192,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    request_build_max_workers: int = 1,
    request_build_max_pending: int | None = 8,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Create a native MiMo-ASR scheduler wired like other ASR backends."""

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    server_args_overrides = dict(server_args_overrides or {})
    enable_async_decode = bool(server_args_overrides.pop("enable_async_decode", False))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    audio_tokenizer = MiMoAudioTokenizerAdapter(
        audio_tokenizer_path,
        device=device,
    )

    defaults: dict[str, Any] = {
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "enable_torch_compile": enable_torch_compile,
        "mem_fraction_static": mem_fraction_static,
        "max_prefill_tokens": 8192,
        "chunked_prefill_size": 8192,
        "sampling_backend": "pytorch",
        "dtype": dtype,
    }
    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        **defaults,
    )

    server_args = build_sglang_server_args(
        model_path,
        context_length=int(max_new_tokens) + 8192,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="MiMo-V2.5-ASR",
        server_args=server_args,
    )

    want_cuda_graph, (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        model_arch_override="MiMoV2ASRForCausalLM",
    )

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    init_mm_embedding_cache(mm_embedding_cache_size_bytes)

    model_worker.model_runner.model.return_hidden_states_output = True
    output_proc = MiMoASROutputProcessor()
    request_builder, result_adapter = make_mimo_asr_scheduler_adapters(
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        max_new_tokens=max_new_tokens,
    )

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=MiMoASRModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
        post_batch_result_hook=commit_mimo_decode_groups_after_sglang,
        enable_async_decode=enable_async_decode,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
    )


def create_mimo_asr_executor(*args, **kwargs):
    return create_sglang_mimo_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_mimo_asr_executor", "create_mimo_asr_executor"]
