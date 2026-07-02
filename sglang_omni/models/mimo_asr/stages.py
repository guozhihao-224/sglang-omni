# SPDX-License-Identifier: Apache-2.0
"""Stage factory for native SGLang-backed MiMo-V2.5-ASR inference."""

from __future__ import annotations

from typing import Any


def create_sglang_mimo_asr_executor(
    model_path: str,
    *,
    audio_tokenizer_path: str = "XiaomiMiMo/MiMo-Audio-Tokenizer",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 8,
    max_new_tokens: int = 8192,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    request_build_max_workers: int = 1,
    request_build_max_pending: int | None = 8,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Create a MiMo-ASR executor.

    The public signature is intentionally added before the full model port so
    pipeline config and CLI routing can be tested without importing SGLang.
    """

    raise NotImplementedError(
        "MiMo-ASR native executor is not implemented yet. "
        "Next phases need request builders, audio tokenizer, and custom decode."
    )


__all__ = ["create_sglang_mimo_asr_executor"]
