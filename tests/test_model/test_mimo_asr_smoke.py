# SPDX-License-Identifier: Apache-2.0
"""MiMo-ASR single-request smoke test.

This test is intentionally opt-in because it requires the real MiMo-ASR model,
the MiMo audio tokenizer, CUDA, and a local WAV fixture.  Set:

- MIMO_ASR_MODEL_PATH
- MIMO_AUDIO_TOKENIZER_PATH
- MIMO_ASR_TEST_WAV

Then run:

    python -m pytest tests/test_model/test_mimo_asr_smoke.py -q -s
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any

import pytest
import torch

from sglang_omni.models.mimo_asr.stages import create_sglang_mimo_asr_executor
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is required for MiMo-ASR smoke")
    return value


def _extract_text(data: Any) -> str:
    if isinstance(data, StagePayload):
        data = data.data
    if isinstance(data, dict):
        return str(data.get("text", ""))
    return str(data)


@pytest.mark.timeout(900)
def test_mimo_asr_single_wav_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MiMo-ASR smoke")

    model_path = _require_env("MIMO_ASR_MODEL_PATH")
    audio_tokenizer_path = _require_env("MIMO_AUDIO_TOKENIZER_PATH")
    wav_path = _require_env("MIMO_ASR_TEST_WAV")

    max_new_tokens = int(os.environ.get("MIMO_ASR_MAX_NEW_TOKENS", "256"))
    scheduler = create_sglang_mimo_asr_executor(
        model_path,
        audio_tokenizer_path=audio_tokenizer_path,
        device=os.environ.get("MIMO_ASR_DEVICE", "cuda:0"),
        max_running_requests=1,
        max_new_tokens=max_new_tokens,
        request_build_max_workers=1,
        request_build_max_pending=1,
        server_args_overrides={
            "context_length": int(os.environ.get("MIMO_ASR_CONTEXT_LENGTH", "4096")),
            "enable_async_decode": False,
        },
    )

    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()

    request_id = "mimo-asr-smoke-0"
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={"audio_path": wav_path},
            params={
                "language": os.environ.get("MIMO_ASR_LANGUAGE", "auto"),
                "max_new_tokens": max_new_tokens,
            },
        ),
        data=None,
    )

    try:
        scheduler.inbox.put(
            IncomingMessage(
                request_id=request_id,
                type="new_request",
                data=payload,
            )
        )

        deadline = time.time() + int(os.environ.get("MIMO_ASR_TIMEOUT_S", "600"))
        last_msg = None
        while time.time() < deadline:
            try:
                msg = scheduler.outbox.get(timeout=1.0)
            except queue.Empty:
                continue
            last_msg = msg
            if msg.request_id != request_id:
                continue
            if msg.type == "error":
                raise AssertionError(f"MiMo-ASR smoke failed: {msg.data!r}")
            if msg.type == "result":
                text = _extract_text(msg.data)
                assert text.strip(), f"empty transcript: {msg.data!r}"
                for token in ("<|empty|>", "<|eot|>", "<|eostm|>"):
                    assert token not in text
                return

        raise TimeoutError(f"MiMo-ASR smoke timed out; last_msg={last_msg!r}")
    finally:
        scheduler.stop()
        thread.join(timeout=30)
