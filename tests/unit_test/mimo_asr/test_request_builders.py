# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import sglang_omni.models.mimo_asr.request_builders as request_builders
from sglang_omni.models.mimo_asr.prompt import (
    MIMO_AUDIO_TAG_CHINESE,
    MIMO_AUDIO_TAG_ENGLISH,
    MIMO_EMPTY_TOKEN,
    MIMO_EMPTY_TOKEN_ID,
    build_mimo_asr_prompt,
    resolve_audio_tag,
    strip_mimo_asr_special_text,
)
from sglang_omni.models.mimo_asr.request_builders import (
    MiMoASRRequestData,
    make_mimo_asr_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeTokenizer:
    vocab_size = 151680

    def __init__(self) -> None:
        self.decoded: list[dict] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == MIMO_EMPTY_TOKEN
        return MIMO_EMPTY_TOKEN_ID

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert not add_special_tokens
        empty_count = text.count(MIMO_EMPTY_TOKEN)
        audio_tag_id = 700 if MIMO_AUDIO_TAG_CHINESE in text else 701 if MIMO_AUDIO_TAG_ENGLISH in text else 0
        return SimpleNamespace(
            input_ids=[11, audio_tag_id] + [MIMO_EMPTY_TOKEN_ID] * empty_count + [12]
        )

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        self.decoded.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        pieces = {
            20: " hello",
            21: " world",
            30: MIMO_EMPTY_TOKEN,
            31: MIMO_AUDIO_TAG_CHINESE,
            99: "<|eot|>",
        }
        return "".join(pieces[token_id] for token_id in token_ids)


class _FakeAudioTokenizer:
    def __init__(self, frames: int = 5) -> None:
        self.frames = frames
        self.calls: list[dict] = []

    def encode(self, audio, *, sample_rate: int):
        self.calls.append({"samples": len(audio), "sample_rate": sample_rate})
        return torch.arange(self.frames * 8, dtype=torch.long).reshape(self.frames, 8)


def test_mimo_prompt_helpers_map_language_and_placeholders() -> None:
    assert resolve_audio_tag("zh") == MIMO_AUDIO_TAG_CHINESE
    assert resolve_audio_tag("english") == MIMO_AUDIO_TAG_ENGLISH
    assert resolve_audio_tag("auto") is None
    assert resolve_audio_tag("zh", "<custom>") == "<custom>"

    prompt = build_mimo_asr_prompt(5, audio_tag=MIMO_AUDIO_TAG_CHINESE)

    assert prompt.count(MIMO_EMPTY_TOKEN) == 2
    assert MIMO_AUDIO_TAG_CHINESE in prompt
    assert strip_mimo_asr_special_text(
        f"{MIMO_EMPTY_TOKEN} hello {MIMO_AUDIO_TAG_CHINESE}<|eot|>"
    ) == "hello"


def test_mimo_request_builder_records_empty_offsets_and_audio_codes(monkeypatch) -> None:
    monkeypatch.setattr(
        request_builders,
        "load_audio",
        lambda source: np.zeros(2400, dtype=np.float32),
    )
    audio_tokenizer = _FakeAudioTokenizer(frames=5)
    request_builder, _ = make_mimo_asr_scheduler_adapters(
        tokenizer=_FakeTokenizer(),
        audio_tokenizer=audio_tokenizer,
        max_new_tokens=128,
    )
    payload = StagePayload(
        request_id="req-mimo",
        request=OmniRequest(
            inputs={"audio_bytes": b"wav"},
            params={"language": "zh", "max_new_tokens": 64},
        ),
        data={},
    )

    data = request_builder(payload)

    assert audio_tokenizer.calls == [{"samples": 2400, "sample_rate": 24000}]
    assert data.language == "zh"
    assert data.audio_tag == MIMO_AUDIO_TAG_CHINESE
    assert data.max_new_tokens == 64
    assert data.audio_duration_s == 0.1
    audio_item = data.req.multimodal_inputs.mm_items[0]
    assert audio_item.feature.shape == (5, 8)
    assert audio_item.model_specific_data["num_empty_tokens"] == 2
    assert audio_item.model_specific_data["audio_lengths"] == [5]
    start, end = audio_item.offsets[0]
    assert end - start + 1 == 2
    assert data.prompt_token_ids[start : end + 1] == [audio_item.pad_value] * 2


def test_mimo_request_builder_fails_on_placeholder_mismatch(monkeypatch) -> None:
    class _BadTokenizer(_FakeTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            return SimpleNamespace(input_ids=[11, MIMO_EMPTY_TOKEN_ID, 12])

    monkeypatch.setattr(
        request_builders,
        "load_audio",
        lambda source: np.zeros(2400, dtype=np.float32),
    )
    request_builder, _ = make_mimo_asr_scheduler_adapters(
        tokenizer=_BadTokenizer(),
        audio_tokenizer=_FakeAudioTokenizer(frames=5),
        max_new_tokens=128,
    )
    payload = StagePayload(
        request_id="req-mimo",
        request=OmniRequest(inputs={"audio_bytes": b"wav"}),
        data={},
    )

    try:
        request_builder(payload)
    except ValueError as exc:
        assert "placeholder mismatch" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("placeholder mismatch should fail")


def test_mimo_result_adapter_extracts_text_channel_and_strips_specials() -> None:
    tokenizer = _FakeTokenizer()
    _, result_adapter = make_mimo_asr_scheduler_adapters(
        tokenizer=tokenizer,
        audio_tokenizer=_FakeAudioTokenizer(),
        max_new_tokens=128,
    )
    payload = StagePayload(
        request_id="req-mimo",
        request=OmniRequest(inputs={}),
        data={},
    )
    data = MiMoASRRequestData(
        output_ids=[20, 1000, 1001, 1002, 30, 1003, 1004, 1005, 21, 1006, 1007, 1008, 31, 1009, 1010, 1011, 99],
        stage_payload=payload,
        language="zh",
        audio_tag=MIMO_AUDIO_TAG_CHINESE,
        audio_duration_s=1.25,
    )

    result = result_adapter(data)

    assert result.data["text"] == "hello world"
    assert result.data["audio_tag"] == MIMO_AUDIO_TAG_CHINESE
    assert tokenizer.decoded[-1] == {
        "token_ids": [20, 30, 21, 31, 99],
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
    }
