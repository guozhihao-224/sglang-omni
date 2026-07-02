# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for MiMo-V2.5-ASR."""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

from .prompt import (
    MIMO_EMPTY_TOKEN,
    MIMO_EMPTY_TOKEN_ID,
    MIMO_STOP_TOKEN_ID,
    build_mimo_asr_prompt,
    resolve_audio_tag,
    strip_mimo_asr_special_text,
)
from .tool_funcs.audio_lengths import MIMO_ASR_GROUP_SIZE, mimo_asr_num_empty_tokens

_SAMPLE_RATE = 24000


@dataclass
class MiMoASRRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str = "auto"
    audio_tag: str | None = None
    engine_start_s: float = 0.0


def _audio_source_from_payload(payload: StagePayload) -> Any:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file"):
            value = inputs.get(key)
            if value is not None:
                return value
        for key in ("audio_path", "path", "url"):
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


def load_audio(source: Any) -> np.ndarray:
    """Load mono float32 audio at MiMo's 24 kHz input sample rate."""

    import torchaudio

    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, bytearray):
        source = bytes(source)

    if isinstance(source, bytes):
        audio, sample_rate = torchaudio.load(io.BytesIO(source))
    elif isinstance(source, str):
        audio, sample_rate = torchaudio.load(source)
    else:
        raise ValueError(f"Unsupported MiMo-ASR audio input: {type(source).__name__}")

    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0).to(torch.float32)
    if sample_rate != _SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, _SAMPLE_RATE)
    return audio.cpu().numpy()


def _audio_fingerprint(audio: np.ndarray, codes: torch.Tensor) -> str:
    audio_bytes = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    code_bytes = codes.detach().cpu().contiguous().numpy().astype(np.int64).tobytes()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(audio_bytes)
    digest.update(code_bytes)
    return digest.hexdigest()


def _audio_fingerprint_int(fingerprint: str) -> int:
    return int(fingerprint[:16], 16)


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer(prompt, add_special_tokens=False)
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return list(encoded["input_ids"])


def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def _to_code_tensor(codes: Any) -> torch.Tensor:
    if isinstance(codes, torch.Tensor):
        tensor = codes.to(dtype=torch.long)
    else:
        tensor = torch.tensor(codes, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"MiMo-ASR audio codes must be [T, C], got {tuple(tensor.shape)}")
    return tensor


def _extract_text_channel(output_ids: list[int], group_size: int) -> list[int]:
    if group_size <= 1:
        return list(output_ids)
    return list(output_ids[::group_size])


def make_mimo_asr_scheduler_adapters(
    *,
    tokenizer: Any,
    audio_tokenizer: Any,
    max_new_tokens: int,
) -> tuple[Callable[[StagePayload], MiMoASRRequestData], Callable[[Any], StagePayload]]:
    """Build request/result adapters for MiMo-ASR."""

    vocab_size = int(getattr(tokenizer, "vocab_size", 151680))

    def request_builder(payload: StagePayload) -> MiMoASRRequestData:
        params = payload.request.params or {}
        audio = load_audio(_audio_source_from_payload(payload))
        audio_duration_s = float(len(audio) / _SAMPLE_RATE)
        codes = _to_code_tensor(audio_tokenizer.encode(audio, sample_rate=_SAMPLE_RATE))
        num_code_frames = int(codes.shape[0])
        num_empty_tokens = mimo_asr_num_empty_tokens(num_code_frames)
        language = str(params.get("language") or "auto")
        audio_tag = resolve_audio_tag(language, params.get("audio_tag"))
        prompt = build_mimo_asr_prompt(num_code_frames, audio_tag=audio_tag)
        input_ids = _encode_prompt(tokenizer, prompt)

        empty_token_id = int(
            getattr(tokenizer, "convert_tokens_to_ids", lambda token: MIMO_EMPTY_TOKEN_ID)(
                MIMO_EMPTY_TOKEN
            )
        )
        if empty_token_id < 0:
            empty_token_id = MIMO_EMPTY_TOKEN_ID

        audio_item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=_audio_fingerprint_int(_audio_fingerprint(audio, codes)),
            feature=codes,
            model_specific_data={
                "audio_codes": codes,
                "audio_lengths": [num_code_frames],
                "num_empty_tokens": num_empty_tokens,
                "group_size": MIMO_ASR_GROUP_SIZE,
            },
        )
        audio_item.set_pad_value()
        empty_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == empty_token_id]
        if len(empty_positions) != num_empty_tokens:
            raise ValueError(
                "MiMo-ASR prompt placeholder mismatch: "
                f"expected {num_empty_tokens}, found {len(empty_positions)}"
            )
        input_ids = [audio_item.pad_value if token_id == empty_token_id else token_id for token_id in input_ids]
        if empty_positions:
            audio_item.offsets = [(empty_positions[0], empty_positions[-1])]
        else:
            audio_item.offsets = []

        mm_inputs = MultimodalInputs(
            mm_items=[audio_item],
            num_image_tokens=num_empty_tokens,
        )
        mm_inputs.audio_token_id = empty_token_id

        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop_token_ids=[MIMO_STOP_TOKEN_ID],
        )
        sampling_params.normalize(tokenizer=None)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
            extra_key=audio_item.hash,
        )
        req.multimodal_inputs = mm_inputs
        req._codec_suppress_tokens = None

        return MiMoASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=0.0,
            audio_duration_s=audio_duration_s,
            language=language,
            audio_tag=audio_tag,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: MiMoASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        text_ids = _extract_text_channel(output_ids, MIMO_ASR_GROUP_SIZE)
        text = strip_mimo_asr_special_text(_decode_token_ids(tokenizer, text_ids))
        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "language": data.language,
                "audio_tag": data.audio_tag,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": engine_time_s,
                "usage": {"engine_time_s": engine_time_s},
                "modality": "text",
            },
        )

    return request_builder, result_adapter


__all__ = [
    "MiMoASRRequestData",
    "load_audio",
    "make_mimo_asr_scheduler_adapters",
]
