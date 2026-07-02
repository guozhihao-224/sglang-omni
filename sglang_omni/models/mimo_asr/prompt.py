# SPDX-License-Identifier: Apache-2.0
"""Prompt helpers for MiMo-V2.5-ASR."""

from __future__ import annotations

from .tool_funcs.audio_lengths import mimo_asr_num_empty_tokens

MIMO_EMPTY_TOKEN = "<|empty|>"
MIMO_EMPTY_TOKEN_ID = 151667
MIMO_STOP_TOKEN_ID = 151645
MIMO_AUDIO_TAG_CHINESE = "<chinese>"
MIMO_AUDIO_TAG_ENGLISH = "<english>"
MIMO_STRIP_TOKENS = (
    MIMO_EMPTY_TOKEN,
    "<|eot|>",
    "<|eostm|>",
    MIMO_AUDIO_TAG_CHINESE,
    MIMO_AUDIO_TAG_ENGLISH,
)


def resolve_audio_tag(language: str | None = None, audio_tag: str | None = None) -> str | None:
    """Resolve user language/audio_tag params to MiMo's ASR tag."""

    if audio_tag:
        tag = audio_tag.strip()
        return tag or None
    if language is None:
        return None
    lang = language.strip().lower()
    if not lang or lang == "auto":
        return None
    if lang in {"zh", "cn", "zho", "chi", "chinese"} or lang.startswith("zh"):
        return MIMO_AUDIO_TAG_CHINESE
    if lang in {"en", "eng", "english"} or lang.startswith("en"):
        return MIMO_AUDIO_TAG_ENGLISH
    return None


def build_mimo_asr_prompt(num_code_frames: int, *, audio_tag: str | None = None) -> str:
    """Build a deterministic MiMo-ASR prompt skeleton.

    The full implementation should be kept byte-identical to official
    ``get_asr_sft_prompt`` once the tokenizer/front-end port lands.  This helper
    centralizes the placeholder count and produces a stable testable prompt now.
    """

    num_empty_tokens = mimo_asr_num_empty_tokens(num_code_frames)
    audio_placeholders = MIMO_EMPTY_TOKEN * num_empty_tokens
    tag_suffix = audio_tag or ""
    return (
        "<|im_start|>user\n"
        "Please transcribe the following audio.\n"
        f"{audio_placeholders}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{tag_suffix}"
    )


def strip_mimo_asr_special_text(text: str) -> str:
    """Remove MiMo control tags that may appear in decoded transcripts."""

    for token in MIMO_STRIP_TOKENS:
        text = text.replace(token, "")
    return text.strip()


__all__ = [
    "MIMO_AUDIO_TAG_CHINESE",
    "MIMO_AUDIO_TAG_ENGLISH",
    "MIMO_EMPTY_TOKEN",
    "MIMO_EMPTY_TOKEN_ID",
    "MIMO_STOP_TOKEN_ID",
    "build_mimo_asr_prompt",
    "resolve_audio_tag",
    "strip_mimo_asr_special_text",
]
