# SPDX-License-Identifier: Apache-2.0
"""Length helpers for MiMo-V2.5-ASR audio-code layout."""

from __future__ import annotations

MIMO_ASR_AUDIO_CHANNELS = 8
MIMO_ASR_GROUP_SIZE = 4


def _validate_group_size(group_size: int) -> None:
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")


def _validate_code_frames(num_code_frames: int) -> None:
    if num_code_frames < 0:
        raise ValueError(
            f"num_code_frames must be >= 0, got {num_code_frames}"
        )


def mimo_asr_padded_code_frames(
    num_code_frames: int,
    group_size: int = MIMO_ASR_GROUP_SIZE,
) -> int:
    """Pad audio-code frame count to MiMo's local group size."""

    _validate_group_size(group_size)
    _validate_code_frames(num_code_frames)
    return ((num_code_frames + group_size - 1) // group_size) * group_size


def mimo_asr_num_empty_tokens(
    num_code_frames: int,
    group_size: int = MIMO_ASR_GROUP_SIZE,
) -> int:
    """Return text-row ``<|empty|>`` tokens needed for audio codes.

    MiMo-ASR maps one text placeholder to one local group of audio code frames.
    With the released checkpoint's ``group_size=4``, ``T`` code frames produce
    ``ceil(T / 4)`` placeholder tokens after padding.
    """

    return mimo_asr_padded_code_frames(num_code_frames, group_size) // group_size


def mimo_asr_flat_prefill_tokens(
    text_group_count: int,
    *,
    audio_channels: int = MIMO_ASR_AUDIO_CHANNELS,
    group_size: int = MIMO_ASR_GROUP_SIZE,
) -> int:
    """Return flat sequence tokens for grouped text + speech-channel layout."""

    if text_group_count < 0:
        raise ValueError(
            f"text_group_count must be >= 0, got {text_group_count}"
        )
    if audio_channels < 0:
        raise ValueError(f"audio_channels must be >= 0, got {audio_channels}")
    _validate_group_size(group_size)
    return text_group_count * (audio_channels + 1) * group_size
