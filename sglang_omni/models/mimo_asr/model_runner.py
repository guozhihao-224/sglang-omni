# SPDX-License-Identifier: Apache-2.0
"""MiMo-ASR runner helpers.

PR #898-style ASR stages use the standard OmniScheduler/ModelRunner stack. MiMo
keeps that entry shape, but its decode step expands one sampled text token into
one flattened MiMo token group.  This module holds the model-runner-side helper
logic separately from scheduler integration so it can be tested without a live
SGLang runtime.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.scheduling.types import RequestOutput, SchedulerOutput


class MiMoASRModelRunner(ModelRunner):
    """ASR runner scaffold for MiMo's grouped decode semantics.

    The full scheduler integration still needs to teach SGLang how to advance a
    request by a flattened MiMo group.  The helper below is intentionally pure:
    it consumes sampled text ids plus hidden states and returns per-row flat
    groups using methods implemented on ``MiMoV2ASRForCausalLM``.
    """

    def build_decode_groups(
        self,
        text_token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
        *,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_p: float = 0.95,
        generator: torch.Generator | None = None,
        text_tail_token_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return build_mimo_decode_groups(
            self.model,
            text_token_ids,
            hidden_states,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
            text_tail_token_id=text_tail_token_id,
        )


class MiMoASROutputProcessor:
    """Output processor for grouped MiMo decode ids.

    Standard ASR processors expose one sampled token id per request. MiMo needs
    to surface a flattened token group while preserving one scheduler row per
    request. This processor is deliberately small and does not mutate SGLang
    Req state by itself; scheduler integration must decide how to commit the
    group to KV/request state.
    """

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        groups = model_output.next_token_ids
        if groups is None:
            group_list: list[Any] = []
        elif isinstance(groups, torch.Tensor):
            group_list = groups.detach().cpu().tolist()
        else:
            group_list = groups

        outputs: dict[str, RequestOutput] = {}
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            data = group_list[row_idx] if row_idx < len(group_list) else None
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=data,
                finished=False,
            )
        return outputs


def build_mimo_decode_groups(
    model: Any,
    text_token_ids: torch.Tensor,
    hidden_states: torch.Tensor | None,
    *,
    do_sample: bool = True,
    temperature: float = 0.9,
    top_p: float = 0.95,
    generator: torch.Generator | None = None,
    text_tail_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand sampled text ids into flattened MiMo decode groups.

    Returns ``(groups, stopped)`` where ``groups`` is
    ``[batch, (audio_channels + 1) * group_size]`` and ``stopped`` is a bool
    tensor with one row per request.
    """

    text_token_ids = torch.as_tensor(text_token_ids, dtype=torch.long)
    if text_token_ids.ndim != 1:
        raise ValueError(
            "text_token_ids must be 1-D, got "
            f"shape {tuple(text_token_ids.shape)}"
        )
    batch_size = int(text_token_ids.shape[0])
    hidden_rows = _normalize_hidden_rows(hidden_states, batch_size)

    groups: list[torch.Tensor] = []
    stopped: list[bool] = []
    for row_idx, token_id in enumerate(text_token_ids.tolist()):
        row_hidden = (
            None if hidden_rows is None else hidden_rows[row_idx : row_idx + 1]
        )
        group, is_stopped = model.build_decode_step(
            int(token_id),
            row_hidden,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
            text_tail_token_id=text_tail_token_id,
        )
        groups.append(group)
        stopped.append(bool(is_stopped))

    return torch.stack(groups, dim=0), torch.tensor(
        stopped,
        dtype=torch.bool,
        device=text_token_ids.device,
    )


def _normalize_hidden_rows(
    hidden_states: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor | None:
    if hidden_states is None:
        return None
    if hidden_states.ndim == 3:
        hidden_states = hidden_states[:, -1, :]
    if hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must be [batch, hidden] or [batch, seq, hidden], "
            f"got shape {tuple(hidden_states.shape)}"
        )
    if int(hidden_states.shape[0]) != batch_size:
        raise ValueError(
            "hidden_states batch size must match text_token_ids "
            f"({hidden_states.shape[0]} != {batch_size})"
        )
    return hidden_states


__all__ = [
    "MiMoASRModelRunner",
    "MiMoASROutputProcessor",
    "build_mimo_decode_groups",
]
