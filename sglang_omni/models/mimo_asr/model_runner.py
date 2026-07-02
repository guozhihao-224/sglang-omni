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

    SGLang's scheduler still advances KV/cache state with one text token per
    request.  MiMo's full flattened decode group is therefore carried on a
    MiMo-specific result attribute for streaming/result adaptation instead of
    replacing ``next_token_ids`` or ``schedule_batch.output_ids``.
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

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._expand_sampled_text_ids(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        self._expand_sampled_text_ids(result, forward_batch, schedule_batch, requests)

    def _expand_sampled_text_ids(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if result.next_token_ids is None:
            result.next_token_ids = self._sample_next_token_ids(
                result.logits_output,
                forward_batch,
                schedule_batch,
                requests,
            )
        hidden_states = getattr(result.logits_output, "hidden_states", None)
        text_token_ids = result.next_token_ids
        groups, stopped = self.build_decode_groups(
            text_token_ids,
            hidden_states,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
        result.mimo_asr_decode_groups = groups.to(device=text_token_ids.device)
        result.mimo_asr_stopped = stopped.to(device=text_token_ids.device)

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        groups = getattr(result, "mimo_asr_decode_groups", None)
        if groups is None:
            return
        stopped = getattr(result, "mimo_asr_stopped", None)
        stopped_values = _stopped_to_list(stopped) if stopped is not None else []
        group_rows = groups.detach().cpu().tolist()
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            if row_idx >= len(group_rows):
                continue
            req_output = outputs[sched_req.request_id]
            req_output.data = group_rows[row_idx]
            if row_idx < len(stopped_values):
                req_output.finished = bool(stopped_values[row_idx])


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
        groups = getattr(model_output, "mimo_asr_decode_groups", None)
        if groups is None:
            groups = model_output.next_token_ids
        stopped = getattr(model_output, "mimo_asr_stopped", None)
        stopped_values = _stopped_to_list(stopped) if stopped is not None else []
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
                finished=bool(stopped_values[row_idx])
                if row_idx < len(stopped_values)
                else False,
            )
        return outputs


def commit_mimo_decode_groups_to_reqs(
    reqs: list[Any],
    groups: torch.Tensor,
    stopped: torch.Tensor | list[bool] | tuple[bool, ...],
) -> list[Any]:
    """Append flat MiMo groups to Req.output_ids and return stopped reqs.

    This intentionally does not call SGLang's ``check_finished`` because the
    stop token is at the first text slot of a flat group, not necessarily the
    last id appended. Scheduler integration should mark returned reqs finished
    using the runtime's native finish-reason API.
    """

    if groups.ndim != 2:
        raise ValueError(
            f"groups must be [batch, group_len], got {tuple(groups.shape)}"
        )
    if int(groups.shape[0]) != len(reqs):
        raise ValueError(
            "groups batch size must match reqs "
            f"({groups.shape[0]} != {len(reqs)})"
        )
    stopped_values = _stopped_to_list(stopped)
    if len(stopped_values) != len(reqs):
        raise ValueError(
            "stopped length must match reqs "
            f"({len(stopped_values)} != {len(reqs)})"
        )

    stopped_reqs: list[Any] = []
    group_rows = groups.detach().cpu().tolist()
    for req, group, is_stopped in zip(reqs, group_rows, stopped_values, strict=True):
        req.output_ids.extend(int(token_id) for token_id in group)
        if is_stopped:
            setattr(req, "_mimo_asr_stopped", True)
            stopped_reqs.append(req)
    return stopped_reqs


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


def _stopped_to_list(stopped: torch.Tensor | list[bool] | tuple[bool, ...]) -> list[bool]:
    if isinstance(stopped, torch.Tensor):
        return [bool(value) for value in stopped.detach().cpu().tolist()]
    return [bool(value) for value in stopped]


__all__ = [
    "commit_mimo_decode_groups_to_reqs",
    "MiMoASRModelRunner",
    "MiMoASROutputProcessor",
    "build_mimo_decode_groups",
]
