#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate per-replica SeedTTS ``speed_results.json`` into one pool summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"missing summary in {path}")
    return summary


def aggregate(paths: list[Path]) -> dict:
    summaries = [_load_summary(p) for p in paths]
    qps = [float(s.get("throughput_qps") or 0.0) for s in summaries]
    completed = [
        int(s.get("completed_requests") or s.get("completed") or 0) for s in summaries
    ]
    failed = [
        int(s.get("failed_requests") or s.get("failed") or 0) for s in summaries
    ]

    def mean_key(key: str) -> float | None:
        vals = [float(s[key]) for s in summaries if s.get(key) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    # Pool TTFC: average of per-replica percentiles (approximate; for exact
    # pool percentiles, merge raw per-request TTFC lists).
    return {
        "n_replicas": len(summaries),
        "aggregate_throughput_qps": round(sum(qps), 3),
        "per_replica_throughput_qps": [round(x, 3) for x in qps],
        "completed_total": sum(completed),
        "failed_total": sum(failed),
        "audio_ttfp_mean_s": _round_or_none(mean_key("audio_ttfp_mean_s")),
        "audio_ttfp_median_s": _round_or_none(mean_key("audio_ttfp_median_s")),
        "audio_ttfp_p95_s": _round_or_none(mean_key("audio_ttfp_p95_s")),
        "audio_ttfp_p99_s": _round_or_none(mean_key("audio_ttfp_p99_s")),
        "latency_mean_s": _round_or_none(mean_key("latency_mean_s")),
        "latency_p99_s": _round_or_none(mean_key("latency_p99_s")),
        "sources": [str(p) for p in paths],
    }


def _round_or_none(value: float | None) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("speed_json", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()
    result = aggregate(args.speed_json)
    result["label"] = args.label
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
