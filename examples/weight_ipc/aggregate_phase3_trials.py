#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate repeated Phase-3 ``phase3_summary.json`` trials into mean/std."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _mean_std(vals: list[float]) -> dict[str, float | None]:
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": round(mean, 3),
        "std": round(std, 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
        "values": [round(v, 3) for v in vals],
    }


def _get(d: dict[str, Any], *keys: str) -> float | None:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if cur is None:
        return None
    return float(cur)


def aggregate_root(root: Path) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for p in sorted(root.glob("trial_*/phase3_summary.json")):
        trials.append(json.loads(p.read_text(encoding="utf-8")))

    metrics = {
        "dp3_unshared_peak_qps": ("dp3_compare", "dp3_unshared_qps"),
        "dp3_shared_peak_qps": ("dp3_compare", "dp3_shared_qps"),
        "dp3_shared_vs_unshared_pct": ("dp3_compare", "shared_vs_unshared_pct"),
        "dp3_shared_ttfc34_qps": ("dp3_shared_ttfc34", "aggregate_throughput_qps"),
        "dp3_shared_ttfc34_p99_s": ("dp3_shared_ttfc34", "audio_ttfp_p99_s"),
        "dp4_shared_peak_qps": ("dp4_shared_peak", "aggregate_throughput_qps"),
        "dp4_shared_peak_ttfc_p99_s": ("dp4_shared_peak", "audio_ttfp_p99_s"),
        "dp4_shared_ttfc34_qps": ("dp4_shared_ttfc34", "aggregate_throughput_qps"),
        "dp4_shared_ttfc34_p99_s": ("dp4_shared_ttfc34", "audio_ttfp_p99_s"),
    }

    series: dict[str, list[float]] = {k: [] for k in metrics}
    g2_pass = 0
    for t in trials:
        for name, path in metrics.items():
            v = _get(t, *path)
            if v is not None:
                series[name].append(v)
        cmp_ = t.get("dp3_compare") or {}
        if cmp_.get("g2_pass"):
            g2_pass += 1

    stats = {k: _mean_std(v) for k, v in series.items()}
    return {
        "root": str(root),
        "n_trials": len(trials),
        "g2_pass_count": g2_pass,
        "metrics": stats,
        "trial_dirs": [str(p.parent) for p in sorted(root.glob("trial_*/phase3_summary.json"))],
    }


def render_md(agg: dict[str, Any]) -> str:
    m = agg["metrics"]

    def row(label: str, key_qps: str, key_p99: str | None = None) -> str:
        q = m[key_qps]
        qps = f"{q['mean']} ± {q['std']}" if q["mean"] is not None else "—"
        vals = ", ".join(str(x) for x in (q.get("values") or []))
        if key_p99:
            p = m[key_p99]
            p99 = f"{p['mean']} ± {p['std']}" if p["mean"] is not None else "—"
            return f"| {label} | {qps} | {p99} | `{vals}` |"
        return f"| {label} | {qps} | — | `{vals}` |"

    lines = [
        "# Phase-3 multi-trial summary (Equal KV fixed)",
        "",
        f"- Root: `{agg['root']}`",
        f"- Trials with summary: **{agg['n_trials']}**",
        f"- G2 pass count: **{agg['g2_pass_count']}** / {agg['n_trials']}",
        "",
        "## Aggregate (mean ± std)",
        "",
        "| Config | Aggregate QPS | TTFC p99 (s) | Per-trial QPS |",
        "|---|---:|---:|---|",
        row("DP3 unshared peak", "dp3_unshared_peak_qps"),
        row("DP3 shared peak", "dp3_shared_peak_qps"),
        row("DP3 shared @ ~34 offered", "dp3_shared_ttfc34_qps", "dp3_shared_ttfc34_p99_s"),
        row("DP4 shared peak", "dp4_shared_peak_qps", "dp4_shared_peak_ttfc_p99_s"),
        row("DP4 shared @ ~34 offered", "dp4_shared_ttfc34_qps", "dp4_shared_ttfc34_p99_s"),
        "",
        "## Shared vs unshared (DP3 peak)",
        "",
    ]
    pct = m["dp3_shared_vs_unshared_pct"]
    if pct["mean"] is not None:
        lines.append(
            f"- mean shared−unshared: **{pct['mean']}%** "
            f"(std {pct['std']}; values {pct.get('values')})"
        )
        lines.append(
            f"- G2 (shared not >2% slower): "
            f"{'pass' if agg['g2_pass_count'] == agg['n_trials'] else 'see per-trial'}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()
    agg = aggregate_root(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    md = render_md(agg)
    args.md.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
