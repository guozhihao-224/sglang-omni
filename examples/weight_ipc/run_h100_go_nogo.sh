#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# H100 weight-IPC Go/No-Go helper: DP2 parity, then optional DP4@100k startup.
#
# Prerequisites:
#   - Idle H100, CUDA MPS available, numactl
#   - sgl-omni on PATH (e.g. export PATH=/path/to/.venv/bin:$PATH)
#   - Higgs weights available locally or via HF cache
#
# Examples:
#   GPU_ID=0 CORE_BLOCKS="0-7 8-15" \
#     bash examples/weight_ipc/run_h100_go_nogo.sh dp2-parity
#
#   GPU_ID=0 CORE_BLOCKS="0-7 8-15 16-23 24-31" \
#     bash examples/weight_ipc/run_h100_go_nogo.sh dp4-startup
#
#   bash examples/weight_ipc/run_h100_go_nogo.sh down [RUN_ID]
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LAUNCH="$ROOT/examples/mps_dp/launch.sh"
PARITY="$ROOT/examples/weight_ipc/validate_dp_parity.py"
CMD=${1:-}
RUN_ARG=${2:-}

die() { echo "error: $*" >&2; exit 1; }

need_sgl_omni() {
  command -v sgl-omni >/dev/null || die "sgl-omni not on PATH; activate the project venv first"
  command -v numactl >/dev/null || die "numactl is required"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
}

latest_run_for_gpu() {
  local gpu=$1
  local state_root=${STATE_ROOT:-/tmp/sglang-omni-same-gpu-dp/$UID}
  # Newest run directory for this GPU (mtime).
  ls -1dt "$state_root"/gpu-"$gpu"/run-* 2>/dev/null | head -n1
}

dp2_parity() {
  need_sgl_omni
  local gpu=${GPU_ID:-0}
  local n=2
  local blocks=${CORE_BLOCKS:-}
  local tokens=${MAX_TOTAL_TOKENS:-100000}
  local model=${MODEL:-bosonai/higgs-tts-3-4b}
  local base_port=${BASE_PORT:-8801}
  local parity_n=${PARITY_N:-30}
  [ -n "$blocks" ] || die "CORE_BLOCKS is required (N=$n non-overlapping NUMA blocks)"

  echo "==> launching DP2 WEIGHT_IPC=1 on GPU $gpu"
  WEIGHT_IPC=1 N=$n GPU_ID=$gpu CORE_BLOCKS="$blocks" \
    MAX_TOTAL_TOKENS=$tokens MODEL="$model" BASE_PORT=$base_port \
    bash "$LAUNCH" up

  local state
  state=$(latest_run_for_gpu "$gpu")
  [ -n "$state" ] || die "no launcher state after up for GPU $gpu"
  echo "==> run state: $state"
  [ -f "$state/weight_ipc/READY" ] || die "missing weight IPC READY under $state"

  local out=${OUT_JSON:-$state/logs/dp2_parity.json}
  echo "==> running leader/follower parity (n=$parity_n)"
  python "$PARITY" \
    --leader-url "http://127.0.0.1:$base_port" \
    --follower-url "http://127.0.0.1:$((base_port+1))" \
    --n "$parity_n" \
    --out-json "$out"
  echo "==> parity JSON: $out"
  echo "==> leave servers up for inspection, or: bash $0 down $(basename "$state")"
}

dp4_startup() {
  need_sgl_omni
  local gpu=${GPU_ID:-0}
  local n=4
  local blocks=${CORE_BLOCKS:-}
  local tokens=${MAX_TOTAL_TOKENS:-100000}
  local model=${MODEL:-bosonai/higgs-tts-3-4b}
  local base_port=${BASE_PORT:-8801}
  [ -n "$blocks" ] || die "CORE_BLOCKS is required (N=$n non-overlapping NUMA blocks)"

  echo "==> launching DP4 WEIGHT_IPC=1 @ Equal KV=$tokens on GPU $gpu"
  WEIGHT_IPC=1 N=$n GPU_ID=$gpu CORE_BLOCKS="$blocks" \
    MAX_TOTAL_TOKENS=$tokens MODEL="$model" BASE_PORT=$base_port \
    bash "$LAUNCH" up

  local state
  state=$(latest_run_for_gpu "$gpu")
  [ -n "$state" ] || die "no launcher state after up for GPU $gpu"
  echo "==> DP4 shared startup OK: $state"
  echo "weight_ipc READY=$( [ -f "$state/weight_ipc/READY" ] && echo yes || echo no )"
  grep -h 'weight_ipc:' "$state"/logs/replica_*.log || true
  echo "==> leave servers up, or: bash $0 down $(basename "$state")"
}

case "$CMD" in
  dp2-parity) dp2_parity ;;
  dp4-startup) dp4_startup ;;
  down) bash "$LAUNCH" down "$RUN_ARG" ;;
  list) bash "$LAUNCH" list ;;
  *)
    die "usage: $0 dp2-parity|dp4-startup|down [RUN_ID]|list"
    ;;
esac
