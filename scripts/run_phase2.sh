#!/bin/bash
# Follow-up knowledge-embedding experiments. One GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="${TRANSOLVER_ROOT:-$ROOT/third_party/Transolver/PDE-Solving-StandardBenchmark}"
PY="${PYTHON:-python3}"
DATA="${OFFICIAL_DATA_DIR:-$ROOT/data/official}"
GPU="${GPU:-0}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$BENCH:${PYTHONPATH:-}"

if [[ ! -f "$BENCH/model_dict.py" ]]; then
  echo "Transolver benchmark not found at: $BENCH" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

mkdir -p "$ROOT/runs/official"
cd "$BENCH"

run_darcy() {
  local group="$1" epochs="$2" ntrain="${3:-1000}"
  local out="$ROOT/runs/official/darcy_${group}"
  mkdir -p "$out"
  echo "==== darcy $group ntrain=$ntrain gpu=$GPU epochs=$epochs ===="
  "$PY" "$ROOT/scripts/official/exp_darcy.py" \
    --gpu "$GPU" \
    --model Transolver_Structured_Mesh_2D \
    --n-hidden 128 --n-heads 8 --n-layers 8 \
    --lr 0.001 --max_grad_norm 0.1 \
    --batch-size 4 --slice_num 64 --unified_pos 1 --ref 8 \
    --downsample 5 --eval 0 --epochs "$epochs" --ntrain "$ntrain" \
    --group "$group" \
    --data_path "$DATA" \
    --save_name "darcy_${group}" \
    --ckpt_dir "$out" \
    --summary_json "$out/summary.json" \
    | tee "$out/train.log"
}

run_elas() {
  local group="$1" epochs="$2"
  local out="$ROOT/runs/official/elas_${group}"
  mkdir -p "$out"
  echo "==== elas $group gpu=$GPU epochs=$epochs ===="
  "$PY" "$ROOT/scripts/official/exp_elas.py" \
    --gpu "$GPU" \
    --model Transolver_Irregular_Mesh \
    --n-hidden 128 --n-heads 8 --n-layers 8 \
    --lr 0.001 --max_grad_norm 0.1 \
    --batch-size 1 --slice_num 64 --unified_pos 0 --ref 8 \
    --eval 0 --epochs "$epochs" \
    --group "$group" \
    --data_path "$DATA" \
    --save_name "elas_${group}" \
    --ckpt_dir "$out" \
    --summary_json "$out/summary.json" \
    | tee "$out/train.log"
}

# Structure / BC embedding (plan: symmetry-BC in network structure)
run_darcy HARD 500

# PDE in loss, but match discrete residual of labels (not R=1)
run_darcy MATCH 500

# Dynamic weighting on the failed strong form: conflict-control algorithm
run_darcy DYN 200

# Sparse data: where PDE knowledge can actually beat dense-label official A
run_darcy S100A 300 100
run_darcy S100M 300 100
run_darcy S100C 300 100

# Constitutive inequality on von Mises
run_elas POS 300

echo "phase2 done"
