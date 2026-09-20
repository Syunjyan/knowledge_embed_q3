#!/bin/bash
# Official Transolver cases, one GPU, sequential.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="${TRANSOLVER_ROOT:-$ROOT/third_party/Transolver/PDE-Solving-StandardBenchmark}"
PY="${PYTHON:-python3}"
DATA="${OFFICIAL_DATA_DIR:-$ROOT/data/official}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-500}"

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

wait_file() {
  local f="$1"
  echo "waiting for $f"
  while [[ ! -s "$f" ]]; do
    sleep 15
  done
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

run_darcy() {
  local group="$1" epochs="$2"
  local out="$ROOT/runs/official/darcy_${group}"
  mkdir -p "$out"
  echo "==== darcy $group gpu=$GPU epochs=$epochs ===="
  "$PY" "$ROOT/scripts/official/exp_darcy.py" \
    --gpu "$GPU" \
    --model Transolver_Structured_Mesh_2D \
    --n-hidden 128 --n-heads 8 --n-layers 8 \
    --lr 0.001 --max_grad_norm 0.1 \
    --batch-size 4 --slice_num 64 --unified_pos 1 --ref 8 \
    --downsample 5 --eval 0 --epochs "$epochs" \
    --group "$group" \
    --data_path "$DATA" \
    --save_name "darcy_${group}" \
    --ckpt_dir "$out" \
    --summary_json "$out/summary.json" \
    | tee "$out/train.log"
}

run_ns() {
  local group="$1" epochs="$2"
  local out="$ROOT/runs/official/ns_${group}"
  mkdir -p "$out"
  echo "==== ns $group gpu=$GPU epochs=$epochs ===="
  "$PY" "$ROOT/scripts/official/exp_ns.py" \
    --gpu "$GPU" \
    --model Transolver_Structured_Mesh_2D \
    --n-hidden 256 --n-heads 8 --n-layers 8 \
    --lr 0.001 \
    --batch-size 2 --slice_num 32 --unified_pos 1 --ref 8 \
    --eval 0 --epochs "$epochs" \
    --group "$group" \
    --data_path "$DATA" \
    --save_name "ns_${group}" \
    --ckpt_dir "$out" \
    --summary_json "$out/summary.json" \
    | tee "$out/train.log"
}

run_plas() {
  local group="$1" epochs="$2"
  local out="$ROOT/runs/official/plas_${group}"
  mkdir -p "$out"
  echo "==== plas $group gpu=$GPU epochs=$epochs ===="
  "$PY" "$ROOT/scripts/official/exp_plas.py" \
    --gpu "$GPU" \
    --model Transolver_Structured_Mesh_2D \
    --n-hidden 128 --n-heads 8 --n-layers 8 \
    --lr 0.001 --max_grad_norm 0.1 \
    --batch-size 8 --slice_num 64 --unified_pos 0 --ref 8 \
    --eval 0 --epochs "$epochs" \
    --group "$group" \
    --data_path "$DATA/plas_N987_T20.mat" \
    --save_name "plas_${group}" \
    --ckpt_dir "$out" \
    --summary_json "$out/summary.json" \
    | tee "$out/train.log"
}

JOBS="${JOBS:-all}"

wait_file "$DATA/elasticity/Meshes/Random_UnitCell_sigma_10.npy"
wait_file "$DATA/elasticity/Meshes/Random_UnitCell_XY_10.npy"
run_elas A "$EPOCHS"
if [[ "$JOBS" == "smoke" ]]; then
  echo "smoke elas A done"
  exit 0
fi

wait_file "$DATA/piececonst_r421_N1024_smooth1.mat"
wait_file "$DATA/piececonst_r421_N1024_smooth2.mat"
run_darcy A "$EPOCHS"
run_darcy C "$EPOCHS"
run_darcy D "$EPOCHS"

wait_file "$DATA/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"
run_ns A "$EPOCHS"
run_ns C "$EPOCHS"
run_ns D "$EPOCHS"

wait_file "$DATA/plas_N987_T20.mat"
run_plas A "$EPOCHS"

echo "official sequential done"
