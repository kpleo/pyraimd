#!/usr/bin/env bash
#SBATCH --job-name=w_boot_label432
#SBATCH --partition=9242
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%4
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Flagship-B (W) bootstrap, stage 2, 432-cell: DFT labeling of the 49
# committee frames (order bulk300/bulk3000/bulk6000/spike -> indices
# 0-11/12-23/24-35/36-48). Shape: ONE 9242 node per task (96 ranks,
# -ndiag 24) — probe A measured the same iteration rate as 192 ranks, so
# single-node tasks double throughput; %4 = four concurrent tasks (the
# 4-node cap). --timeout-s 21600: disordered frames need >24 iterations
# (the 3h default was killing converging tails; frame 1 died twice at
# exactly 3h). Resumable: stable per-task output + skip-if-labeled +
# stable startpot chain dirs.
# W protocol per the ecut campaign: mv 0.01, beta 0.3, 70/700 Ry, nbnd
# headroom +50 (nbnd 3074 at 432 atoms); scf_check pins david 8 + full_acc.
# Neimeng A "9242" contract: 96 CPU/node, --mem=360G, 4-node cap.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

ALL_FRAMES="$PROJECT/calculations/w_bootstrap_frames_432/all.extxyz"
N_FRAMES=$("$PROJECT/envs/py312/bin/python" -c "from ase.io import read; print(len(read('$ALL_FRAMES', ':')))")
PER_TASK=$(( (N_FRAMES + 5) / 6 ))
FIRST=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
LAST=$(( FIRST + PER_TASK - 1 ))
[ $LAST -ge $N_FRAMES ] && LAST=$(( N_FRAMES - 1 ))

install -d -m 700 "$PROJECT/sync"
LABELS_OUT="$PROJECT/sync/w_bootstrap_labels_432_t${SLURM_ARRAY_TASK_ID}.jsonl"
touch "$LABELS_OUT"
RUN_DIR="$PROJECT/calculations/w_bootstrap_label_432/task_${SLURM_ARRAY_TASK_ID}"
install -d -m 700 "$RUN_DIR"

echo "host=$(hostname) job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID frames=$FIRST..$LAST of $N_FRAMES"

PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 96 pw.x -ndiag 24"
USE_STARTPOT=0
for i in $(seq $FIRST $LAST); do
  if grep -q "\"frame_index\": $i," "$LABELS_OUT"; then
    echo "=== frame $i already labeled — skip ==="
    USE_STARTPOT=1
    continue
  fi
  EXTRA=""
  if [ "$USE_STARTPOT" -eq 1 ]; then EXTRA="--startpot"; fi
  echo "=== frame $i ($(date '+%T')) $EXTRA ==="
  if "$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/scf_check.py" \
    --atomsXYZ "$ALL_FRAMES" \
    --frame-index "$i" \
    --pseudo-dir "$PROJECT/inputs" \
    --pw-cmd "$PW" \
    --metallic --smearing mv --degauss 0.01 --mixing-beta 0.3 \
    --nbnd-headroom 50 --electron-maxstep 200 \
    --ecutwfc 70 --ecutrho 700 --conv-thr 1e-6 \
    $EXTRA \
    --out-jsonl "$LABELS_OUT" \
    --workdir "$RUN_DIR/chain"; then
    USE_STARTPOT=1
  else
    echo "FRAME $i FAILED — wiping the chain density; next frame starts from scratch"
    rm -rf "$RUN_DIR/chain/qe_runs"
    USE_STARTPOT=0
  fi
done

echo "done"
