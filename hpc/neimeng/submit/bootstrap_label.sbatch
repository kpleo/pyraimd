#!/usr/bin/env bash
#SBATCH --job-name=bootstrap_label
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Flagship-A bootstrap, stage 2: DFT labeling of the committee-generated
# frames. Four nodes in parallel; each chains density restarts through its
# slice (first frame from scratch, then startpot). Metallic protocol (fd
# 0.02, beta 0.20, nbnd=942) per docs/hpc.md 2026-08-23.
# Submit with: sbatch --dependency=afterok:<frames_job_id> bootstrap_label.sbatch
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

FRAMES_DIR="$PROJECT/calculations/bootstrap_frames"
ALL_FRAMES="$FRAMES_DIR/all.extxyz"
cat "$FRAMES_DIR/frames_450K.extxyz" "$FRAMES_DIR/frames_300K.extxyz" > "$ALL_FRAMES"
N_FRAMES=$("$PROJECT/envs/py312/bin/python" -c "from ase.io import read; print(len(read('$ALL_FRAMES', ':')))")
PER_TASK=$(( (N_FRAMES + 3) / 4 ))
FIRST=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
LAST=$(( FIRST + PER_TASK - 1 ))
[ $LAST -ge $N_FRAMES ] && LAST=$(( N_FRAMES - 1 ))
echo "host=$(hostname) job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID frames=$FIRST..$LAST of $N_FRAMES"

RUN_DIR="$PROJECT/calculations/bootstrap_label_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
install -d -m 700 "$RUN_DIR"

USE_STARTPOT=0
for i in $(seq $FIRST $LAST); do
  EXTRA=""
  if [ "$USE_STARTPOT" -eq 1 ]; then EXTRA="--startpot"; fi
  echo "=== frame $i ($(date '+%T')) $EXTRA ==="
  if "$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/scf_check.py" \
    --atomsXYZ "$ALL_FRAMES" \
    --frame-index "$i" \
    --pseudo-dir "$PROJECT/inputs" \
    --pw-cmd "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x" \
    --metallic --smearing fd --degauss 0.02 --mixing-beta 0.20 \
    --nbnd-headroom 150 --electron-maxstep 400 \
    $EXTRA \
    --out-jsonl "$RUN_DIR/labels.jsonl" \
    --workdir "$RUN_DIR/chain"; then
    USE_STARTPOT=1
  else
    echo "FRAME $i FAILED — wiping the chain density; next frame starts from scratch"
    rm -rf "$RUN_DIR/chain/qe_runs"
    USE_STARTPOT=0
  fi
done

install -d -m 700 "$PROJECT/sync"
cp "$RUN_DIR/labels.jsonl" "$PROJECT/sync/bootstrap_labels_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
echo "done"
