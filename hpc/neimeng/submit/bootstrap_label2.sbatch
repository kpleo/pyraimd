#!/usr/bin/env bash
#SBATCH --job-name=bootstrap_label2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Flagship-A bootstrap, stage 2 continuation: finish the frame slices to 60
# total labels (32 done in 7616227, +7 frames per task here). Reuses the
# prior tasks' chain directories, so each task resumes startpot from its
# last converged density.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
PRIOR=7616227

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

FIRSTS=(8 27 46 65)
LASTS=(14 33 52 71)
FIRST=${FIRSTS[$SLURM_ARRAY_TASK_ID]}
LAST=${LASTS[$SLURM_ARRAY_TASK_ID]}

ALL_FRAMES="$PROJECT/calculations/bootstrap_frames/all.extxyz"
RUN_DIR="$PROJECT/calculations/bootstrap_label_${PRIOR}_${SLURM_ARRAY_TASK_ID}"
test -d "$RUN_DIR/chain"  # prior chain density must exist
echo "host=$(hostname) job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID frames=$FIRST..$LAST (continuation of $PRIOR)"

USE_STARTPOT=1  # resume from the prior task's converged density
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
cp "$RUN_DIR/labels.jsonl" "$PROJECT/sync/bootstrap_labels_${PRIOR}_${SLURM_ARRAY_TASK_ID}.jsonl"
echo "done"
