#!/usr/bin/env bash
#SBATCH --job-name=w_boot_label
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Flagship-B (W) bootstrap, stage 2: DFT labeling of the 49 committee
# frames (concatenation order = segment order bulk300/bulk3000/bulk6000/
# spike, so frame indices are 0-11 / 12-23 / 24-35 / 36-48). Four nodes in
# parallel, density restart chaining within each slice. W protocol from the
# ecut campaign (w_ecut_conv): mv 0.01, beta 0.3, 70/700 Ry, nbnd headroom
# +50 (semicore kjpaw: 14 valence e-/W -> nbnd 4852 at 686 atoms);
# scf_check.py pins diago_david_ndim=8 + diago_full_acc internally.
# Submit with: sbatch --dependency=afterok:<frames_job_id> w_bootstrap_label.sbatch
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

FRAMES_DIR="$PROJECT/calculations/w_bootstrap_frames"
ALL_FRAMES="$FRAMES_DIR/all.extxyz"
cat "$FRAMES_DIR/frames_bulk300K.extxyz" \
    "$FRAMES_DIR/frames_bulk3000K.extxyz" \
    "$FRAMES_DIR/frames_bulk6000K.extxyz" \
    "$FRAMES_DIR/frames_spike.extxyz" > "$ALL_FRAMES"
N_FRAMES=$("$PROJECT/envs/py312/bin/python" -c "from ase.io import read; print(len(read('$ALL_FRAMES', ':')))")
PER_TASK=$(( (N_FRAMES + 3) / 4 ))
FIRST=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
LAST=$(( FIRST + PER_TASK - 1 ))
[ $LAST -ge $N_FRAMES ] && LAST=$(( N_FRAMES - 1 ))
echo "host=$(hostname) job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID frames=$FIRST..$LAST of $N_FRAMES"

RUN_DIR="$PROJECT/calculations/w_bootstrap_label_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
    --metallic --smearing mv --degauss 0.01 --mixing-beta 0.3 \
    --nbnd-headroom 50 --electron-maxstep 200 \
    --ecutwfc 70 --ecutrho 700 --conv-thr 1e-6 \
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
cp "$RUN_DIR/labels.jsonl" "$PROJECT/sync/w_bootstrap_labels_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
echo "done"
