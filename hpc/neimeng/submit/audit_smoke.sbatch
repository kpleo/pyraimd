#!/usr/bin/env bash
#SBATCH --job-name=audit_smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# E3 (red-team roadmap): audit the 11 accepted steps of smoke v4+v4.5
# (steps 31-32, 40-43, 60-64 on the 106-atom electrolyte mini box) —
# a SECOND system at a SECOND budget (eps=0.25): does r drift within those
# streaks, and does the measured drift rate transfer? SCF protocol matches
# the smoke labels exactly (ecutwfc 50 / ecutrho 400, conv_thr 1e-8,
# insulating, no smearing); each frame is an independent fresh single point
# (~3.5 min each, serial on one node).
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
AUDIT="$PROJECT/calculations/smoke_7615821/audit"

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID"

install -d -m 700 "$PROJECT/sync"
for i in $(seq 0 10); do
  echo "=== audit frame $i ($(date '+%T')) ==="
  "$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/scf_check.py" \
    --atomsXYZ "$AUDIT/accepted_frames.extxyz" \
    --frame-index "$i" \
    --pseudo-dir "$PROJECT/inputs" \
    --pw-cmd "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x" \
    --ecutwfc 50 --ecutrho 400 --conv-thr 1e-8 --nbnd-headroom 0 \
    --out-jsonl "$PROJECT/sync/audit_smoke_labels_${SLURM_JOB_ID}.jsonl" \
    --workdir "$AUDIT/work_$i"
done

echo "done"
