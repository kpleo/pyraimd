#!/usr/bin/env bash
#SBATCH --job-name=audit_accepted
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Accepted-step audit, batch 1 (the decisive experiment): post-hoc DFT
# single points on the 6 accepted steps (37-42) of flagship-A segment 4.
# True max-atom error vs the stored ML prediction -> violation count at
# eps=1.0 vs alpha=0.05 (Clopper-Pearson CI). Each frame is an independent
# fresh SCF in its own workdir (no density chain across unrelated frames).
# Frozen metallic-interface protocol. %2 concurrency: leave nodes for the
# production segment running in parallel.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
AUDIT="$PROJECT/calculations/flagship_a_prod/audit"

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

i="$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_ARRAY_JOB_ID task=$i"

install -d -m 700 "$PROJECT/sync" "$AUDIT/work_$i"
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/scf_check.py" \
  --atomsXYZ "$AUDIT/accepted_frames.extxyz" \
  --frame-index "$i" \
  --pseudo-dir "$PROJECT/inputs" \
  --pw-cmd "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x" \
  --metallic --smearing fd --degauss 0.02 --mixing-beta 0.20 \
  --nbnd-headroom 150 --electron-maxstep 400 \
  --out-jsonl "$PROJECT/sync/audit_labels_${SLURM_ARRAY_JOB_ID}_${i}.jsonl" \
  --workdir "$AUDIT/work_$i"

echo "done"
