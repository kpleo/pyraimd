#!/usr/bin/env bash
#SBATCH --job-name=w_boot_merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# Flagship-B (W) bootstrap, stage 3b: merge the member-parallel K=1
# checkpoints into the production K=2 mixed checkpoint + held-out gate on
# the ensemble mean (go/no-go: fMax <= ~0.35 eV/A, same bar as flagship A;
# held-out covers all four condition segments). Submit with
#   sbatch --dependency=afterok:<train_array_jobid> w_bootstrap_merge.sbatch
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID"

cd "$PROJECT"
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/bootstrap_merge.py" \
  --member-checkpoint "$PROJECT/sync/committee_w_bootstrap_m0.pt" \
                      "$PROJECT/sync/committee_w_bootstrap_m1.pt" \
  --model "$PROJECT/software/mace-mp-0b3-medium.model" \
          "$PROJECT/software/mace-mpa-0-medium.model" \
  --labels "$PROJECT/sync/w_bootstrap_labels_*.jsonl" \
  --frames "$PROJECT/calculations/w_bootstrap_frames/all.extxyz" \
  --held-out-per-segment 3 --segment-boundary "12,24,36" \
  --checkpoint "$PROJECT/sync/committee_w_bootstrap.pt" \
  --report "$PROJECT/sync/w_bootstrap_merge_${SLURM_JOB_ID}.json"

echo "done"
