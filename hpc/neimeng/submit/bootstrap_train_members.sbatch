#!/usr/bin/env bash
#SBATCH --job-name=boot_train_member
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=11:00:00
#SBATCH --array=0-1
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Flagship-A bootstrap, stage 3 (member-parallel resubmission after the
# 7616839 OOM): each array task trains ONE backbone as a K=1 committee on
# the 52 train labels (30 epochs, readout+products) — exact, because
# finetune trains members independently. Probe 7616841 sizing: 20.6
# s/label/epoch/member at 16 threads (VmHWM 41.5 GB) -> ~8.9 h per member;
# 28 threads OOMs (29 GB einsum workspace), 16 threads proven safe.
# bootstrap_merge.py merges the two K=1 checkpoints afterwards.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

MODELS=("$PROJECT/software/mace-mp-0b3-medium.model" "$PROJECT/software/mace-mpa-0-medium.model")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

install -d -m 700 "$PROJECT/sync"
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID model=$MODEL"

cd "$PROJECT"
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/bootstrap_train.py" \
  --labels "$PROJECT/sync/bootstrap_labels_7616227_*.jsonl" \
  --frames "$PROJECT/calculations/bootstrap_frames/all.extxyz" \
  --model "$MODEL" --n-members 1 \
  --held-out-per-segment 4 \
  --epochs 30 \
  --checkpoint "$PROJECT/sync/committee_bootstrap_m${SLURM_ARRAY_TASK_ID}.pt" \
  --report "$PROJECT/sync/bootstrap_train_m${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.json"

echo "done"
