#!/usr/bin/env bash
#SBATCH --job-name=pyraimd2_epochs
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Epoch-budget probe (post-7615920 economics): products recipe at 15/25
# epochs vs the 50-epoch reference (held-out fMax 0.360) on the 32 stored
# smoke labels — quantifies the fine-tune cost/accuracy tradeoff. One array
# task per epoch budget, parallel nodes.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export OMP_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export MKL_NUM_THREADS=28

EPOCHS=(15 25)
EPOCHS="${EPOCHS[$SLURM_ARRAY_TASK_ID]}"

install -d -m 700 "$PROJECT/sync"
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID epochs=$EPOCHS"

"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/capacity_probe.py" \
  --db "$PROJECT/calculations/smoke_7615761/loop.db" \
  --run-id smoke-7615761 \
  --model "$PROJECT/software/mace-mp-0b3-medium.model" \
  --variant products --epochs "$EPOCHS" \
  --train-size 24 \
  --test-last 8 \
  --out "$PROJECT/sync/epochs_probe_${SLURM_ARRAY_JOB_ID}.jsonl"

echo "done"
