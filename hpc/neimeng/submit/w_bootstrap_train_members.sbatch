#!/usr/bin/env bash
#SBATCH --job-name=w_boot_train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --array=0-1
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Flagship-B (W) bootstrap, stage 3: member-parallel pre-training. Each
# array task trains ONE backbone as a K=1 committee on the stratified
# train split (37 train / 12 held-out of 49 labels; boundaries 12,24,36
# between the bulk300/3000/6000 K and spike segments), 30 epochs,
# readout+products. Exact because finetune trains members independently;
# w_bootstrap_merge.sbatch merges afterwards. Sizing: 20.6 s/label/epoch
# at 474 atoms (probe 7616841) -> ~30 s at 686 -> ~9.3 h/member; 16
# in-process torch threads (28 OOMs the einsum workspace).
# Submit with: sbatch --dependency=afterok:<label_job_id> w_bootstrap_train_members.sbatch
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
  --labels "$PROJECT/sync/w_bootstrap_labels_*.jsonl" \
  --frames "$PROJECT/calculations/w_bootstrap_frames/all.extxyz" \
  --model "$MODEL" --n-members 1 \
  --held-out-per-segment 3 --segment-boundary "12,24,36" \
  --epochs 30 \
  --checkpoint "$PROJECT/sync/committee_w_bootstrap_m${SLURM_ARRAY_TASK_ID}.pt" \
  --report "$PROJECT/sync/w_bootstrap_train_m${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.json"

echo "done"
