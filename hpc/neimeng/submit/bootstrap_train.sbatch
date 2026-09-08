#!/usr/bin/env bash
#SBATCH --job-name=bootstrap_train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# Flagship-A bootstrap, stage 3: offline committee pre-training on the 60
# SCF labels (52 train / 8 temperature-stratified held-out), production
# recipe — K=2 mixed [0b3, MPA-0], readout+products, 30 epochs. Produces
# the warm-start checkpoint for the production closed loop and the
# held-out fMAE/fMax gate (go/no-go: held-out fMax <= ~0.35 eV/A).
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export OMP_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export MKL_NUM_THREADS=28

install -d -m 700 "$PROJECT/sync"
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID"

cd "$PROJECT"
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/bootstrap_train.py" \
  --labels "$PROJECT/sync/bootstrap_labels_7616227_*.jsonl" \
  --frames "$PROJECT/calculations/bootstrap_frames/all.extxyz" \
  --model "$PROJECT/software/mace-mp-0b3-medium.model" \
          "$PROJECT/software/mace-mpa-0-medium.model" \
  --held-out-per-segment 4 \
  --epochs 30 \
  --checkpoint "$PROJECT/sync/committee_bootstrap.pt" \
  --report "$PROJECT/sync/bootstrap_train_${SLURM_JOB_ID}.json"

echo "done"
