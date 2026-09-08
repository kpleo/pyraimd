#!/usr/bin/env bash
#SBATCH --job-name=bootstrap_frames
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --array=0-1
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Flagship-A bootstrap, stage 1: committee-driven frame generation on the
# 474-atom interface box. Two independent segments (task 0: 450 K seeding,
# Leung protocol; task 1: 300 K), 300 Langevin steps each, sampled every 8
# steps (~38 frames/segment, written incrementally). Frames are labeled by
# the follow-up SCF array (bootstrap_label.sbatch).
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"

TEMPS=(450 300)
TEMP=${TEMPS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="$PROJECT/calculations/bootstrap_frames"
install -d -m 700 "$OUT_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID T=${TEMP}K"

"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/committee_md.py" \
  --atomsXYZ "$PROJECT/inputs/generated/interface_li100_lp30_474.xyz" \
  --model "$PROJECT/software/mace-mp-0b3-medium.model" "$PROJECT/software/mace-mpa-0-medium.model" \
  --temperature "$TEMP" \
  --n-steps 300 \
  --sample-every 8 \
  --seed $((20250822 + SLURM_ARRAY_TASK_ID)) \
  --out "$OUT_DIR/frames_${TEMP}K.extxyz"

echo "done"
