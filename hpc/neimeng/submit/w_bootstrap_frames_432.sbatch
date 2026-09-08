#!/usr/bin/env bash
#SBATCH --job-name=w_boot_frames432
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Flagship-B (W) bootstrap on the 432-atom fallback cell (6^3; design-m3 v2
# fallback clause, triggered by the 686 label RAM wall: ~815 GB total
# footprint, weeks-to-months of label time for the Act-2 statistics).
# Same four condition segments as the 686 attempt:
#   task 0: bulk 300 K    Langevin, 220 steps, sample 20 -> 12 frames
#   task 1: bulk 3000 K   Langevin (superheated solid), same
#   task 2: bulk 6000 K   Langevin (liquid), same
#   task 3: spike (two-temperature init from w_spike_432.xyz), NVE
#           (--friction 0 --keep-velocities), 600 steps, sample 50 -> 13
# The 7^3 = 686 cell survives as the committee-only top end of the
# finite-size self-proof series (design-m3 self-proof b).
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"

INPUTS=("$PROJECT/inputs/generated/w_bulk_432.xyz" "$PROJECT/inputs/generated/w_bulk_432.xyz" "$PROJECT/inputs/generated/w_bulk_432.xyz" "$PROJECT/inputs/generated/w_spike_432.xyz")
TEMPS=(300 3000 6000 300)
NSTEPS=(220 220 220 600)
SAMPLE=(20 20 20 50)
NAMES=(bulk300K bulk3000K bulk6000K spike)

IN_XYZ="${INPUTS[$SLURM_ARRAY_TASK_ID]}"
TEMP="${TEMPS[$SLURM_ARRAY_TASK_ID]}"
NST="${NSTEPS[$SLURM_ARRAY_TASK_ID]}"
SE="${SAMPLE[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"

EXTRA=()
if [ "$NAME" = "spike" ]; then
  EXTRA=(--friction 0 --keep-velocities)
fi

OUT_DIR="$PROJECT/calculations/w_bootstrap_frames_432"
install -d -m 700 "$OUT_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID seg=$NAME T=${TEMP}K nsteps=$NST sample=$SE extra=${EXTRA[*]:-none}"

"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/committee_md.py" \
  --atomsXYZ "$IN_XYZ" \
  --model "$PROJECT/software/mace-mp-0b3-medium.model" "$PROJECT/software/mace-mpa-0-medium.model" \
  --temperature "$TEMP" \
  --n-steps "$NST" \
  --sample-every "$SE" \
  --seed $((20260831 + SLURM_ARRAY_TASK_ID)) \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  --out "$OUT_DIR/frames_${NAME}.extxyz"

echo "done"
