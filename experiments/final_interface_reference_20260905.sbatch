#!/bin/bash
#SBATCH --job-name=pyraimd_final_ref
#SBATCH --partition=9242
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --time=03:10:00
#SBATCH --array=0,9%2
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/final_campaign_20260905/logs/reference_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/final_campaign_20260905/logs/reference_%A_%a.err
set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
TASK="$PROJECT/experiments/final_campaign_20260905"
RUNTIME="$PROJECT/experiments/forward_h2o_20260905/envs/cloud_hot_20260905"
export LC_ALL=C PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=never
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PATH="$PROJECT/envs/qe/bin:$PATH"
export PYTHONPATH="$TASK/workspace/src:$RUNTIME"
cd "$TASK/workspace"
date -Is
hostname
scontrol show job "$SLURM_JOB_ID"
"$RUNTIME/bin/uv" run --no-project --python "$PROJECT/envs/py312/bin/python" \
    python experiments/final_interface_reference_20260905.py
date -Is
