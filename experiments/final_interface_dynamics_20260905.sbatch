#!/bin/bash
#SBATCH --job-name=pyraimd_final_md
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --array=0-3%2
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/final_campaign_20260905/logs/dynamics_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/final_campaign_20260905/logs/dynamics_%A_%a.err
set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
TASK="$PROJECT/experiments/final_campaign_20260905"
RUNTIME="$PROJECT/experiments/forward_h2o_20260905/envs/cloud_hot_20260905"
export LC_ALL=C PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=never
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=12 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$TASK/workspace/src:$RUNTIME"
cd "$TASK/workspace"
date -Is
hostname
scontrol show job "$SLURM_JOB_ID"
"$RUNTIME/bin/uv" run --no-project --python "$PROJECT/envs/py312/bin/python" \
    python experiments/final_interface_dynamics_20260905.py
date -Is
