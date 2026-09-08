#!/bin/bash
#SBATCH --job-name=pyraimd_final_gate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --dependency=afterany:7623587
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/final_campaign_20260905/logs/gate_%j.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/final_campaign_20260905/logs/gate_%j.err
set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
TASK="$PROJECT/experiments/final_campaign_20260905"
RUNTIME="$PROJECT/experiments/forward_h2o_20260905/envs/cloud_hot_20260905"
export LC_ALL=C PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=never
cd "$TASK/workspace"
"$RUNTIME/bin/uv" run --no-project --python "$PROJECT/envs/py312/bin/python" \
    python experiments/final_interface_release_probes_20260905.py
