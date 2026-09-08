#!/bin/bash
#SBATCH --job-name=pyraimd_hot_forward
#SBATCH --partition=9242
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/forward_h2o_20260905/logs/hot_forward_%j.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/forward_h2o_20260905/logs/hot_forward_%j.err

set -euo pipefail
TASK=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/forward_h2o_20260905
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
RUNTIME="$TASK/envs/cloud_hot_20260905"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=never LC_ALL=C
export H2O_MODEL_PATH="$TASK/models/20231210mace128L0_energy_epoch249model"
export PYTHONPATH="$RUNTIME:$TASK/workspace/src"
cd "$TASK/workspace"
date -Is
hostname
scontrol show job "$SLURM_JOB_ID"
# Small H2O inference/SCF is serial; four independent arms use the four cores.
# No GPU, retraining, MPI, or exclusive whole-node allocation is requested.
srun --ntasks=1 --cpus-per-task=4 "$RUNTIME/bin/uv" run --no-project \
    --python "$PROJECT/envs/py312/bin/python" python experiments/cloud_hot_forward_20260905.py
"$RUNTIME/bin/uv" run --no-project --python "$PROJECT/envs/py312/bin/python" \
    python experiments/analyze_forward_pilot_20260905.py \
    --directory analysis/execution_20260905/h2o_hot_cloud
date -Is
