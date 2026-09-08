#!/bin/bash
#SBATCH --job-name=pyraimd_feasibility
#SBATCH --partition=9242
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/forward_h2o_20260905/logs/feasibility_%j.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/forward_h2o_20260905/logs/feasibility_%j.err
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
# Twelve independent one-core trajectories share the allocation; the same
# pre-trained weights and fixed rules are used throughout all 48 tasks.
srun --ntasks=1 --cpus-per-task=12 "$RUNTIME/bin/uv" run --no-project \
    --python "$PROJECT/envs/py312/bin/python" python experiments/cloud_feasibility_20260905.py
for LEVEL in kinetic_300 kinetic_600 kinetic_1200; do
    "$RUNTIME/bin/uv" run --no-project --python "$PROJECT/envs/py312/bin/python" \
        python experiments/analyze_forward_pilot_20260905.py \
        --directory "analysis/execution_20260905/h2o_feasibility/$LEVEL"
done
date -Is
