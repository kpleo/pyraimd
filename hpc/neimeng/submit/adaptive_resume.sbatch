#!/usr/bin/env bash
#SBATCH --job-name=pyraimd2_resume
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%j.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%j.err

# pyraimd2 M3: continuation of a previous adaptive run (--resume machinery).
# v4.5: extend smoke 7615821 by 48 steps to measure the steady-state
# acceptance rate after burn-in.  MD state + conformal window rebuild from
# the store; the committee warm-starts on the most recent --train-window
# labels (v4 predates the checkpoint hook), then checkpoints after every
# fine-tune.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
RESUME_JOB=7615821  # the run being continued

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"  # pw.x + mpirun for the QeEngine
export OMP_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export MKL_NUM_THREADS=28

RUN_DIR="$PROJECT/calculations/smoke_${RESUME_JOB}"
test -f "$RUN_DIR/loop.db"  # resume target must exist

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID resumes=$RESUME_JOB"
command -v pw.x mpirun
"$PROJECT/envs/py312/bin/python" --version

"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/adaptive_loop.py" \
  --atomsXYZ "$PROJECT/inputs/mini_electrolyte_106.xyz" \
  --cell 11.8 \
  --model "$PROJECT/software/mace-mp-0b3-medium.model" "$PROJECT/software/mace-mpa-0-medium.model" \
  --pseudo-dir "$PROJECT/inputs" \
  --pw-cmd "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x" \
  --run-id "smoke-${RESUME_JOB}" \
  --resume \
  --n-steps 48 \
  --ecutwfc 50 \
  --eps-acc 0.25 \
  --window 24 \
  --train-window 32 \
  --n-label 16 \
  --workdir "$RUN_DIR"

install -d -m 700 "$PROJECT/sync"
cp "$RUN_DIR/summary.json" "$PROJECT/sync/resume_summary_${SLURM_JOB_ID}.json"
echo "done"
