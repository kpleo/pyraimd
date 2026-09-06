#!/usr/bin/env bash
#SBATCH --job-name=pyraimd2_smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

# Pyramid: remote adaptive-loop smoke test on the 106-atom mini box.
# Whole node for one python driver; QE labels run via mpirun inside the same
# allocation (sequential phases: torch predicts, pw.x labels).
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.
# v3: mixed-backbone committee (0b3 + MPA-0, K=2) with the RMS spread fix —
# cross-backbone r ~ 2.2 vs ~65 same-backbone; acceptance should become
# reachable around step ~20-30 if post-fine-tune errors land near 0.2 eV/A.

set -euo pipefail
PROJECT="${PYRAMID_PROJECT:?Set PYRAMID_PROJECT to the remote project directory}"

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"  # pw.x + mpirun for the QeEngine
export OMP_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export MKL_NUM_THREADS=28

RUN_DIR="$PROJECT/calculations/smoke_${SLURM_JOB_ID}"
install -d -m 700 "$RUN_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID"
command -v pw.x mpirun
"$PROJECT/envs/py312/bin/python" --version

"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/adaptive_loop.py" \
  --atomsXYZ "$PROJECT/inputs/mini_electrolyte_106.xyz" \
  --cell 11.8 \
  --model "$PROJECT/software/mace-mp-0b3-medium.model" "$PROJECT/software/mace-mpa-0-medium.model" \
  --pseudo-dir "$PROJECT/inputs" \
  --pw-cmd "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x" \
  --run-id "smoke-${SLURM_JOB_ID}" \
  --n-steps 32 \
  --ecutwfc 50 \
  --eps-acc 0.25 \
  --window 24 \
  --workdir "$RUN_DIR"

install -d -m 700 "$PROJECT/sync"
cp "$RUN_DIR/summary.json" "$PROJECT/sync/smoke_summary_${SLURM_JOB_ID}.json"
echo "done"
