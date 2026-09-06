#!/usr/bin/env bash
#SBATCH --job-name=pyramid_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=pyramid_%j.out
#SBATCH --error=pyramid_%j.err

# Submit from the repository root after installing .[dev,pyscf]. Select the
# partition/account and memory policy required by your own cluster at submission.
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
python -m pytest tests/unit tests/test_smoke.py -q
python examples/energetic_loop.py
python examples/energetic_pyscf.py
