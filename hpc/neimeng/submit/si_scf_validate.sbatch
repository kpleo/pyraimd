#!/usr/bin/env bash
#SBATCH --job-name=qe_si_validate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

# pyraimd2 QE validation: Si bulk SCF, Neimeng A-zone "normal" partition.
# Example allocation: normal = 28 CPU/node, --mem=120G, no GPU.
# Small validation job by design (provider skill: validate launcher with a
# tiny job before production shapes).

set -euo pipefail
source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/cloud_projects/pyraimd2/envs/qe"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT="$HOME/cloud_projects/pyraimd2"
RUN_DIR="$PROJECT/calculations/si_validate_${SLURM_JOB_ID}"
install -d -m 700 "$RUN_DIR"
cd "$RUN_DIR"
cp "$PROJECT/inputs/si_bulk.scf.in" .

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID"
echo "ntasks=$SLURM_NTASKS"
command -v pw.x
command -v mpirun

mpirun -np "$SLURM_NTASKS" pw.x -in si_bulk.scf.in > si_bulk.scf.out

echo "--- results ---"
grep -E "^!|total force|convergence has been achieved|convergence NOT" si_bulk.scf.out || true
echo "--- sync artifacts ---"
install -d -m 700 "$PROJECT/sync"
cp si_bulk.scf.out "$PROJECT/sync/si_bulk_scf_${SLURM_JOB_ID}.out"
echo "done"
