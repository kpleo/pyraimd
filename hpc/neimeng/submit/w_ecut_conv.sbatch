#!/usr/bin/env bash
#SBATCH --job-name=qe_w_ecut_conv
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# pyraimd2 M3 (flagship B): ecut convergence series for the bcc W bulk box
# (128 atoms, Gamma, PBE, mv smearing). One array element per ecutwfc value.
# %2: shares the 4-node account cap with the electrolyte campaigns.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU (docs/hpc.md).

set -euo pipefail
ECUTS=(40 50 60 70 80 90)
ECUT=${ECUTS[$SLURM_ARRAY_TASK_ID]}
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/qe"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_DIR="$PROJECT/calculations/w_ecut_conv/ecut_$ECUT"
install -d -m 700 "$RUN_DIR"
cd "$RUN_DIR"
cp "$PROJECT/inputs/generated/w_ecut_$ECUT/pw.in" pw.in

echo "host=$(hostname) date=$(date '+%F %T %Z') job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ecut=$ECUT ntasks=$SLURM_NTASKS"
t0=$(date +%s)
mpirun -np "$SLURM_NTASKS" pw.x -in pw.in > pw.out
t1=$(date +%s)

ENERGY=$(grep -E "^!" pw.out | tail -1 | awk '{print $5}')
CONV=$(grep -c "convergence has been achieved" pw.out || true)
echo "RESULT ecut=$ECUT energy_ry=$ENERGY converged=$CONV wall_s=$((t1 - t0))" | tee "$PROJECT/outputs/w_ecut_conv_${ECUT}.txt"
