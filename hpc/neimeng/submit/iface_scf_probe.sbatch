#!/usr/bin/env bash
#SBATCH --job-name=iface_scf_probe
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# Interface-box SCF protocol probe (7616078 oscillated unconverged: accuracy
# bouncing 0.018-0.26 Ry over 191 iterations — metal-surface Fermi-level
# response). Four metallic-protocol candidates in parallel, one per node:
#   t0: mv  degauss=0.02  beta=0.20  headroom=150
#   t1: mv  degauss=0.03  beta=0.15  headroom=150
#   t2: fd  degauss=0.02  beta=0.20  headroom=250
#   t3: mv  degauss=0.01  beta=0.10  headroom=150  (low-beta control)
# electron_maxstep=400; engine timeout 10800 s inside a 4 h wall.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

SMEARING=(mv mv fd mv)
DEGAUSS=(0.02 0.03 0.02 0.01)
BETA=(0.20 0.15 0.20 0.10)
HEADROOM=(150 150 250 150)

RUN_DIR="$PROJECT/calculations/iface_scf_probe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
install -d -m 700 "$RUN_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID"
echo "smearing=${SMEARING[$SLURM_ARRAY_TASK_ID]} degauss=${DEGAUSS[$SLURM_ARRAY_TASK_ID]} beta=${BETA[$SLURM_ARRAY_TASK_ID]} headroom=${HEADROOM[$SLURM_ARRAY_TASK_ID]}"

"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/scf_check.py" \
  --atomsXYZ "$PROJECT/inputs/generated/interface_li100_lp30_474.xyz" \
  --pseudo-dir "$PROJECT/inputs" \
  --pw-cmd "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x" \
  --metallic \
  --smearing "${SMEARING[$SLURM_ARRAY_TASK_ID]}" \
  --degauss "${DEGAUSS[$SLURM_ARRAY_TASK_ID]}" \
  --mixing-beta "${BETA[$SLURM_ARRAY_TASK_ID]}" \
  --nbnd-headroom "${HEADROOM[$SLURM_ARRAY_TASK_ID]}" \
  --electron-maxstep 400 \
  --workdir "$RUN_DIR"

echo "done"
