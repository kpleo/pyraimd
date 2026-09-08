#!/usr/bin/env bash
#SBATCH --job-name=iface_scf_check
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%j.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%j.err

# Flagship-A prerequisite: single-point SCF protocol validation on the
# 474-atom Li(100)|LP30 interface box (metallic slab + liquid, Gamma).
# Full §3.1 protocol: 60/600 Ry, mv smearing 0.01, nbnd=nelec/2+60,
# local-TF mixing ndim=12, Davidson ndim=8 full-acc, conv_thr=1e-6.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

RUN_DIR="$PROJECT/calculations/iface_scf_check_${SLURM_JOB_ID}"
install -d -m 700 "$RUN_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID"
command -v pw.x mpirun

"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/scf_check.py" \
  --atomsXYZ "$PROJECT/inputs/generated/interface_li100_lp30_474.xyz" \
  --pseudo-dir "$PROJECT/inputs" \
  --pw-cmd "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x" \
  --metallic \
  --workdir "$RUN_DIR"

echo "done"
