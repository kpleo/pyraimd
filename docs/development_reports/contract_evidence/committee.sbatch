#!/usr/bin/env bash
#SBATCH --job-name=committee_persist
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/contract_al/logs/slurm_%x_%j.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/contract_al/logs/slurm_%x_%j.err

set -euo pipefail
source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/cloud_projects/pyraimd2/envs/py312"
export PYTHONPATH="$HOME/cloud_projects/pyraimd2/software/pyraimd2-dev/src"
export PATH="$HOME/cloud_projects/pyraimd2/envs/qe/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

E="$HOME/cloud_projects/pyraimd2/experiments/contract_al"
cd "$E"
mkdir -p /tmp/ps && cp $HOME/cloud_projects/pyraimd2/inputs/*.UPF /tmp/ps/
echo "=== process 1: 2 labels -> 1 update -> tensor artifact ==="
python committee_persist_al.py "$E" process1
echo "=== process 2: fresh process resume and fidelity checks ==="
python committee_persist_al.py "$E" process2
