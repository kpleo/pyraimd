#!/usr/bin/env bash
#SBATCH --job-name=al_adaptive
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
echo "=== validate: cross-kind contract now passes ==="
pyramid validate run-md-adaptive-c1.toml | tee validate-c1.txt
echo "=== adaptive 4 steps ==="
pyramid run run-md-adaptive-c1.toml
echo "=== resume +2 (fresh process) ==="
pyramid resume runs/al-adaptive-c1 --steps 2
echo "=== inspect + export ==="
pyramid inspect runs/al-adaptive-c1 --json | tee inspect-c1.json
pyramid export runs/al-adaptive-c1 --force-source driving
echo "=== actual SCF count from the attempt ledger ==="
python - <<'PY'
import json
events = [json.loads(line) for line in
          open("runs/al-adaptive-c1/events.jsonl")]
attempts = [e for e in events
            if e.get("type") == "attempt" and e.get("record") == "physical_attempt"]
tasks = [e for e in events if e.get("type") == "task"
         and e.get("operation") == "reference"]
print(json.dumps({
    "logical_reference_requests": len(tasks),
    "physical_attempts": len(attempts),
    "failed_attempts": sum(1 for e in attempts if e["status"] == "failed"),
    "cache_hits": sum(1 for e in tasks if e.get("status") == "cache_hit"),
}, indent=2))
PY
