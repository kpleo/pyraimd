#!/bin/bash
# Submit the already frozen experiment exactly once during the 12-hour handoff window.
set -euo pipefail
export LC_ALL=C
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
TASK="$PROJECT/experiments/final_campaign_20260905"
DATA="$TASK/workspace/analysis/final_campaign_20260905/interface"
RUNTIME="$PROJECT/experiments/forward_h2o_20260905/envs/cloud_hot_20260905"
export FUTURE_PROTOCOL_SHA256=01ab35a2d5efe3d2607231cfe8a8930d159322d5e35b3ad3fdfdc92f61ba1e76
test "$(date +%s)" -lt 1788666946
cd "$TASK/workspace"
"$RUNTIME/bin/uv" run --no-project --python "$PROJECT/envs/py312/bin/python" python - <<'PY'
import hashlib,json,runpy
from pathlib import Path
data=Path('analysis/final_campaign_20260905/interface')
expected='01ab35a2d5efe3d2607231cfe8a8930d159322d5e35b3ad3fdfdc92f61ba1e76'
assert hashlib.sha256((data/'future_protocol.json').read_bytes()).hexdigest()==expected
helper=runpy.run_path('experiments/final_interface_future_reference_20260905.py')
for name in ('future_protocol.json','check_protocol.json'):
    p=json.loads((data/name).read_text())
    assert p['status']=='frozen'
    helper['verify_hashes'](p['source_files_sha256'],Path('.'))
    helper['verify_hashes'](p['input_sha256'],data)
    helper['verify_hashes'](p['runtime_files_sha256'])
    helper['validate_stopping_policy'](p,json.loads((data/'goal_based_closure.json').read_text()))
assert len(list((data/'budget_slots').glob('*.json')))==18
for name in ('future_reference_results','check_reference_results'):
    assert not (data/name).exists() or not any((data/name).iterdir())
print('Frozen science, runtime and all 18 consumed development slots verified.',flush=True)
PY

RELEASE="$DATA/release_20260906"
mkdir "$RELEASE"
date -Is > "$RELEASE/started_at.txt"
cp experiments/final_interface_release_20260906.sh "$RELEASE/submission_script.sh"
# A response failure leaves the release lock and any scheduler response in place.
# Inspect these records and Slurm; never rerun this script or delete its markers.
sbatch --parsable --no-requeue --array=5,25%2 \
  experiments/final_interface_future_reference_20260905.sbatch > "$RELEASE/priority.jobid"
PRIORITY=$(cat "$RELEASE/priority.jobid")
[[ "$PRIORITY" =~ ^[0-9]+$ ]]
sbatch --parsable --array=0-5%1 \
  experiments/final_interface_check_reference_20260906.sbatch > "$RELEASE/checks.jobid"
CHECKS=$(cat "$RELEASE/checks.jobid")
[[ "$CHECKS" =~ ^[0-9]+$ ]]
sbatch --parsable --no-requeue --dependency="afterok:$PRIORITY" --array=0-4,6-24,26-39%2 \
  experiments/final_interface_future_reference_20260905.sbatch > "$RELEASE/remaining.jobid"
REMAINING=$(cat "$RELEASE/remaining.jobid")
[[ "$REMAINING" =~ ^[0-9]+$ ]]
date -Is > "$RELEASE/submitted_at.txt"
printf 'priority=%s\nchecks=%s\nremaining=%s\n' "$PRIORITY" "$CHECKS" "$REMAINING"
squeue -u df103967 -o '%.18i %.9P %.26j %.8T %.12M %.5C %.20R'
