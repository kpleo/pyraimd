#!/bin/bash
#SBATCH --job-name=prepared-qe-demo   # adjust to your site
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
# Prepared-launcher invocation template: the state is valid ONLY inside
# this one allocation; a resubmission is a new allocation and must
# re-prepare.  Everything below is a placeholder for you to fill in.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. prepare ONCE per allocation into a job-private state directory
STATE="${SLURM_TMPDIR:-/tmp}/prepared-qe-${SLURM_JOB_ID}"
python3 "$HERE/launcher.py" prepare \
    --setup "$HERE/setup.sh" --state "$STATE" \
    --mpi mpirun --pw pw.x --ranks "$SLURM_NTASKS"
HASH="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["payload_sha256"])' \
    "$STATE/state.json")"

# 2. point your Pyramid configuration's reference.pw_cmd at the prepared
#    launcher, with the CONCRETE state directory and hash of this job
#    (QE's `-in` argument is appended after `--` by the engine):
#
#    pw_cmd = ["<absolute path to launcher.py>", "run",
#              "--state", "<STATE>", "--expected-sha256", "<HASH>", "--"]

# 3. run your workflow (resume keeps working: Pyramid's own identity and
#    resume checks still apply unchanged)
pyramid run run.toml

# 4. job-end cleanup removes ONLY this example's job-private state
rm -rf "$STATE"
