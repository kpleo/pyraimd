#!/bin/sh
# Public end-to-end demo of the persistent QE density chain on a FAKE
# pw.x: it verifies the program flow only — no real DFT is produced.
#
#   3-step reference MD with [density] persist -> fresh-process resume
#   (+2 steps) -> delayed producer release receipts -> generation
#   tombstones -> dry-run and actual reclaim of old generations
#
# Usage (from the repository root, with the package installed):
#   sh examples/density_persist_qe/demo_fake.sh [WORK_DIR]
set -eu

WORK=${1:-demo-density-work}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
FIXTURE="$ROOT/tests/data/qe_si_scf.out"
PY=${PYTHON:-python3}
export WORK

# Never delete user data: a fresh directory is created for the demo; an
# existing path is refused unless it is an empty directory.
if [ -e "$WORK" ]; then
    if [ -d "$WORK" ] && [ -z "$(ls -A "$WORK")" ]; then
        echo "using the existing empty directory $WORK"
    else
        echo "refusing to use the existing path $WORK (it is not an empty directory); nothing was touched" >&2
        exit 2
    fi
fi
mkdir -p "$WORK/pseudos"
# the fake pw.x never reads the pseudopotential, but path validation does
# require the file to exist
touch "$WORK/pseudos/Si.pbe-n-kjpaw_psl.1.0.0.UPF"
WORK_ABS=$(CDPATH= cd -- "$WORK" && pwd)
export WORK_ABS

cat > "$WORK_ABS/fake_pwx.sh" <<EOF
#!/bin/bash
# a genuine warm start: the staged input density exists BEFORE this run
# writes its own products — report the read exactly like QE 7.5 does (an
# atomic start has no staged file and prints no read marker)
seed=tmp/pyraimd2.save
if test -f "\$seed/charge-density.dat" || test -f "\$seed/charge-density.hdf5"; then
  echo '     The initial density is read from file :'
  echo '     ./tmp/pyraimd2.save/charge-density'
fi
mkdir -p tmp/pyraimd2.save
echo fake-density > tmp/pyraimd2.save/charge-density.dat
echo '<xml/>' > tmp/pyraimd2.save/data-file-schema.xml
cat "$FIXTURE"
EOF
chmod +x "$WORK_ABS/fake_pwx.sh"

"$PY" - <<'EOF'
import os
from ase import Atoms
from ase.io import write
write(os.path.join(os.environ["WORK"], "structure.extxyz"),
      Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
            cell=[5.43] * 3, pbc=True))
EOF

cp "$HERE/run.toml" "$WORK_ABS/run.toml"
cd "$WORK_ABS"
# point the toy configuration at the fake pw.x (absolute path: the engine
# launches from per-attempt working directories)
"$PY" - <<'EOF'
from pathlib import Path
path = Path("run.toml")
path.write_text(path.read_text().replace(
    'pw_cmd = ["pw.x"]', f'pw_cmd = ["{__import__("os").environ["WORK_ABS"]}/fake_pwx.sh"]'))
EOF

echo "== run 3 steps (publish -> warm start -> delayed release -> reclaim) =="
"$PY" -c 'from pyraimd2.config import load_config
from pyraimd2.workflows import run_workflow
run_workflow(load_config("run.toml"), verbose=True, handle_sigint=False)'

echo "== resume +2 steps in a FRESH process =="
"$PY" -c 'from pyraimd2.workflows import resume_workflow
resume_workflow("run", 2, verbose=True, handle_sigint=False)'

echo "== registry state (latest / attach history / reclaim tombstones) =="
"$PY" - <<'EOF'
import json
from pyraimd2.runtime.restart import inspect_density_registry
view = inspect_density_registry("run")
print(json.dumps({"state": view["state"], "latest": view["latest"],
                  "attach_history": view["attach_history"],
                  "reclaimed": view["reclaimed"],
                  "on_disk": [r["name"] for r in view["resources"]
                              if r["kind"] == "generation"]},
                 indent=1))
EOF

echo "== scratch lifecycle and consumption receipts =="
"$PY" - <<'EOF'
import json
from pathlib import Path
records = [json.loads(p.read_text())
           for p in sorted(Path("run/scratch_records").rglob("*.json"))]
for r in records:
    ev = r.get("released_evidence") or {}
    print(r["request_id"], r["attempt_id"], "->", r["state"],
          "| proof:", ev.get("proof"), "| out gen:",
          ev.get("density_output_generation"), "| consumed by eval:",
          (ev.get("consumed_by") or {}).get("evaluation_id"))
EOF

echo "== dry-run the old-generation reclaim (nothing is deleted) =="
"$PY" - <<'EOF'
from pyraimd2.runtime.restart import plan_density_reclaim
plan = plan_density_reclaim("run")
for r in plan["resources"]:
    print(f'{r["name"]:<28} {r.get("decision"):<18} {"; ".join(r.get("reasons", []))}')
print(plan["space_bound_note"])
EOF

echo "== execute the reclaim against the fresh plan, then repeat (no-op) =="
"$PY" - <<'EOF'
from pyraimd2.runtime.restart import (plan_density_reclaim,
                                      execute_density_reclaim)
receipt = execute_density_reclaim("run", plan=plan_density_reclaim("run"))
print("first :", receipt["status"], receipt["reclaimed"])
receipt = execute_density_reclaim("run", plan=plan_density_reclaim("run"))
print("repeat:", receipt["status"], receipt["reclaimed"])
EOF

echo "demo done: $WORK_ABS (kept for inspection; delete it yourself)"
