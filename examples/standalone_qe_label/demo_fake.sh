#!/bin/sh
# Public end-to-end demo of the unified scratch lifecycle on a FAKE pw.x:
# it verifies the program flow only — no real DFT label is produced.
#
#   install → configure a unified tmp root with retention="results" → run
#   a standalone label → read the archived result → inspect/clean the root
#
# Usage (from the repository root, with the package installed):
#   sh examples/standalone_qe_label/demo_fake.sh [WORK_DIR]
set -eu

WORK=${1:-demo-scratch-work}
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
WORK_ABS=$(CDPATH= cd -- "$WORK" && pwd)

cat > "$WORK_ABS/fake_pwx.sh" <<EOF
#!/bin/bash
mkdir -p tmp/pyraimd2.save && echo fake-density > tmp/pyraimd2.save/charge-density.dat
cat "$FIXTURE"
EOF
chmod +x "$WORK_ABS/fake_pwx.sh"

"$PY" - <<'EOF'
import os
from ase import Atoms
from ase.io import write
write(os.path.join(os.environ["WORK"], "si2.extxyz"),
      Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]],
            cell=[5.43] * 3, pbc=True))
EOF

cd "$WORK_ABS"
echo "== run the standalone label (tmp/ + retention=results) =="
"$PY" "$HERE/label.py" --structure si2.extxyz --run-root label \
    --scratch-root ./tmp --pw-cmd "$WORK_ABS/fake_pwx.sh" \
    --pseudo-dir ./pseudos --label case-00

echo "== archived result (outside the unified tmp) =="
cat label/*/attempt-1/density_manifest.json

echo "== pyramid scratch inspect =="
pyramid scratch inspect --root ./tmp
echo "== pyramid scratch clean (dry-run) =="
pyramid scratch clean --root ./tmp --dry-run
echo "== pyramid scratch clean (idempotent repeat) =="
pyramid scratch clean --root ./tmp
