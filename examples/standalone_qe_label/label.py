"""Standalone QE energy/force label on the unified scratch lifecycle.

One label per invocation: build the engine with a unified managed
``scratch_root`` and ``retention="results"``, compute — the attempt runs
in its exclusive directory under that root, the verified result is
archived into the label's persistent run directory, and the attempt's
scratch subtree is reclaimed right away.  The script prints the archived
result location and the scratch record state.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from ase.io import read as ase_read

from pyraimd2.engines.qe_engine import QeConfig, QeEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="the label task's persistent run root "
                             "(archived results land here)")
    parser.add_argument("--scratch-root", type=Path, default=Path("./tmp"),
                        help="the unified managed tmp root (default: ./tmp)")
    parser.add_argument("--pw-cmd", default="pw.x",
                        help="pw.x launch command (default: pw.x)")
    parser.add_argument("--pseudo-dir", type=Path, required=True)
    parser.add_argument("--label", default="label")
    args = parser.parse_args()

    atoms = ase_read(args.structure)
    pw_cmd = ((args.pw_cmd,) if Path(args.pw_cmd).is_file()
              else tuple(shlex.split(args.pw_cmd)))
    config = QeConfig(pseudo_dir=str(args.pseudo_dir),
                      pw_cmd=pw_cmd,
                      scratch_root=str(args.scratch_root),
                      retention="results")
    engine = QeEngine(config, run_root=args.run_root)
    result = engine.compute(atoms, label=args.label)

    # the archive location and lifecycle record of THIS computation come
    # from its own handle — never from a guessed counter or the first
    # record in the run root
    handle = engine._last_scratch_handle
    if handle is not None:
        archive_dir = Path(handle.archive_dir)
        record_path = Path(handle.record_path)
    else:
        archive_dir = Path(engine.last_attempt_records[-1]["directory"])
        record_path = None
    state = (json.loads(record_path.read_text()) if record_path is not None
             else {"state": "unmanaged"})
    label_path = args.run_root / f"label-{args.label}.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps({
        "label": args.label,
        "engine": engine.name,
        "reference_fingerprint": engine.fingerprint,
        "energy_eV": result.energy,
        "forces_eV_A": result.forces.tolist(),
        "stress_eV_A3": (None if result.stress is None
                         else result.stress.tolist()),
        "archive_dir": str(archive_dir),
        "scratch_record": (None if record_path is None else str(record_path)),
        "scratch_state": state["state"],
    }, indent=2) + "\n")

    print(f"label written: {label_path}")
    print(f"energy: {result.energy:.8f} eV")
    print("archived result files:")
    for path in sorted(archive_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(archive_dir)}")
    print(f"scratch: {state['state']} "
          f"({state.get('removed_bytes', 0)} bytes reclaimed from "
          f"{state.get('scratch_dir', '-')})")


if __name__ == "__main__":
    sys.exit(main())
