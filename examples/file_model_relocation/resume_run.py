"""Step 2 of the file-model relocation example: resume and export.

Usage:
  python examples/file_model_relocation/resume_run.py --run RUN_DIR
  python examples/file_model_relocation/resume_run.py --run RUN_DIR \
      --model /absolute/path/to/model.dat

Without ``--model`` the run resumes in place.  After a relocation the
recorded model path no longer exists, so the resume refuses clearly and
names the missing resource; passing ``--model`` maps the declared
``reference.potential`` role to the file's new location.  The file's bytes
are re-verified against the run's baseline before any computation — the
mapping is never remembered, so every later restart after a move needs it
again.  On success the committed trajectory is exported (driving forces);
an existing export target is never overwritten — the conflict is refused
before any resume happens, so retrying never adds a second batch of steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pyraimd2.workflows import resume_workflow
from pyraimd2.workflows.export import export_run
from pyraimd2.workflows.setup import WorkflowError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True,
                        help="the run directory (possibly relocated)")
    parser.add_argument("--extra-steps", type=int, default=2,
                        help="additional NVE steps (default 2)")
    parser.add_argument("--model", type=Path, default=None,
                        help="current location of the declared model file "
                             "after a relocation (absolute path)")
    parser.add_argument("--export", type=Path, default=None,
                        help="extxyz output (default: export-driving.extxyz "
                             "inside the run directory); an existing target "
                             "is never overwritten")
    args = parser.parse_args()

    # Resolve the real export target and refuse a conflict BEFORE resuming:
    # an existing file (or a link pointing at one) keeps its content, and a
    # late export failure must never push the user into retrying with a
    # second batch of steps.  The export API's own no-overwrite default
    # stays as the last line of defense.
    output = (args.export if args.export is not None
              else args.run / "export-driving.extxyz")
    if os.path.lexists(output):
        print(f"export target exists: {output}; nothing was resumed or "
              "written — pass --export with a different path (or move the "
              "existing file aside)", file=sys.stderr)
        raise SystemExit(2)

    resource_paths = None
    if args.model is not None:
        model = args.model.resolve()  # the mapping takes absolute paths
        resource_paths = {"reference.potential": str(model)}
    try:
        result = resume_workflow(args.run, args.extra_steps,
                                 resource_paths=resource_paths,
                                 verbose=False)
    except WorkflowError as error:
        print(f"resume refused: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    report = export_run(args.run, force_source="driving", output=output)
    print(json.dumps({
        "steps_completed": result.steps_completed,
        "export": {"output": str(report["output"]),
                   "frames": report["frames"],
                   "force_source": report["force_source"]},
    }, indent=2))


if __name__ == "__main__":
    main()
