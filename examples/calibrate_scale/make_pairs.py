#!/usr/bin/env python3
"""Write a tiny paired-forces calibration file (analytic teaching data).

Three frames of two toy atoms in the harmonic well: fast forces from the
k = 0.8 well, reference forces exactly 1.25 times them (the same teaching
relationship as the scaled demo configurations — not material data, not a
fitted coefficient).  `pyramid calibrate-scale` recovers scale = 1.25
with a vanishing after-scaling training residual.

Usage: python make_pairs.py [--output pairs.npz]
"""
import argparse
from pathlib import Path

import numpy as np

POSITIONS = np.array([
    [[0.87, 0.92, 0.89], [0.93, 0.88, 0.91]],
    [[0.88, 0.91, 0.88], [0.94, 0.87, 0.92]],
    [[0.86, 0.93, 0.90], [0.92, 0.89, 0.90]],
])
R0 = 0.9
K_FAST = 0.8
TEACHING_RATIO = 1.25


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="pairs.npz")
    args = parser.parse_args()
    # F = -k (r - r0) of the fast well; the reference is exactly
    # TEACHING_RATIO times it, so the closed form has an exact answer
    fast = -K_FAST * (POSITIONS - R0)
    reference = TEACHING_RATIO * fast
    frame_ids = [f"toy-frame-{index}" for index in range(len(POSITIONS))]
    output = Path(args.output)
    np.savez(output,
             reference_forces_eV_A=reference,
             fast_forces_eV_A=fast,
             frame_ids=np.asarray(frame_ids),
             reference_id=np.asarray("toy-harmonic-reference-k1.0"),
             fast_model_id=np.asarray("toy-harmonic-fast-k0.8"),
             force_unit=np.asarray("eV/angstrom"))
    print(f"wrote {output} ({len(POSITIONS)} frames x 2 atoms, "
          f"reference = {TEACHING_RATIO} x fast exactly)")
    return output


if __name__ == "__main__":
    main()
