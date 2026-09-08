"""Temperature-segment-stratified train/held-out splits (bootstrap stage 3).

Frame-generation campaigns run several condition segments (temperatures,
spike initial condition, ...) back to back in one concatenated extxyz. A
trustworthy held-out estimate needs every segment represented in the test
set, so the split rule is: within each segment (frame-index ranges
delimited by ``boundaries``), the LAST k labeled frames are held out.
Time-ordered within each segment, so validation never sees the future of a
segment the committee trained on.
"""

from __future__ import annotations

import numpy as np


def parse_boundaries(spec: str) -> list[int]:
    """Parse a CLI boundary spec: "38" -> [38]; "12,24,36" -> [12, 24, 36]."""
    return [int(x) for x in str(spec).split(",")]


def segment_held_out(idxs, boundaries: list[int], k: int) -> set[int]:
    """Frame indices held out by the per-segment last-k rule.

    ``boundaries`` are the frame indices at which segments 1..N-1 start
    (segment 0 covers everything below the first boundary). A
    single-element list reproduces the original two-segment split.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    idxs = np.sort(np.asarray(idxs, dtype=int))
    edges = [-np.inf, *sorted(boundaries), np.inf]
    held: set[int] = set()
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = idxs[(idxs >= lo) & (idxs < hi)]
        if seg.size:
            held.update(int(i) for i in seg[-k:])
    return held
