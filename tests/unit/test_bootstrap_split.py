"""segment_held_out: per-segment last-k split (bootstrap stage 3)."""

from __future__ import annotations

import numpy as np

from pyraimd2.analysis import parse_boundaries, segment_held_out


def test_single_boundary_matches_legacy_two_segment_split():
    # Legacy behaviour: one boundary b, last k of each side held out.
    idxs = np.arange(76)
    held = segment_held_out(idxs, [38], 4)
    assert held == {34, 35, 36, 37, 72, 73, 74, 75}


def test_multi_boundary_four_segments():
    # W bootstrap layout: 49 frames, segments at 12 / 24 / 36, spike tail
    # runs to frame 48.
    idxs = np.arange(49)
    held = segment_held_out(idxs, [12, 24, 36], 3)
    assert held == {9, 10, 11, 21, 22, 23, 33, 34, 35, 46, 47, 48}


def test_unsorted_and_subset_indices():
    # The label array may miss frames (failed SCF); the rule applies to the
    # labeled subset, in order.
    idxs = np.array([47, 3, 40, 11, 22, 36, 9, 33, 46, 21, 34, 10])
    held = segment_held_out(idxs, [12, 24, 36], 2)
    assert held == {10, 11, 21, 22, 33, 34, 46, 47}


def test_k_larger_than_segment_holds_out_whole_segment():
    idxs = np.array([0, 1, 50, 51])
    held = segment_held_out(idxs, [25], 5)
    assert held == {0, 1, 50, 51}


def test_parse_boundaries():
    assert parse_boundaries("38") == [38]
    assert parse_boundaries("12,24,36") == [12, 24, 36]
    assert parse_boundaries(38) == [38]  # argparse may hand an int through
