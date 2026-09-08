"""segment_held_out: per-segment last-k split (bootstrap stage 3)."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_k_zero_raises_value_error():
    # k=0 degenerates to seg[0:]: every segment is held out in full and the
    # training set is silently emptied — reject it up front.
    with pytest.raises(ValueError, match=r"k must be >= 1"):
        segment_held_out(np.arange(10), [5], 0)


def test_negative_k_raises_value_error():
    # k=-1 degenerates to seg[1:], silently dropping the first frame of
    # every segment instead of the last — same class of footgun as k=0.
    with pytest.raises(ValueError, match=r"k must be >= 1"):
        segment_held_out(np.arange(10), [5], -1)


@pytest.mark.parametrize(
    "k, expected",
    [
        (1, {4, 9}),
        (2, {3, 4, 8, 9}),
        (3, {2, 3, 4, 7, 8, 9}),
        (4, {1, 2, 3, 4, 6, 7, 8, 9}),
        (5, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
    ],
)
def test_valid_k_one_to_five_holds_out_last_k_of_each_segment(k, expected):
    # Guard against over-fixing: every k >= 1 keeps the legacy per-segment
    # last-k rule intact (k=5 saturates both size-5 segments).
    held = segment_held_out(np.arange(10), [5], k)
    assert held == expected
