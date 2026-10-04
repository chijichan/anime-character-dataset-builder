"""Dedup core: hashing, pairwise distances, candidate gates, grouping policy."""

from __future__ import annotations

import numpy as np
import pytest

from acdb.pipeline.stages.dedup import (
    candidate_pairs,
    greedy_groups,
    hamming_matrix,
    hashes_to_bits,
)
from acdb.utils.image import hamming_distance, perception_hash


def _bits(*hashes: int) -> np.ndarray:
    return hashes_to_bits([f"{value:016x}" for value in hashes])


def test_hashes_to_bits_matches_manual_unpacking() -> None:
    bits = hashes_to_bits(["0000000000000001", "8000000000000000"])
    assert bits.shape == (2, 64)
    assert bits[0].sum() == 1 and bits[0][-1] == 1
    assert bits[1].sum() == 1 and bits[1][0] == 1


def test_hamming_matrix_is_exact_and_symmetric() -> None:
    bits = _bits(0b1011, 0b1001, 0b0000)
    distances = hamming_matrix(bits)
    assert distances[0, 1] == 1
    assert distances[0, 2] == 3
    assert distances[1, 2] == 2
    assert np.array_equal(distances, distances.T)
    assert np.array_equal(np.diag(distances), np.zeros(3, dtype=np.uint16))


def test_hamming_distance_helper_agrees_with_matrix() -> None:
    assert hamming_distance("0000000000000000", "000000000000000f") == 4
    with pytest.raises(ValueError):
        hamming_distance("00", "0000")
    with pytest.raises(ValueError):
        hamming_distance("zz", "00")


def test_candidate_pairs_respects_threshold_and_order() -> None:
    bits = _bits(0b0000, 0b0001, 0b1111)
    distances = hamming_matrix(bits)
    assert candidate_pairs(distances, 1) == [(0, 1)]
    assert candidate_pairs(distances, 4) == [(0, 1), (0, 2), (1, 2)]
    assert candidate_pairs(distances, 0) == []


def test_candidate_pairs_can_be_restricted_to_one_video() -> None:
    bits = _bits(0b0000, 0b0001)
    distances = hamming_matrix(bits)
    assert candidate_pairs(distances, 1, groups=["a", "a"], same_group_only=True) == [(0, 1)]
    assert candidate_pairs(distances, 1, groups=["a", "b"], same_group_only=True) == []


def test_greedy_groups_does_not_chain_transitively() -> None:
    """A~B and B~C must not merge A and C into one group."""
    # A - B - C where the representative of the first group is A.
    neighbours = {0: [1], 1: [0, 2], 2: [1]}
    groups, overfull = greedy_groups(order=[0, 1, 2], neighbours=neighbours, max_group_size=10)
    assert groups == [[0, 1]]
    assert overfull == []
    # C stays unassigned, so it survives on its own (it is simply not in a group).
    assert 2 not in [member for group in groups for member in group]


def test_greedy_groups_anchors_on_the_best_representative() -> None:
    neighbours = {0: [1, 2], 1: [0], 2: [0]}
    groups, _ = greedy_groups(order=[2, 1, 0], neighbours=neighbours, max_group_size=10)
    assert groups == [[2, 0]]


def test_greedy_groups_reports_overfull_groups() -> None:
    neighbours = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}
    groups, overfull = greedy_groups(order=[0, 1, 2, 3], neighbours=neighbours, max_group_size=2)
    assert len(groups[0]) == 2
    assert overfull, "the surplus member must be reported, not silently dropped"


def test_perception_hash_is_stable_and_discriminating() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    same = image.copy()
    other = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

    assert perception_hash(image) == perception_hash(same)
    assert hamming_distance(perception_hash(image), perception_hash(other)) > 10


def test_perception_hash_survives_jpeg_noise() -> None:
    import cv2

    rng = np.random.default_rng(1)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[10:50, 10:50] = 200
    image = (image.astype(np.float32) + rng.normal(0, 4, image.shape)).clip(0, 255)
    ok, encoded = cv2.imencode(".jpg", image.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    assert ok
    noisy = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert hamming_distance(perception_hash(image.astype(np.uint8)), perception_hash(noisy)) <= 8
