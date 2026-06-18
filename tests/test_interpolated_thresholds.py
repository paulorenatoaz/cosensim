import math

from coinfosim.results.accumulator import LossAccumulator
from coinfosim.results.analysis import (
    cooperative_threshold,
    cooperative_threshold_interpolated,
    standard_threshold_comparisons,
)


CLF = "linear_svm"
A = (0,)
B = (0, 1)


def _accumulator(deltas):
    """Build losses where Delta(n) = L_A(n) - L_B(n)."""
    acc = LossAccumulator()
    for rep, (n, delta) in enumerate(deltas.items()):
        loss_b = 0.30
        loss_a = loss_b + delta
        acc.add(n, A, CLF, rep, loss_a)
        acc.add(n, B, CLF, rep, loss_b)
        # Extra subsets required by standard_threshold_comparisons.
        acc.add(n, (1,), CLF, rep, loss_a + 0.10)
        acc.add(n, (0, 1, 2), CLF, rep, loss_b - 0.01)
        acc.add(n, (0, 2), CLF, rep, loss_b - 0.02)
    return acc


def test_interpolated_threshold_known_crossing():
    acc = _accumulator({4: -0.10, 8: -0.05, 16: 0.05, 32: 0.10})

    result = cooperative_threshold_interpolated(acc, CLF, A, B, [4, 8, 16, 32])

    assert result.n_star_grid == 16
    assert result.n_before == 8
    assert result.n_after == 16
    assert math.isclose(result.delta_before, -0.05)
    assert math.isclose(result.delta_after, 0.05)
    assert math.isclose(result.n_star_interpolated, 12.0)
    assert result.status == "interpolated"


def test_interpolated_threshold_no_crossing():
    acc = _accumulator({4: -0.10, 8: -0.05, 16: -0.01, 32: 0.0})

    result = cooperative_threshold_interpolated(acc, CLF, A, B, [4, 8, 16, 32])

    assert result.n_star_grid is None
    assert result.n_star_interpolated is None
    assert result.delta_before is None
    assert result.delta_after is None
    assert result.status == "no_crossing"


def test_interpolated_threshold_already_winning_at_first_sample_size():
    acc = _accumulator({4: 0.02, 8: 0.03, 16: 0.04})

    result = cooperative_threshold_interpolated(acc, CLF, A, B, [4, 8, 16])

    assert result.n_star_grid == 4
    assert result.n_star_interpolated == 4.0
    assert result.n_before is None
    assert result.n_after == 4
    assert result.delta_before is None
    assert math.isclose(result.delta_after, 0.02)
    assert result.status == "left_censored"


def test_interpolated_threshold_exact_zero_before_win():
    acc = _accumulator({4: -0.10, 8: 0.0, 16: 0.08})

    result = cooperative_threshold_interpolated(acc, CLF, A, B, [4, 8, 16])

    assert result.n_star_grid == 16
    assert result.n_star_interpolated == 8.0
    assert result.n_before == 8
    assert result.n_after == 16
    assert result.delta_before == 0.0
    assert math.isclose(result.delta_after, 0.08)
    assert result.status == "exact_zero"


def test_discrete_threshold_remains_available():
    acc = _accumulator({4: -0.10, 8: -0.05, 16: 0.05})

    result = cooperative_threshold(acc, CLF, A, B, [4, 8, 16])

    assert result.n_star == 16


def test_standard_threshold_comparisons_include_interpolated_fields():
    acc = _accumulator({4: -0.10, 8: -0.05, 16: 0.05})

    df = standard_threshold_comparisons(
        acc,
        [CLF],
        [4, 8, 16],
        [(0,), (1,), (0, 1), (0, 2), (0, 1, 2)],
    )

    for column in (
        "n_star",
        "n_star_grid",
        "n_star_interpolated",
        "n_before",
        "n_after",
        "delta_before",
        "delta_after",
        "threshold_status",
    ):
        assert column in df.columns
