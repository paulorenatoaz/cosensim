import numpy as np
import pytest

from coinfosim.datasets.occupancy import OCCUPANCY_CHANNELS, load_occupancy_data
from coinfosim.samplers.real import RealDatasetSampler


def _sampler(base_seed=123):
    data = load_occupancy_data("data/raw/occupancy")
    return RealDatasetSampler(
        data.train_dataset,
        data.test_dataset,
        base_seed=base_seed,
        channel_names=data.channel_names,
        name="occupancy",
    )


def test_real_sampler_metadata_for_occupancy():
    sampler = _sampler()

    assert sampler.model.d == 5
    assert sampler.model.K == 2
    assert sampler.model.class_labels == (0, 1)
    assert sampler.model.channel_names == OCCUPANCY_CHANNELS
    assert sampler.min_class_count == 1729
    assert sampler.class_count(0) == 6414
    assert sampler.class_count(1) == 1729


def test_real_sampler_balanced_output():
    sampler = _sampler()
    train = sampler.sample_train(n_per_class=16, replication_id=0)

    assert train.X.shape == (32, 5)
    labels, counts = np.unique(train.y, return_counts=True)
    assert tuple(labels) == (0, 1)
    assert tuple(counts) == (16, 16)


def test_real_sampler_is_deterministic():
    sampler = _sampler()

    a = sampler.sample_train(n_per_class=32, replication_id=7)
    b = sampler.sample_train(n_per_class=32, replication_id=7)

    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)


def test_real_sampler_changes_with_replication_id():
    sampler = _sampler()

    a = sampler.sample_train(n_per_class=32, replication_id=7)
    b = sampler.sample_train(n_per_class=32, replication_id=8)

    assert not np.array_equal(a.X, b.X)


def test_real_sampler_prefix_nesting():
    sampler = _sampler()

    small = sampler.sample_train(n_per_class=8, replication_id=3)
    large = sampler.sample_train(n_per_class=32, replication_id=3)

    assert np.array_equal(small.X[:8], large.X[:8])
    assert np.array_equal(small.y[:8], large.y[:8])
    assert np.array_equal(small.X[8:16], large.X[32:40])
    assert np.array_equal(small.y[8:16], large.y[32:40])


def test_real_sampler_fixed_test_set_reused():
    sampler = _sampler()

    first = sampler.sample_test()
    second = sampler.sample_test()

    assert first is second
    assert first.X.shape == (12417, 5)


def test_real_sampler_rejects_oversized_n_per_class():
    sampler = _sampler()

    with pytest.raises(ValueError, match="minority-class count"):
        sampler.sample_train(n_per_class=1730, replication_id=0)
