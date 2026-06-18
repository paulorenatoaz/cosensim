import numpy as np

from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.scenarios.occupancy import build_gaussian_anchored_occupancy_model


def test_gaussian_anchored_occupancy_shapes_and_labels():
    data = load_occupancy_data("data/raw/occupancy")
    scenario = build_gaussian_anchored_occupancy_model(data)
    model = scenario.model

    assert scenario.d == 5
    assert model.d == 5
    assert model.class_labels == (0, 1)
    assert tuple(scenario.channel_names) == tuple(data.channel_names)

    for label in (0, 1):
        assert scenario.means[label].shape == (5,)
        assert scenario.covariances[label].shape == (5, 5)
        assert np.allclose(scenario.covariances[label], scenario.covariances[label].T)
        np.linalg.cholesky(scenario.covariances[label])


def test_gaussian_anchored_occupancy_estimates_from_standardized_train_data():
    data = load_occupancy_data("data/raw/occupancy")
    scenario = build_gaussian_anchored_occupancy_model(data)

    for label in data.class_labels:
        class_X = data.train_dataset.X[data.train_dataset.y == label]
        assert np.allclose(scenario.means[label], class_X.mean(axis=0))
        assert np.allclose(
            scenario.covariances[label],
            np.cov(class_X, rowvar=False, ddof=1)
            + scenario.ridge_by_class[label] * np.eye(data.d),
        )


def test_gaussian_anchored_occupancy_model_can_be_sampled():
    data = load_occupancy_data("data/raw/occupancy")
    scenario = build_gaussian_anchored_occupancy_model(data)
    sampler = GaussianClassConditionalSampler(
        scenario.model,
        base_seed=23,
        test_samples_per_class=20,
    )

    train = sampler.sample_train(n_per_class=8, replication_id=0)
    test = sampler.sample_test()

    assert train.X.shape == (16, 5)
    assert test.X.shape == (40, 5)
    assert tuple(sorted(np.unique(train.y))) == (0, 1)
    assert tuple(sorted(np.unique(test.y))) == (0, 1)
