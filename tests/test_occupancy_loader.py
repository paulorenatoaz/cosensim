from pathlib import Path

import numpy as np

from coinfosim.datasets.occupancy import (
    OCCUPANCY_CHANNELS,
    load_occupancy_data,
)
from coinfosim.reports.occupancy_dataset import generate_occupancy_dataset_report


RAW_DIR = Path("data/raw/occupancy")


def test_required_occupancy_raw_files_exist():
    for name in ("datatraining.txt", "datatest.txt", "datatest2.txt"):
        assert (RAW_DIR / name).exists()


def test_occupancy_loader_row_counts_and_columns():
    data = load_occupancy_data(RAW_DIR)

    assert data.row_counts() == {
        "datatraining.txt": 8143,
        "datatest.txt": 2665,
        "datatest2.txt": 9752,
    }
    assert tuple(data.channel_names) == OCCUPANCY_CHANNELS
    assert data.target_name == "Occupancy"
    assert data.class_labels == (0, 1)
    assert list(data.raw_train.columns) == [
        "row_id",
        "date",
        "Temperature",
        "Humidity",
        "Light",
        "CO2",
        "HumidityRatio",
        "Occupancy",
    ]
    assert data.train_dataset.X.shape == (8143, 5)
    assert data.test_dataset.X.shape == (2665 + 9752, 5)


def test_occupancy_loader_has_no_missing_required_values():
    data = load_occupancy_data(RAW_DIR)
    required = ["row_id", "date", *OCCUPANCY_CHANNELS, "Occupancy"]

    assert not data.raw_train.loc[:, required].isna().any().any()
    assert not data.raw_test.loc[:, required].isna().any().any()


def test_occupancy_standardization_is_fit_on_train_pool_only():
    data = load_occupancy_data(RAW_DIR)

    train_X = data.standardized_train.loc[:, OCCUPANCY_CHANNELS]
    assert np.allclose(train_X.mean().to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(train_X.std(ddof=0).to_numpy(), 1.0, atol=1e-12)

    manual_test = (
        data.raw_test.loc[:, OCCUPANCY_CHANNELS] - data.standardization.means
    ) / data.standardization.stds
    assert np.allclose(
        manual_test.to_numpy(),
        data.standardized_test.loc[:, OCCUPANCY_CHANNELS].to_numpy(),
    )


def test_occupancy_class_counts_by_file():
    data = load_occupancy_data(RAW_DIR)
    assert data.class_counts_by_file() == {
        "datatraining.txt": {0: 6414, 1: 1729},
        "datatest.txt": {0: 1693, 1: 972},
        "datatest2.txt": {0: 7703, 1: 2049},
    }


def test_occupancy_dataset_report_is_generated(tmp_path):
    data = load_occupancy_data(RAW_DIR)
    out = generate_occupancy_dataset_report(data, tmp_path)
    text = out.read_text(encoding="utf-8")

    assert out.name == "occupancy_dataset_report.html"
    assert "Occupancy Dataset Report" in text
    assert "datatraining.txt" in text
    assert "datatest.txt + datatest2.txt" in text
    assert "Standardization parameters" in text
    assert "Training-pool correlation matrix" in text
    assert "data:image/png;base64," in text
