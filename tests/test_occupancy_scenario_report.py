from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.reports.occupancy_scenario import generate_occupancy_scenario_report
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.scenarios.occupancy import build_gaussian_anchored_occupancy_model
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


def _tiny_config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(4, 8),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=20,
        ci_half_width_target=0.05,
        base_seed=19,
    )


def _results():
    data = load_occupancy_data("data/raw/occupancy")
    config = _tiny_config()

    real_sampler = RealDatasetSampler(
        data.train_dataset,
        data.test_dataset,
        base_seed=config.base_seed,
        channel_names=data.channel_names,
    )
    real_result = CooperativeMonteCarloSimulator(
        real_sampler.model,
        config,
        sampler=real_sampler,
        metadata={"channel_names": list(data.channel_names)},
    ).run()

    anchored = build_gaussian_anchored_occupancy_model(data)
    gaussian_sampler = GaussianClassConditionalSampler(
        anchored.model,
        base_seed=config.base_seed,
        test_samples_per_class=config.test_samples_per_class,
    )
    gaussian_result = CooperativeMonteCarloSimulator(
        anchored.model,
        config,
        sampler=gaussian_sampler,
        metadata={"channel_names": list(data.channel_names)},
    ).run()
    return data, real_result, gaussian_result


def test_occupancy_scenario_report_links_and_comparison(tmp_path):
    data, real_result, gaussian_result = _results()
    out = generate_occupancy_scenario_report(
        real_result,
        gaussian_result,
        output_dir=tmp_path,
        channel_names=data.channel_names,
    )
    text = out.read_text(encoding="utf-8")

    assert out.name == "occupancy_scenario_report.html"
    assert "Occupancy Scenario Report" in text
    assert "occupancy_dataset_report.html" in text
    assert "occupancy_real_monte_carlo_report.html" in text
    assert "occupancy_gaussian_anchored_monte_carlo_report.html" in text
    assert "Best subset at largest smoke n" in text
    assert "Real-data top-ranked subsets" in text
    assert "Gaussian-anchored top-ranked subsets" in text
    assert "Interpolated N-star availability" in text
    assert "Channel subsets</dt><dd>31" in text
    assert "Classifiers</dt><dd>3" in text
