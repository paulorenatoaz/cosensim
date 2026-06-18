#!/usr/bin/env python3
"""Run the Sprint 2 Occupancy Detection scenario."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.reports.occupancy_dataset import generate_occupancy_dataset_report
from coinfosim.reports.occupancy_monte_carlo import (
    generate_occupancy_gaussian_anchored_monte_carlo_report,
    generate_occupancy_real_monte_carlo_report,
)
from coinfosim.reports.occupancy_scenario import generate_occupancy_scenario_report
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.scenarios.occupancy import build_gaussian_anchored_occupancy_model
from coinfosim.simulation.config import VALID_MODES, get_mode_config
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALID_MODES, default="smoke")
    parser.add_argument("--raw-dir", default="data/raw/occupancy")
    parser.add_argument("--output-dir", default="output/reports")
    args = parser.parse_args()

    if args.mode == "full":
        raise SystemExit(
            "Full mode is intentionally disabled for Sprint 2 local validation."
        )

    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_mode_config(args.mode)
    data = load_occupancy_data(args.raw_dir)
    dataset_report = generate_occupancy_dataset_report(data, output_dir)

    real_sampler = RealDatasetSampler(
        data.train_dataset,
        data.test_dataset,
        base_seed=config.base_seed,
        channel_names=data.channel_names,
        name="occupancy_real_data",
    )
    real_result = CooperativeMonteCarloSimulator(
        real_sampler.model,
        config,
        sampler=real_sampler,
        metadata={
            "scenario_name": "Occupancy Detection",
            "experiment_arm": "real_data",
            "channel_names": list(data.channel_names),
            "standardization": "train_pool_only",
        },
    ).run()
    real_report = generate_occupancy_real_monte_carlo_report(
        real_result, output_dir
    )

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
        metadata={
            "scenario_name": "Occupancy Detection",
            "experiment_arm": "gaussian_anchored",
            "channel_names": list(data.channel_names),
            "standardization": "train_pool_only",
            "gaussian_ridge_by_class": dict(anchored.ridge_by_class),
        },
    ).run()
    gaussian_report = generate_occupancy_gaussian_anchored_monte_carlo_report(
        gaussian_result,
        data.channel_names,
        output_dir,
    )
    scenario_report = generate_occupancy_scenario_report(
        real_result,
        gaussian_result,
        output_dir=output_dir,
        channel_names=data.channel_names,
    )

    real_summary = _result_summary(real_result, real_report)
    gaussian_summary = _result_summary(gaussian_result, gaussian_report)
    real_summary_path = output_dir / "occupancy_real_monte_carlo_summary.json"
    gaussian_summary_path = (
        output_dir / "occupancy_gaussian_anchored_monte_carlo_summary.json"
    )
    real_summary_path.write_text(json.dumps(real_summary, indent=2), encoding="utf-8")
    gaussian_summary_path.write_text(
        json.dumps(gaussian_summary, indent=2), encoding="utf-8"
    )

    runtime = time.time() - start
    print(f"mode: {config.mode}")
    print(f"sample_sizes: {list(config.sample_sizes)}")
    print(f"runtime_seconds: {runtime:.2f}")
    print(f"dataset_report: {dataset_report}")
    print(f"scenario_report: {scenario_report}")
    print(f"real_report: {real_report}")
    print(f"gaussian_report: {gaussian_report}")
    print(f"real_summary: {real_summary_path}")
    print(f"gaussian_summary: {gaussian_summary_path}")
    print(
        "real_stopping: "
        + json.dumps(real_summary["stopping_info"], sort_keys=True)
    )
    print(
        "gaussian_stopping: "
        + json.dumps(gaussian_summary["stopping_info"], sort_keys=True)
    )
    return 0


def _result_summary(result, report_path: Path) -> dict:
    return {
        "report_path": str(report_path),
        "mode": result.config.mode,
        "sample_sizes": list(result.sample_sizes),
        "number_of_subsets": len(result.subsets),
        "number_of_classifiers": len(result.classifier_names),
        "classifier_names": list(result.classifier_names),
        "fixed_test_size": result.metadata.get("fixed_test_size"),
        "metadata": result.metadata,
        "stopping_info": {
            str(n): {
                "replications": result.stopping_info[n].replications,
                "reason": result.stopping_info[n].reason,
                "max_ci_half_width": result.stopping_info[n].max_ci_half_width,
            }
            for n in result.sample_sizes
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
