"""Top-level Occupancy Detection scenario report."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence

import pandas as pd

from coinfosim.classifiers.registry import classifier_label
from coinfosim.results.analysis import standard_threshold_comparisons
from coinfosim.results.summary import summary_dataframe
from coinfosim.reports.monte_carlo import subset_display_label
from coinfosim.simulation.monte_carlo import SimulationResult


def generate_occupancy_scenario_report(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    output_dir: Path | str = "output/reports",
    dataset_report: str = "occupancy_dataset_report.html",
    real_report: str = "occupancy_real_monte_carlo_report.html",
    gaussian_report: str = "occupancy_gaussian_anchored_monte_carlo_report.html",
    filename: str = "occupancy_scenario_report.html",
    channel_names: Sequence[str] | None = None,
) -> Path:
    """Generate the top-level Sprint 2 Occupancy scenario report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    channel_names = tuple(
        channel_names
        or real_result.metadata.get("channel_names", [])
        or gaussian_result.metadata.get("channel_names", [])
    )

    real_best = _best_at_largest_n(real_result, channel_names)
    gaussian_best = _best_at_largest_n(gaussian_result, channel_names)
    comparison = _best_comparison(real_best, gaussian_best)
    real_top = _top_rankings(real_result, channel_names)
    gaussian_top = _top_rankings(gaussian_result, channel_names)
    nstar_summary = _nstar_availability(real_result, gaussian_result, channel_names)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>CoInfoSim - Occupancy Scenario Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0 auto; max-width: 1100px; padding: 2rem; color: #222; line-height: 1.5; }}
  h1 {{ font-size: 1.8rem; border-bottom: 3px solid #1f77b4; padding-bottom: .4rem; }}
  h2 {{ font-size: 1.3rem; margin-top: 2rem; color: #1f3b66; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }}
  .notice {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .8rem 1rem; margin: 1rem 0; }}
  .question {{ font-style: italic; background: #eef5fb; padding: .8rem 1rem; border-left: 4px solid #1f77b4; }}
  table.data {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .88rem; }}
  table.data th, table.data td {{ border: 1px solid #ccc; padding: .34rem .55rem; text-align: center; }}
  table.data th {{ background: #f0f4f8; }}
  dl.meta {{ display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; }}
  dl.meta dt {{ font-weight: 600; color: #444; }}
  code {{ background: #f5f5f5; padding: .1rem .3rem; border-radius: 3px; }}
</style>
</head>
<body>

<h1>CoInfoSim - Occupancy Scenario Report</h1>
<p class="question">Does the cooperative advantage among real information channels in the Occupancy Detection dataset resemble the cooperative advantage predicted by a Gaussian model parameterized from the same real data?</p>

<div class="notice"><strong>Mode validation.</strong> This report is generated from Sprint 2 {html.escape(str(real_result.config.mode))} mode with sample sizes {list(real_result.sample_sizes)}. Full mode was not run.</div>

<h2>Navigation</h2>
<ul>
  <li><a href="{html.escape(dataset_report)}">Dataset report</a></li>
  <li><a href="{html.escape(real_report)}">Real-data Monte Carlo report</a></li>
  <li><a href="{html.escape(gaussian_report)}">Gaussian-anchored Monte Carlo report</a></li>
</ul>

<h2>Scenario summary</h2>
<dl class="meta">
  <dt>Channels</dt><dd>{html.escape(", ".join(channel_names))}</dd>
  <dt>Sample sizes</dt><dd>{list(real_result.sample_sizes)}</dd>
  <dt>Channel subsets</dt><dd>{len(real_result.subsets)}</dd>
  <dt>Classifiers</dt><dd>{len(real_result.classifier_names)} ({html.escape(", ".join(classifier_label(c) for c in real_result.classifier_names))})</dd>
  <dt>Real fixed test rows</dt><dd>{real_result.metadata.get("fixed_test_size")}</dd>
  <dt>Gaussian fixed test rows</dt><dd>{gaussian_result.metadata.get("fixed_test_size")}</dd>
</dl>

<h2>Best subset at largest smoke n</h2>
{_dataframe_html(comparison, float_cols={"Real mean loss": ".4f", "Gaussian mean loss": ".4f"})}

<h2>Real-data top-ranked subsets</h2>
{_dataframe_html(real_top, float_cols={"Mean test loss": ".4f", "Std. error": ".4f"})}

<h2>Gaussian-anchored top-ranked subsets</h2>
{_dataframe_html(gaussian_top, float_cols={"Mean test loss": ".4f", "Std. error": ".4f"})}

<h2>Interpolated N-star availability</h2>
{_dataframe_html(nstar_summary)}

<h2>Interpretation note</h2>
<p>The tables above compare only smoke-mode estimates. They are useful for validating the pipeline and for a first look at whether the real-data cooperative patterns resemble the Gaussian-anchored surrogate, but they are not a replacement for a larger run.</p>

</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path


def _best_at_largest_n(
    result: SimulationResult, channel_names: Sequence[str]
) -> pd.DataFrame:
    n = max(result.sample_sizes)
    summary = _summary_with_named_subsets(result, channel_names)
    summary = summary[summary["n_per_class"] == n].copy()
    summary = summary.sort_values(["classifier", "mean_loss", "standard_error"])
    return summary.groupby("classifier", as_index=False).first()


def _best_comparison(real_best: pd.DataFrame, gaussian_best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for classifier in real_best["classifier"]:
        real_row = real_best[real_best["classifier"] == classifier].iloc[0]
        gaussian_row = gaussian_best[gaussian_best["classifier"] == classifier].iloc[0]
        rows.append(
            {
                "Classifier": real_row["classifier_label"],
                "Real best subset": real_row["subset_label"],
                "Real mean loss": real_row["mean_loss"],
                "Gaussian best subset": gaussian_row["subset_label"],
                "Gaussian mean loss": gaussian_row["mean_loss"],
                "Same best subset": "yes"
                if real_row["subset_label"] == gaussian_row["subset_label"]
                else "no",
            }
        )
    return pd.DataFrame(rows)


def _top_rankings(
    result: SimulationResult,
    channel_names: Sequence[str],
    top_k: int = 5,
) -> pd.DataFrame:
    n = max(result.sample_sizes)
    summary = _summary_with_named_subsets(result, channel_names)
    summary = summary[summary["n_per_class"] == n].copy()
    summary = summary.sort_values(["classifier", "mean_loss", "standard_error"])
    summary["Rank"] = summary.groupby("classifier").cumcount() + 1
    summary = summary[summary["Rank"] <= top_k]
    return summary[
        ["classifier_label", "Rank", "subset_label", "mean_loss", "standard_error"]
    ].rename(
        columns={
            "classifier_label": "Classifier",
            "subset_label": "Subset",
            "mean_loss": "Mean test loss",
            "standard_error": "Std. error",
        }
    )


def _nstar_availability(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    channel_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for arm, result in (
        ("Real-data", real_result),
        ("Gaussian-anchored", gaussian_result),
    ):
        thresholds = standard_threshold_comparisons(
            result.accumulator,
            result.classifier_names,
            result.sample_sizes,
            result.subsets,
        )
        for _, row in thresholds.iterrows():
            rows.append(
                {
                    "Arm": arm,
                    "Classifier": row["classifier_label"],
                    "Comparison": _comparison_label(row, channel_names),
                    "N* grid": row["n_star_grid"],
                    "N* interpolated available": "yes"
                    if pd.notna(row["n_star_interpolated"])
                    else "no",
                    "Status": row["threshold_status"],
                }
            )
    return pd.DataFrame(rows)


def _comparison_label(row, channel_names: Sequence[str]) -> str:
    return (
        f"{subset_display_label(row['subset_b'], channel_names)} vs "
        f"{subset_display_label(row['subset_a'], channel_names)}"
    )


def _summary_with_named_subsets(
    result: SimulationResult, channel_names: Sequence[str]
) -> pd.DataFrame:
    summary = summary_dataframe(
        result.accumulator,
        result.sample_sizes,
        result.subsets,
        result.classifier_names,
    )
    summary = summary.copy()
    summary["subset_label"] = [
        subset_display_label(subset, channel_names) for subset in summary["subset"]
    ]
    return summary


def _dataframe_html(df: pd.DataFrame, float_cols: dict[str, str] | None = None) -> str:
    float_cols = float_cols or {}
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            value = row[col]
            if col in float_cols and pd.notna(value):
                cells.append(f"<td>{float(value):{float_cols[col]}}</td>")
            elif pd.isna(value):
                cells.append("<td>&mdash;</td>")
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<table class='data'><thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
