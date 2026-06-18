"""HTML dataset report for the Sprint 2 Occupancy Detection scenario."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coinfosim.datasets.occupancy import OccupancyData


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _dataframe_html(df: pd.DataFrame, float_cols: Optional[Dict[str, str]] = None) -> str:
    float_cols = float_cols or {}
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if col in float_cols and not pd.isna(val):
                cells.append(f"<td>{float(val):{float_cols[col]}}</td>")
            else:
                cells.append(f"<td>{html.escape(str(val))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<table class='data'><thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _class_distribution_image(data: OccupancyData) -> str:
    counts = data.class_counts_by_file()
    labels = sorted(data.class_labels)
    files = list(counts)
    x = np.arange(len(files))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, label in enumerate(labels):
        values = [counts[file].get(label, 0) for file in files]
        ax.bar(x + (i - 0.5) * width, values, width, label=f"class {label}")
    ax.set_xticks(x)
    ax.set_xticklabels(files, rotation=20, ha="right")
    ax.set_ylabel("Rows")
    ax.set_title("Class distribution by source file")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return _fig_to_base64(fig)


def _correlation_heatmap_image(data: OccupancyData) -> str:
    corr = data.train_correlation(standardized=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Training-pool channel correlation")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_to_base64(fig)


def _standardized_summary_image(data: OccupancyData) -> str:
    summary = data.standardized_channel_summary()
    fig, ax = plt.subplots(figsize=(7, 4))
    channels = list(data.channel_names)
    x = np.arange(len(channels))
    train_means = summary[summary["split"] == "train_pool"].set_index("channel").loc[channels, "mean"]
    test_means = summary[summary["split"] == "fixed_test"].set_index("channel").loc[channels, "mean"]
    width = 0.35
    ax.bar(x - width / 2, train_means, width, label="train pool")
    ax.bar(x + width / 2, test_means, width, label="fixed test")
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=25, ha="right")
    ax.set_ylabel("Standardized mean")
    ax.set_title("Standardized channel means")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return _fig_to_base64(fig)


def generate_occupancy_dataset_report(
    data: OccupancyData,
    output_dir: Path | str = "output/reports",
    filename: str = "occupancy_dataset_report.html",
) -> Path:
    """Generate the Occupancy dataset HTML report and return its path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    row_counts_df = pd.DataFrame(
        [{"source_file": name, "rows": rows} for name, rows in data.row_counts().items()]
    )
    class_rows = []
    for name, counts in data.class_counts_by_file().items():
        row = {"source_file": name}
        for label in data.class_labels:
            row[f"class_{label}"] = counts.get(label, 0)
        class_rows.append(row)
    class_counts_df = pd.DataFrame(class_rows)
    source_df = pd.DataFrame(
        [
            {"source_file": name, "path": str(path)}
            for name, path in data.source_files.items()
        ]
    )
    standardization_df = data.standardization.as_dataframe().reset_index()
    standardization_df = standardization_df.rename(columns={"index": "channel"})
    raw_summary = data.raw_channel_summary()
    standardized_summary = data.standardized_channel_summary()
    corr_df = data.train_correlation(standardized=True).reset_index()
    corr_df = corr_df.rename(columns={"index": "channel"})

    float_fmt = {"mean": ".6f", "std": ".6f", "min": ".6f", "max": ".6f"}
    standardization_fmt = {"mean": ".6f", "std": ".6f"}

    class_distribution = _class_distribution_image(data)
    corr_heatmap = _correlation_heatmap_image(data)
    standardized_summary_plot = _standardized_summary_image(data)

    channels = ", ".join(data.channel_names)
    total_test = data.raw_test.shape[0]
    total_train = data.raw_train.shape[0]

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>CoInfoSim — Occupancy Dataset Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0 auto; max-width: 1000px; padding: 2rem; color: #222; line-height: 1.5; }}
  h1 {{ font-size: 1.8rem; border-bottom: 3px solid #1f77b4; padding-bottom: .4rem; }}
  h2 {{ font-size: 1.3rem; margin-top: 2rem; color: #1f3b66; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }}
  .notice {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .8rem 1rem; margin: 1rem 0; }}
  .question {{ font-style: italic; background: #eef5fb; padding: .8rem 1rem; border-left: 4px solid #1f77b4; }}
  table.data {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }}
  table.data th, table.data td {{ border: 1px solid #ccc; padding: .35rem .6rem; text-align: center; }}
  table.data th {{ background: #f0f4f8; }}
  .figure {{ text-align: center; margin: 1.2rem 0; }}
  .figure img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
  dl.meta {{ display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; }}
  dl.meta dt {{ font-weight: 600; color: #444; }}
  code {{ background: #f5f5f5; padding: .1rem .3rem; border-radius: 3px; }}
</style>
</head>
<body>

<h1>CoInfoSim — Occupancy Dataset Report</h1>
<p><strong>Scenario:</strong> Occupancy Detection real-data anchored scenario</p>

<div class="notice"><strong>Protocol.</strong> <code>datatraining.txt</code> is the training pool.
<code>datatest.txt</code> and <code>datatest2.txt</code> are concatenated into one fixed test set.
Standardization parameters are learned from the training pool only.</div>

<h2>Source files</h2>
{_dataframe_html(source_df)}

<h2>Train/test protocol</h2>
<dl class="meta">
  <dt>Training pool</dt><dd><code>datatraining.txt</code> ({total_train} rows)</dd>
  <dt>Fixed test set</dt><dd><code>datatest.txt + datatest2.txt</code> ({total_test} rows)</dd>
  <dt>Channels</dt><dd>{html.escape(channels)}</dd>
  <dt>Target</dt><dd><code>{html.escape(data.target_name)}</code></dd>
  <dt>Class labels</dt><dd>{list(data.class_labels)}</dd>
</dl>

<h2>Row counts</h2>
{_dataframe_html(row_counts_df)}

<h2>Class distribution</h2>
{_dataframe_html(class_counts_df)}
<div class="figure"><img src="{class_distribution}" alt="class distribution"/></div>

<h2>Raw channel statistics</h2>
{_dataframe_html(raw_summary, float_cols=float_fmt)}

<h2>Standardization parameters</h2>
<p>Means and standard deviations below are estimated from <code>datatraining.txt</code> only.</p>
{_dataframe_html(standardization_df, float_cols=standardization_fmt)}

<h2>Standardized channel statistics</h2>
{_dataframe_html(standardized_summary, float_cols=float_fmt)}
<div class="figure"><img src="{standardized_summary_plot}" alt="standardized channel means"/></div>

<h2>Training-pool correlation matrix</h2>
{_dataframe_html(corr_df, float_cols={channel: ".3f" for channel in data.channel_names})}
<div class="figure"><img src="{corr_heatmap}" alt="correlation heatmap"/></div>

</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path
