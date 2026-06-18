# CoInfoSim Sprint 2 Tasks — Occupancy Real-Data Anchored Scenario

## Purpose

Sprint 2 implements the first real-data anchored scenario for CoInfoSim using the UCI Occupancy Detection dataset.

The sprint must produce a working smoke-mode experiment and four minimally informative HTML reports:

1. `occupancy_scenario_report.html`
2. `occupancy_dataset_report.html`
3. `occupancy_real_monte_carlo_report.html`
4. `occupancy_gaussian_anchored_monte_carlo_report.html`

The scenario report is the top-level page. It must compare the two Monte Carlo experiments and link to the three detailed reports.

The detailed reports are:

- a dataset report;
- a real-data Monte Carlo report;
- a Gaussian-anchored Monte Carlo report.

The report layout should follow the current Sprint 1 report style, but the visualizations can remain simple. The focus is correctness, reproducibility, and clear tables/curves, not polished visual design.

---

## Scientific goal

The Sprint 2 scenario asks:

> Does the cooperative advantage among real information channels in the Occupancy Detection dataset resemble the cooperative advantage predicted by a Gaussian model parameterized from the same real data?

The sprint therefore has two experimental arms:

\[
\boxed{\text{A. Real-data Monte Carlo}}
\]

\[
\boxed{\text{B. Gaussian-anchored Monte Carlo}}
\]

Both arms use the same evaluation logic:

\[
n \rightarrow r \rightarrow A \rightarrow f
\]

where:

- \(n\) is the number of training samples per class;
- \(r\) is the Monte Carlo replication id;
- \(A\) is a non-empty channel subset;
- \(f\) is a classifier.

The performance metric remains empirical test loss:

\[
L
\]

where \(L\) means the misclassification rate on a fixed test set.

Sprint 2 must still exclude:

- empirical train loss;
- theoretical loss;
- Bayes error;
- Bayes reference error.

---

## Dataset

Use the UCI Occupancy Detection files:

```text
datatraining.txt
datatest.txt
datatest2.txt
```

Expected repository location:

```text
data/raw/occupancy/datatraining.txt
data/raw/occupancy/datatest.txt
data/raw/occupancy/datatest2.txt
```

If these files are not present in that location, the agent must stop and ask the user to place them there before continuing.

### Columns

The data should be read as:

```text
row_id
date
Temperature
Humidity
Light
CO2
HumidityRatio
Occupancy
```

The header in the raw file can be misleading because the first field behaves as a row/index column. The loader should handle this explicitly and robustly.

### Channels

Use the five numerical channels:

\[
X =
(\text{Temperature}, \text{Humidity}, \text{Light}, \text{CO2}, \text{HumidityRatio})
\]

The target is:

\[
Y = \text{Occupancy}.
\]

Therefore:

\[
d=5
\]

and the number of non-empty channel subsets is:

\[
2^5 - 1 = 31.
\]

### Primary train/test protocol

Use the original dataset split:

\[
D_{\text{pool}} = \texttt{datatraining.txt}
\]

\[
D_{\text{test}} = \texttt{datatest.txt} \cup \texttt{datatest2.txt}.
\]

Do not implement an alternative random split in Sprint 2.

### Standardization

Fit standardization parameters only on `datatraining.txt`:

\[
X_j^{std} = \frac{X_j-\bar{x}_j}{s_j}.
\]

Apply the same transformation to:

- `datatraining.txt`;
- `datatest.txt`;
- `datatest2.txt`.

Do not use test data to compute standardization parameters.

---

## Execution modes

Sprint 2 must define the following mode sample sizes:

### smoke

Used now for actual testing:

\[
n \in \{4,8,16,32\}.
\]

Only `smoke` mode should be run during Sprint 2 development validation.

### fast

Defined but not required to run now:

\[
n \in \{4,8,16,32,64,128,256\}.
\]

### full

Defined but not required to run now:

\[
n \in \{4,8,16,32,64,128,256,512,1024\}.
\]

Do not run full tests in this sprint unless explicitly requested later.

---

## Classifiers

Use the same three classifiers already used in Sprint 1:

- Linear SVM;
- Logistic Regression;
- Gaussian Naive Bayes.

With \(d=5\), each replication and sample size evaluates:

\[
31 \times 3 = 93
\]

subset-classifier cells.

---

## Required reports

Sprint 2 must generate four pages.

### 1. Scenario report

Suggested path:

```text
output/reports/occupancy_scenario_report.html
```

Purpose: top-level comparison and navigation page.

Must include:

- scenario title;
- short scientific question;
- dataset summary;
- links to the dataset report;
- links to the real-data Monte Carlo report;
- links to the Gaussian-anchored Monte Carlo report;
- side-by-side comparison of real vs Gaussian-anchored results;
- best subset summary by classifier for each arm;
- interpolated \(N^\star\) summary by classifier for each arm;
- short interpretation section;
- clear note that this is smoke-mode validation unless another mode is used.

### 2. Dataset report

Suggested path:

```text
output/reports/occupancy_dataset_report.html
```

Must include:

- source files used;
- row counts per file;
- train/test protocol;
- class distribution per file;
- channel list;
- basic descriptive statistics for standardized and raw channels;
- correlation matrix or correlation table;
- standardization parameters learned from training data;
- note that `datatraining.txt` is used as the pool and `datatest.txt + datatest2.txt` as the fixed test set.

Charts may be simple, for example:

- class distribution bar chart;
- channel correlation heatmap;
- raw or standardized channel summary plot.

### 3. Real-data Monte Carlo report

Suggested path:

```text
output/reports/occupancy_real_monte_carlo_report.html
```

Must include:

- experiment arm: real-data Monte Carlo;
- mode and sample sizes;
- all 31 channel subsets;
- all 3 classifiers;
- fixed test set information;
- Monte Carlo stopping/replication summary;
- loss curves \(L(n)\) by classifier;
- best subset by \(n\) and classifier;
- final ranking at largest \(n\);
- interpolated \(N^\star\) table;
- compact Monte Carlo uncertainty summary;
- no empirical train loss, no theoretical loss, no Bayes error.

### 4. Gaussian-anchored Monte Carlo report

Suggested path:

```text
output/reports/occupancy_gaussian_anchored_monte_carlo_report.html
```

Must include:

- experiment arm: Gaussian-anchored Monte Carlo;
- estimated Gaussian parameters from standardized training data:
  - \(\hat{\mu}_0\);
  - \(\hat{\mu}_1\);
  - \(\hat{\Sigma}_0\);
  - \(\hat{\Sigma}_1\);
- mode and sample sizes;
- all 31 channel subsets;
- all 3 classifiers;
- fixed synthetic test set information;
- Monte Carlo stopping/replication summary;
- loss curves \(L(n)\) by classifier;
- best subset by \(n\) and classifier;
- final ranking at largest \(n\);
- interpolated \(N^\star\) table;
- compact Monte Carlo uncertainty summary;
- no empirical train loss, no theoretical loss, no Bayes error.

---

## \(N^\star\): interpolated threshold

Sprint 2 must not use only the discrete grid threshold.

For a baseline subset \(A\) and a cooperative subset \(B\), define:

\[
\Delta(n) = L_A(n) - L_B(n).
\]

The cooperative subset \(B\) beats \(A\) when:

\[
\Delta(n) > 0.
\]

If the sign change occurs between two evaluated sample sizes \(n_i\) and \(n_{i+1}\), estimate:

\[
\widetilde{N}^{\star}
=
n_i
+
\frac{0-\Delta(n_i)}
{\Delta(n_{i+1})-\Delta(n_i)}
(n_{i+1}-n_i).
\]

The report should show:

- `n_star_grid`: the first evaluated \(n\) where \(B\) beats \(A\);
- `n_star_interpolated`: the interpolated threshold;
- `delta_before`;
- `delta_after`;
- the two bracketing sample sizes if a crossing is found.

If \(B\) already beats \(A\) at the first evaluated sample size, the report can show `n_star_grid` equal to the first sample size and `n_star_interpolated` equal to the first sample size, with a note that the crossing is left-censored by the grid.

If no crossing occurs, both threshold fields should be null/blank and the report should state that no threshold was observed in the evaluated grid.

---

## Channel labels for occupancy

The existing Sprint 1 subset labels use generic labels such as `X1+X3`.

Sprint 2 reports should use readable channel labels when possible:

- `Temperature`;
- `Humidity`;
- `Light`;
- `CO2`;
- `HumidityRatio`.

For compact tables, abbreviated labels are acceptable:

- `Temp`;
- `Hum`;
- `Light`;
- `CO2`;
- `HumRatio`.

Internally, subsets should remain zero-based tuples.

---

## In scope

Sprint 2 includes:

- repository inspection;
- local occupancy dataset loader;
- robust parsing of the three raw files;
- dataset report;
- standardization using training data only;
- real-data sampler with balanced training samples;
- fixed real test set;
- deterministic real sampling;
- prefix-nested real sampling when possible;
- Gaussian-anchored model estimation from standardized training data;
- synthetic Gaussian sampler using the estimated parameters;
- reuse or safe generalization of the existing Monte Carlo engine;
- 31 subsets for five channels;
- 3 classifiers;
- interpolated \(N^\star\);
- four HTML reports;
- CLI or script execution in smoke mode;
- tests for the new loader, sampler, Gaussian anchoring, threshold interpolation, and smoke end-to-end run.

---

## Out of scope

Sprint 2 excludes:

- full-mode execution;
- performance validation in full mode;
- visual diagnostics in the SLACGS geometric style;
- GIF generation;
- ellipses;
- ellipsoids;
- hyperplanes;
- additional datasets;
- alternative random split protocol;
- temporal validation protocol;
- cost-aware channel analysis;
- feature acquisition cost;
- hyperparameter tuning;
- additional classifiers;
- non-Gaussian anchored synthetic models;
- Bayes error.

---

## Checkpoints

The agent must work checkpoint by checkpoint. At the end of each checkpoint, stop, summarize what happened in chat, report tests/checks, and request permission to continue.

### Checkpoint 0 — Repository reconnaissance and Sprint 2 integration plan

Goal: understand the current implementation before editing.

Tasks:

- inspect the current Sprint 1 modules:
  - `src/coinfosim/simulation/monte_carlo.py`;
  - `src/coinfosim/simulation/config.py`;
  - `src/coinfosim/results/analysis.py`;
  - `src/coinfosim/results/summary.py`;
  - `src/coinfosim/reports/sprint1.py`;
  - `src/coinfosim/samplers/gaussian.py`;
  - `src/coinfosim/models/gaussian.py`;
  - `src/coinfosim/simulation/subsets.py`;
  - CLI and/or scripts.
- identify what can be reused as-is;
- identify what must be generalized safely;
- verify whether occupancy raw files exist at `data/raw/occupancy/`;
- propose exact new files/modules;
- do not make large implementation changes.

Expected report:

- current architecture summary;
- proposed Sprint 2 module structure;
- whether the raw files are present;
- risks and design decisions;
- request permission for Checkpoint 1.

---

### Checkpoint 1 — Occupancy dataset loader and dataset report

Goal: load and summarize the dataset.

Tasks:

- create an occupancy loader module;
- parse the three raw files robustly;
- handle the index/timestamp columns correctly;
- expose:
  - raw train dataframe;
  - raw test dataframe;
  - feature matrix;
  - target vector;
  - channel names;
  - class labels;
- implement standardization fit on train only;
- apply standardization to train and test;
- compute dataset summary objects;
- generate `occupancy_dataset_report.html`.

Suggested files:

```text
src/coinfosim/datasets/occupancy.py
src/coinfosim/reports/occupancy_dataset.py
tests/test_occupancy_loader.py
```

Tests/checks:

- file presence check;
- row counts match expected totals;
- feature columns are correct;
- target column is correct;
- class labels are `{0, 1}`;
- no missing values in required columns;
- train-only standardization has approximately zero mean and unit std on train;
- transformed test uses train statistics;
- dataset report file is generated.

Stop after this checkpoint.

---

### Checkpoint 2 — RealDatasetSampler

Goal: support Monte Carlo sampling from a finite real dataset.

Tasks:

- implement `RealDatasetSampler`;
- it must expose:
  - `sample_train(n_per_class, replication_id)`;
  - `sample_test()`;
  - `model` or model-like metadata with `d` and class labels;
- training samples must be balanced by class;
- sampling must be deterministic in `(base_seed, class_label, replication_id)`;
- for each replication and class, sampling should be prefix-nested across `n`;
- fixed test set must be reused;
- fail clearly if requested `n_per_class` exceeds the available count in the minority class.

Suggested files:

```text
src/coinfosim/samplers/real.py
tests/test_real_sampler.py
```

Tests/checks:

- balanced output;
- deterministic output;
- prefix nesting;
- fixed test set;
- valid failure when `n_per_class` is too large;
- `d=5`;
- class labels are `{0, 1}`.

Stop after this checkpoint.

---

### Checkpoint 3 — Generalize/reuse Monte Carlo engine safely

Goal: make the existing Monte Carlo engine work for both Gaussian and real samplers.

Tasks:

- inspect whether the current `CooperativeMonteCarloSimulator` can accept the real sampler without large changes;
- if necessary, generalize typing/docstrings from "Sprint 1 Gaussian only" to "CoInfoSim cooperative Monte Carlo";
- preserve existing Sprint 1 behavior;
- do not break `GaussianClassConditionalSampler`;
- ensure the simulator can take:
  - a Gaussian sampler;
  - a real-data sampler.
- ensure the result metadata records:
  - experiment arm;
  - dataset/scenario name;
  - channel names if available;
  - fixed test size;
  - standardization information when relevant.

Suggested files:

```text
src/coinfosim/simulation/monte_carlo.py
tests/test_monte_carlo_with_real_sampler.py
```

Tests/checks:

- existing synthetic smoke test still passes;
- new real sampler smoke run with tiny config works;
- result has 31 subsets for occupancy;
- all three classifiers run;
- no train loss, theoretical loss, or Bayes error is produced.

Stop after this checkpoint.

---

### Checkpoint 4 — Gaussian anchored model builder

Goal: estimate a Gaussian simulation model from standardized real training data.

Tasks:

- implement a builder that estimates:
  - \(\hat{\mu}_0\);
  - \(\hat{\mu}_1\);
  - \(\hat{\Sigma}_0\);
  - \(\hat{\Sigma}_1\).
- build a `GaussianSimulationModel` from these estimates;
- ensure covariance matrices are positive definite or apply a small ridge regularization if necessary;
- record ridge value if used;
- create metadata linking the Gaussian model to the occupancy dataset.

Suggested files:

```text
src/coinfosim/scenarios/occupancy.py
tests/test_gaussian_anchored_occupancy.py
```

Tests/checks:

- means have shape `(5,)`;
- covariances have shape `(5, 5)`;
- class labels are `{0, 1}`;
- covariance matrices are symmetric positive definite after any required regularization;
- model can be sampled by `GaussianClassConditionalSampler`.

Stop after this checkpoint.

---

### Checkpoint 5 — Interpolated N-star

Goal: replace or extend discrete threshold analysis with interpolated thresholds.

Tasks:

- implement interpolated cooperative threshold analysis;
- keep discrete `n_star_grid`;
- add `n_star_interpolated`;
- add `delta_before`, `delta_after`, and bracketing sample sizes;
- handle:
  - crossing between two sample sizes;
  - already-winning at first sample size;
  - no crossing;
  - exact zero if it occurs.
- update standard threshold comparisons to be usable for both Sprint 1 and Sprint 2;
- avoid breaking existing report code.

Suggested files:

```text
src/coinfosim/results/analysis.py
tests/test_interpolated_thresholds.py
```

Tests/checks:

- known crossing example;
- no crossing example;
- already winning at first sample size;
- exact zero case;
- DataFrame includes both grid and interpolated fields.

Stop after this checkpoint.

---

### Checkpoint 6 — Monte Carlo reports for real and Gaussian arms

Goal: generate two detailed Monte Carlo reports.

Tasks:

- create a reusable report component for Monte Carlo results, or adapt the Sprint 1 report generator safely;
- generate:
  - `occupancy_real_monte_carlo_report.html`;
  - `occupancy_gaussian_anchored_monte_carlo_report.html`.
- reports must include:
  - mode;
  - sample sizes;
  - classifiers;
  - channel subsets;
  - stopping/replication table;
  - loss curves \(L(n)\) by classifier;
  - best subset by \(n\);
  - final ranking at largest \(n\);
  - interpolated \(N^\star\);
  - compact Monte Carlo uncertainty summary.
- use channel names rather than only `X1`, `X2`, etc., in reports when possible;
- keep layout reasonably consistent with the current Sprint 1 HTML report.

Suggested files:

```text
src/coinfosim/reports/monte_carlo.py
src/coinfosim/reports/occupancy_monte_carlo.py
tests/test_occupancy_monte_carlo_reports.py
```

Tests/checks:

- both report files are generated from small/smoke results;
- reports contain expected headings;
- reports include curves/images;
- reports include interpolated \(N^\star\);
- reports do not contain Bayes error, theoretical loss, or train loss.

Stop after this checkpoint.

---

### Checkpoint 7 — Scenario report and orchestration

Goal: run both arms and generate the top-level scenario report.

Tasks:

- implement an occupancy scenario runner/orchestrator;
- run real-data Monte Carlo in smoke mode;
- run Gaussian-anchored Monte Carlo in smoke mode;
- generate:
  - dataset report;
  - real-data Monte Carlo report;
  - Gaussian-anchored Monte Carlo report;
  - scenario report.
- the scenario report must link to the three detailed reports;
- the scenario report must compare:
  - best subsets by classifier;
  - final rankings at largest \(n\);
  - interpolated \(N^\star\) values;
  - observed differences between real and Gaussian-anchored arms.
- add a CLI command or script.

Preferred command:

```bash
coinfosim run-scenario occupancy --mode smoke
```

Acceptable fallback:

```bash
python scripts/run_occupancy_scenario.py --mode smoke
```

Suggested files:

```text
src/coinfosim/reports/occupancy_scenario.py
scripts/run_occupancy_scenario.py
```

Tests/checks:

- scenario runner creates all four HTML reports;
- scenario report links are valid relative links;
- smoke mode uses \(n=\{4,8,16,32\}\);
- fast and full are defined but not run;
- output paths are printed at the end.

Stop after this checkpoint.

---

### Checkpoint 8 — Final smoke validation and cleanup

Goal: validate the Sprint 2 deliverable without running full mode.

Tasks:

- run the smoke command/script end to end;
- run unit tests;
- run import/compile checks;
- inspect generated reports enough to confirm basic contents;
- update README or a short docs section with Sprint 2 usage;
- do not run full mode.

Required checks:

```bash
python -m compileall src
pytest
python scripts/run_occupancy_scenario.py --mode smoke
```

If CLI was implemented:

```bash
coinfosim run-scenario occupancy --mode smoke
```

Final expected artifacts:

```text
output/reports/occupancy_scenario_report.html
output/reports/occupancy_dataset_report.html
output/reports/occupancy_real_monte_carlo_report.html
output/reports/occupancy_gaussian_anchored_monte_carlo_report.html
```

Final report to user:

- files created/modified;
- tests run;
- smoke output paths;
- brief summary of real-data results;
- brief summary of Gaussian-anchored results;
- known limitations;
- clear note that full mode was not run.

---

## Acceptance criteria

Sprint 2 is complete when:

- the occupancy raw files are loaded locally;
- the dataset report is generated;
- the real-data Monte Carlo runs in smoke mode;
- the Gaussian-anchored Monte Carlo runs in smoke mode;
- the scenario report compares both arms;
- all four HTML reports exist;
- the code evaluates all 31 non-empty subsets;
- all three classifiers are evaluated;
- interpolated \(N^\star\) appears in reports;
- no train loss, theoretical loss, or Bayes error is introduced;
- fast and full modes are defined, but only smoke has been executed;
- the final checkpoint report confirms all of the above.
