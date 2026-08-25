# Censoring-aware wind-power forecasting

Code and reproducibility materials for the manuscript **“Censoring-aware
probabilistic forecasting of single-turbine potential available power in
directly coupled wind-to-hydrogen microgrids.”**

This repository implements probabilistic forecasting of potential available
power (PAP) when turbine power observations are right-censored by control caps.
The primary method is a discrete probability-mass model trained with a
censoring-aware survival likelihood (CL-PMF). The repository also contains the
matched point-label, deletion, reconstruction, quantile, censored-quantile,
Tobit, persistence, climatology, and full-label reference models used in the
paper.

> **Release status:** private pre-submission release candidate. The scientific
> code and core tests are complete, but the public release is blocked until the
> items in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) are resolved.

## Repository contents

```text
configs/             Frozen JSON experiment configurations
data/                Local data directory (downloaded data are ignored by Git)
paper_assets/        Quantitative figure-generation code
protocols/           Auditable real-data validation protocol documents
reference_results/   Compact outputs used to verify paper figures and tables
scripts/             Data preparation, benchmark, robustness, and validation entry points
src/                 Reusable forecasting and evaluation package
tests/               Unit and integration tests
```

The wind-to-hydrogen control and value-analysis code from the follow-on study is
not part of this Paper 1 repository.

## Environment

The manuscript results were generated on Windows with Python 3.9.7, PyTorch
2.8.0+cu128, and an NVIDIA GeForce RTX 4060 Laptop GPU. Exact package versions
are recorded in [environment.yml](environment.yml) and
[requirements-lock.txt](requirements-lock.txt).

```bash
conda env create -f environment.yml
conda activate censoring-aware-wind-power
python -m pip install -e . --no-deps
```

CPU execution is supported by the benchmark, but full retraining is
substantially slower than the recorded GPU workflow.

## Quick verification

Run the complete code-level test suite:

```bash
python -m unittest discover -s tests -v
```

Generate the quantitative manuscript figures from the compact archived
outputs, without downloading data or retraining models:

```bash
python paper_assets/figures/make_figures_paper1.py
```

The generated SVG, PDF, and PNG files are written to `figures/`.

## Reproduce the main benchmark

Download and verify the Altahullion data archive (47.4 MB), then extract it to
`data/turbine_data/`:

```bash
python scripts/download_altahullion.py
```

Construct the deterministic S0--S5 semi-synthetic scenarios:

```bash
python scripts/10_altahullion_audit.py
```

Run a one-scenario, one-seed, one-epoch smoke test:

```bash
python scripts/run_pap_benchmark.py \
  --config configs/pap_benchmark_semisyn.json run --smoke --force
```

Run the frozen full benchmark and the uncensored negative control:

```bash
python scripts/run_pap_benchmark.py \
  --config configs/pap_benchmark_semisyn.json run
python scripts/run_pap_benchmark.py \
  --config configs/pap_benchmark_s0_control.json run
```

The main run comprises 11 configured models, five censoring scenarios, and five
random seeds for stochastic models. A result is valid for final comparison only
when `benchmark_status.json` reports both `complete: true` and
`valid_for_final_comparison: true`, with empty `missing_artifacts` and
`nonconverged_runs` fields.

Detailed result-to-command mappings are provided in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Data sources and licensing are
documented in [DATA.md](DATA.md).

## Optional real-data reconstruction audits

After downloading both Hill of Towie archives, rebuild the one-minute fast-log
inputs and the historical T11 truth-channel analysis with:

```bash
python scripts/20_hot_fastlog_to_1min.py --selftest
python scripts/20_hot_fastlog_to_1min.py
python scripts/21_hot_truth_channel.py
python scripts/14_hot_truth_channel_bias.py
```

The real Altahullion release-event protocol can be verified without retraining:

```bash
python scripts/63_release_event_validation.py --verify-protocol-only
```

The unavailable legacy v1.2 protocol source is not imitated. Its historical
hash is retained, while
[`protocols/release_event_protocol_v1_3.md`](protocols/release_event_protocol_v1_3.md)
is a separately hashed reconstruction from the executable code and frozen
122-event artifact.

## Information-separation safeguards

- Ordinary models receive only observed labels and issue-time inputs.
- Only adapters declaring `requires_latent_truth = True` receive the latent PAP
  labels, and their artifacts are marked as oracle results.
- Prediction interfaces contain no truth arrays.
- Time splits preserve complete segments, and forecast windows cannot cross a
  segment or split boundary.
- Seeds, deterministic-algorithm settings, input hashes, protocol identifiers,
  and information-access flags are written to run artifacts.

These contracts are exercised directly by the test suite.

## Citation and license

Citation metadata are available in [CITATION.cff](CITATION.cff). A software DOI
will be added after the submission release is archived in Zenodo.

No open-source license has yet been selected. Until a `LICENSE` file is added,
copyright permission to reuse the source code is not granted. The two source
datasets have their own CC BY 4.0 licenses; see [DATA.md](DATA.md).
