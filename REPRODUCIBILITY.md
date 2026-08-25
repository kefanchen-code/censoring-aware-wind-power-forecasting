# Reproducibility map

Commands are expected to be run from the repository root after creating the
environment and installing the package in editable mode.

## Verification tiers

### Tier 1: code-level verification

```bash
python -m unittest discover -s tests -v
python scripts/run_pap_benchmark.py list-models
```

The test suite checks the censored losses, identical initialization of matched
models, information-access gates, whole-segment splitting, forecast validity,
score calculations, contamination logic, exact sign-flip inference, and S0
construction rules.

### Tier 2: regenerate paper figures from archived summaries

```bash
python paper_assets/figures/make_figures_paper1.py
```

This reads `reference_results/` and writes all supported formats to `figures/`.
It does not retrain a model.

### Tier 3: main benchmark from public data

```bash
python scripts/download_altahullion.py
python scripts/10_altahullion_audit.py
python scripts/run_pap_benchmark.py \
  --config configs/pap_benchmark_semisyn.json run --smoke --force
python scripts/run_pap_benchmark.py \
  --config configs/pap_benchmark_semisyn.json run
python scripts/run_pap_benchmark.py \
  --config configs/pap_benchmark_s0_control.json run
```

To rescore saved predictions without refitting:

```bash
python scripts/run_pap_benchmark.py \
  --config configs/pap_benchmark_semisyn.json compare
```

### Tier 4: Hill of Towie reconstruction and real release events

The Hill of Towie raw-to-derived chain is:

```bash
python scripts/20_hot_fastlog_to_1min.py --selftest
python scripts/20_hot_fastlog_to_1min.py
python scripts/21_hot_truth_channel.py
python scripts/13_hot_contamination_did.py
python scripts/14_hot_truth_channel_bias.py
```

The truth-channel manifest records input paths, route counts, fitted power-
curve points, chronological clean-window hold-out MAE, and output SHA-256
hashes. Script 21 fails closed if the complete public v2 inputs do not reproduce
the archived route composition.

Verify the reconstructed real release-event protocol and unchanged event list:

```bash
python scripts/63_release_event_validation.py --verify-protocol-only
```

After preparing the Altahullion data and running script 10, execute a model
smoke test with `--smoke` or the full v1.3 run without it. New outputs go to
`results/release_event_validation_v1_3/`; the archived v1.2 result is not
overwritten.

## Manuscript result mapping

| Manuscript evidence | Main input/configuration | Analysis or output |
| --- | --- | --- |
| Scenario definitions and binding rates | `scripts/10_altahullion_audit.py` | `results/altahullion_audit/audit_summary.json` |
| Main WIS table | `configs/pap_benchmark_semisyn.json` | `results/pap_benchmark_semisyn/metrics_summary.csv` |
| Censored-target calibration | same main run | `metrics_stratified.csv` |
| Matched statistical comparisons | same main run | `comparisons.json` |
| WIS decomposition | `scripts/52_wis_decomposition.py` | `wis_components*.csv` |
| S0 negative control | `configs/pap_benchmark_s0_control.json` | S0 `comparisons.json` |
| Input-contamination stress test | contamination configurations and `scripts/54_contamination_robustness.py` | `contamination_robustness/` |
| Grid sensitivity | 53/106/212-bin configurations and `scripts/68_bins_sensitivity_table.py` | `bins_sensitivity/` |
| Realistic S6 scenario | `scripts/62_real_command_stats.py`, `scripts/64_s6_realistic_scenario.py` | `pap_benchmark_semisyn_s6/` |
| Direction across 13 turbines | `scripts/66_multi_turbine_data_prep.py`, `scripts/67_multi_turbine_benchmark.py` | `multi_turbine_validation/runs/` |
| LOTO/LOFO tests | `scripts/70_loto_lofo.py` | `loto_lofo/loto_lofo_results.csv` |
| Hill of Towie fast-log conversion | `scripts/20_hot_fastlog_to_1min.py` | `data/hill_of_towie/converted/` |
| Rebuilt T11 truth channel | `scripts/21_hot_truth_channel.py` | `results/hot_truth_channel/` |
| Real release-event validation | `protocols/release_event_protocol_v1_3.md`, `scripts/63_release_event_validation.py` | `release_event_validation_v1_3/` |
| Quantitative paper figures | `paper_assets/figures/make_figures_paper1.py` | `figures/` |

## Frozen protocol

- Forecast history: 60 minutes
- Forecast horizon: 15 minutes
- Split: chronological, whole-segment 70%/15%/15%
- Support: `[0, 1.06]` p.u., 106 bins for the main CL-PMF model
- Quantiles: 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95
- Seeds: 42, 123, 256, 789, 1024
- S0 equivalence margin: WIS 0.005

The longer original Chinese protocol note is retained as
`PROTOCOL_ORIGINAL_ZH.md` for provenance. The JSON configurations are the
machine-readable source of truth.

The separately versioned real release-event protocol is documented in
`protocols/release_event_protocol_v1_3.md`. The legacy v1.2 document is missing;
only its historical SHA-256 remains. The v1.3 document is therefore labelled a
reconstructed replacement and receives its own hash and output directory.
