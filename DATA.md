# Data sources

No raw research data are tracked in this repository. Downloaded files are
stored under `data/`, which is excluded from Git.

## Altahullion wind farm

- Record: [Altahullion wind farm open dataset](https://zenodo.org/records/19948235)
- DOI: `10.5281/zenodo.19948235`
- Version: v1
- License: CC BY 4.0
- Files used: `turbine_data.zip`
- Published archive MD5: `16c7c2e043590444d6584f7f27d6c967`

Run `python scripts/download_altahullion.py` to download, verify, and extract
the archive. The main benchmark uses:

```text
data/turbine_data/fl_df_ALTA2_T11_20250904_to_20260228.parquet
data/turbine_data/scada_df_ALTA2_20250904_20260301.parquet
```

The deterministic scenario-construction script writes S0--S5 under
`results/altahullion_audit/`. `EVENT_SEED=42` fixes the common event mask for
S1--S4 and `CAP_SEED=142` fixes the event-wise random cap depths in S4.

## Hill of Towie wind farm

- Record: [Hill of Towie wind farm open dataset](https://zenodo.org/records/20204946)
- DOI: `10.5281/zenodo.20204946`
- Version: 2.0.0
- License: CC BY 4.0
- Files used: `2026.zip` and selected material derived from
  `turbine_fastlog.zip`
- Published `2026.zip` MD5: `42a2264187912ca63214bacfbc8d4905`

The 10-minute contamination analysis in
`scripts/13_hot_contamination_did.py` reads
`data/hill_of_towie/2026.zip`. The multi-turbine validation uses prepared
one-minute files under `data/hill_of_towie/converted/`.

The original fast-log-to-Parquet conversion step is not yet present in this
release candidate. The compact reference outputs are included so the reported
tables and quantitative figures can be checked, but this missing preparation
step must be restored before the repository is made public.

## Attribution

Both datasets were released by Renewable Energy Systems on behalf of The
Renewables Infrastructure Group. Users must retain the dataset attribution and
comply with the CC BY 4.0 terms. The repository's eventual software license
will apply only to the source code, not to these datasets.
