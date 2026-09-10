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
- Published `turbine_fastlog.zip` MD5: `483acf28d7a901faf5eac611412e1a1f`

The 10-minute contamination analysis in
`scripts/13_hot_contamination_did.py` reads
`data/hill_of_towie/2026.zip`. The multi-turbine validation uses prepared
one-minute files under `data/hill_of_towie/converted/`.

Rebuild the one-minute files directly from the 11.9 GB fast-log archive:

```bash
python scripts/20_hot_fastlog_to_1min.py --selftest
python scripts/20_hot_fastlog_to_1min.py --list
python scripts/20_hot_fastlog_to_1min.py
```

The converter implements the recovered archived contract: causal integer-
second last-observation-carried-forward sampling, followed by one-minute means
and one-minute power minima/maxima. A fully missing turbine-day remains a
timestamp gap. A two-day T01 reconstruction (including the UTC day boundary)
was checked element by element against the archived Parquet input with zero
numerical difference.

Rebuild the historical T11 truth-channel inputs after conversion:

```bash
python scripts/21_hot_truth_channel.py
```

The replacement script verifies the archived route composition before writing
its outputs. The one-minute file contains 96,843 actual-route rows, 47,244
power-curve rows, and 28,713 unavailable rows; the corresponding ten-minute
counts are 10,294, 3,947, and 3,040. The historical neighbour route had zero
coverage and is recorded as unavailable rather than filled using a new,
unverifiable assumption.

## Attribution

Both datasets were released by Renewable Energy Systems on behalf of The
Renewables Infrastructure Group. Users must retain the dataset attribution and
comply with the CC BY 4.0 terms. The repository's software license
applies only to the source code, not to these datasets or to the tracked
derived data artifacts. See `LICENSE-DATA` and `THIRD_PARTY_NOTICES.md`.
