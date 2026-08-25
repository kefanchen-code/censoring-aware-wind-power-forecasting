# Real release-event validation protocol v1.3 (reconstructed replacement)

## Status and provenance

This document is the auditable replacement for the unavailable v1.2 source
document formerly located at `writing/22_release_event_protocol.md`.  The
legacy freeze records its SHA-256 as
`8aa3bca2cce62a559cb55d803d18beb41ea26535473976e654c3cce40772d1bf`,
but the source bytes were not retained and cannot be recovered from the hash.
Consequently, this file is **not** presented as a byte-for-byte recovery.

Version 1.3 was reconstructed from the executable validation code, the
archived v1.2 result summary, the canonical data-audit code, and the unchanged
122-event artifact.  Any v1.3 run is written to a new result directory and
receives a new protocol hash.  The historical v1.2 freeze and results remain
unchanged under `reference_results/release_event_validation/`.

## Scientific scope

This is an observable-only real-data validation on Altahullion turbine T11.
No latent potential available power (PAP) is asserted for curtailed minutes.
Models are trained with free rows as point observations and active curtailed
rows as right-censored observations.  At release events, forecasts are scored
against subsequently measured power; this endpoint is not called latent PAP.

## Frozen data and event artifact

- Rated power: 1330 kW.
- Natural sampling clock: one minute.
- Event file: `results/altahullion_audit/qualified_release_events.csv`.
- Frozen event count: 122.
- Legacy v1.2 event-list SHA-256 over the original CRLF bytes:
  `e351c076d155dea95d94047b2c5a0bc365c7a680b9a859565695df36387d63fe`.
- Cross-platform canonical SHA-256 after normalizing line endings to LF:
  `a7f4c8e896f05453b323071659c2079ba1600dd6ca7083c37415ecc7170723de`.
- The validation script must verify the row count and canonical SHA-256 before
  training, prediction, or scoring. The legacy byte hash remains recorded for
  continuity with v1.2 and the reference CSV is stored without Git text
  normalization.

The event list is produced by `scripts/10_altahullion_audit.py`.  A candidate
release is a transition from active curtailment to no active curtailment.  It
is retained when all of the following pre-specified quality conditions hold:

1. the preceding 15 minutes are curtailed for at least 80% of samples;
2. the following 10 minutes are curtailed for at most 20% of samples;
3. the turbine is active (power above 60 kW) for at least 80% of those
   following 10 minutes;
4. the 25-minute screening window contains no clock gap; and
5. mean wind speed in the preceding 15 minutes is at least 6 m/s.

Neither post-release power magnitude, model forecasts, nor model scores enter
event inclusion.  The derived flag `post_exceeds_pre_by_10pct` is retained
only for report-only sensitivity description; it is never an eligibility
criterion.

## Observable labels

The canonical audit classifies each minute using measured power, PowerRef,
PowerRed, wind speed, and clock continuity.

- A free (`U`) minute is active, has no `PowerRed` flag, has no gap, and has
  `PowerRef >= 0.95 x 1330 kW`.
- A censored minute is any active minute with `PowerRed > 0` and finite
  `PowerRef`, including strict `R` and less certain `I/X` audit labels.
- The observed training label is measured power divided by 1330 kW.
- The cap is `PowerRef / 1330` on censored rows and 1.05 p.u. on free rows.
- Other minutes are unusable and break a contiguous segment.

The strict-`R` audit label is retained only as a sensitivity marker.  It does
not replace the primary active-curtailment definition.

## Time split and forecasting inputs

Usable minutes are divided into maximal contiguous segments.  Complete
segments are assigned chronologically by cumulative row midpoint:

- training: first 60% of usable rows;
- validation: next 15%;
- test: remaining 25%.

No segment may cross a split.  Wind speed is standardized using training-side
rows only.  Every model sees a 60-minute history with four channels:

1. observed power in p.u.;
2. training-standardized wind speed;
3. cap in p.u.; and
4. censoring indicator.

The model training target is 15 minutes ahead.  Windows are valid only when
the full history and target remain inside one usable segment.

## Models and deterministic settings

The primary run uses seeds 42, 123, and 256 and the following models:

- persistence and climatology reporting references;
- B1 (free-only learned model);
- B4 (point-label model with control inputs);
- B6 (censored-likelihood model); and
- CLQR (censored quantile-regression model).

The support is [0, 1.06] p.u. with 106 bins for discrete models.  The reported
quantile levels are 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, and 0.95.
The executable hyperparameters in `scripts/63_release_event_validation.py`
are part of this protocol and must not be edited without a version bump.

## Release-event prediction and endpoint scoring

Only frozen events at or after the first test target time are considered.
Each event must have the preceding 60 exact one-minute timestamps, and all 60
history rows must be usable with finite observed power.  The release minute
itself is not used as an input; the feature stack ends one minute earlier.

The same pre-release probabilistic forecast is compared with measured power
at +1, +5, and +15 minutes after release.  The per-event score is mean pinball
loss over the nine frozen quantile levels.  Missing endpoint measurements are
excluded pairwise for that horizon.  The historical v1.2 accounting was 122
frozen events, 11 in the test period, and 10 with a valid 60-minute history;
v1.3 must report its event accounting rather than silently assuming it.

## Pre-specified comparisons and inference

The three primary comparison families are:

1. censored likelihood: B4 (reference) versus B6 (challenger);
2. censored quantiles: B4 versus CLQR; and
3. censored likelihood versus free-only: B1 versus B6.

Scores are first averaged across seeds within event and model.  Paired score
differences are `score(reference) - score(challenger)`, so a positive value
favours the challenger.  Calendar date is the dependence cluster.

- With at most 16 date clusters, use the exact clustered sign-flip test.
- With more than 16 clusters, use a date-cluster bootstrap (5000 draws) and
  clustered sign-flip test (100000 draws), seed 20260726.
- Apply Holm adjustment across the three endpoint horizons separately within
  each primary comparison family, with family-wise alpha 0.05.
- Comparisons with persistence and climatology are descriptive only and do
  not enter a confirmatory Holm family.

## Versioning rule

The script freezes this document's SHA-256 together with the unchanged event
artifact before computing a score.  Editing this document, the event list,
the model set, the split, endpoints, scoring rule, or inference plan requires
a new protocol version and a new output directory.  Historical freezes must
never be overwritten.
