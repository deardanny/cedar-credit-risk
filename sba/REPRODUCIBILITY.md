# Reproduction, provenance, and data access

## Included and executable

`predict.py` loads the exact frozen November 20 model, applies `src/features.py` and `src/origination_model.py`, and uses `MacroHistory` from the original paper-method code. The published macro table contains aggregate quarter-start vintage statistics, not borrower records. The example is fictional. Public tests exercise horizon boundaries, borrower/time splitting, strict input validation, frozen inference parity, outcome isolation, and future-macro isolation.

The model SHA-256 is `76ca2f6c6910a6cd2d32cb36a12c90af03f368f3b336519fd8a0f074c70f1edc`. The recorded fit has 329,706 loans / 1,743 charge-offs; calibration has 13,282 / 36. Final assessment has 8,632 / 45. `publication-manifest.json` records hashes for every released SBA file except itself.

The publication does not refit or change the model to hide the leakage findings. The original `freeze.json`, `model_freeze.json` and `results.json` retain the analysis hashes/timestamps. The public CLI is a new convenience entry point, separately tested; it is not the original Cedar application server.

## Full-data training and audit

The following original scripts are included as research source. They require local raw/derived files and are not claimed to work from a source-only checkout:

1. Obtain the exact input snapshots and dictionary; verify them against the recorded source hashes before regenerating derived data. Start with the [official SBA 7(a)/504 FOIA dataset](https://data.sba.gov/dataset/7-a-504-foia), the [SBA lending reports](https://www.sba.gov/partners/lenders/7a-loan-program/7a-504-summary-reports), and the vintage details in the audit JSON. The input `Archive.zip` was the June 30, 2026 snapshot, not an automatically refreshed feed. Public availability does not guarantee that the same historical file can still be downloaded.
2. The baseline preparation and public-feature stages are in `scripts/prepare.py`, `scripts/freeze_loss.py`, the `hour_*` scripts, and `scripts/public_features.py`. The experiment expects `data/processed/hour_short_development.pkl` and `data/processed/hour_medium_holdout.pkl`; the exact expected hashes are in `reports/origination_v2/freeze.json`. These derived cohorts are **not shipped**.
3. Earlier model training/evaluation: `scripts/paper_methods.py` and `scripts/paper_evaluate.py`. These regenerate `models/paper_{year}.joblib`, used as baselines by `scripts/origination_experiment.py`.
4. Origination v2 training: `scripts/origination_experiment.py`. Run only in an isolated copy: it intentionally refuses to overwrite an existing completed `results.json`. It uses 2018/2019/2020 selection and a 2024 retrospective assessment.
5. Original-record checks: `scripts/audit_term_snapshots.py`, `scripts/audit_feature_snapshots.py`, `scripts/audit_feature_lineage.py`, and `scripts/audit_feature_impact.py`. They require the original snapshot files plus local cohort/prediction files; row-level intermediate outputs remain local.

Full-data prediction reproduction and 118 independent changed-record verifications were completed in the original local audit. Their saved results are included as **historical evidence**, not portrayed as rerun from the public source-only package. The latest publication verification identifies the checks rerun for this release.

## External data

Macro vintages: [ALFRED](https://alfred.stlouisfed.org/). The original code constructs quarter-start information sets and excludes future vintages; same-day information availability remains an intraday ambiguity for some approvals. Other exploratory collectors use [Census BDS](https://www.census.gov/programs-surveys/bds.html), [BLS QCEW](https://www.bls.gov/cew/) and [FDIC BankFind](https://banks.data.fdic.gov/bankfind-suite/). Collected coverage is not evidence that every field was historically available or improves predictions.

The release contains code, fitted parameters, aggregate results, public-statistics lineage and the analysis PDF. It omits raw/derived borrower datasets, loan-level prediction exports, user-supplied requirements documents, credentials, consultation records and internal application databases. Source terms apply to their respective datasets; no license over third-party raw data is implied.
