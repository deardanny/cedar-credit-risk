# Origination v2: fixed retrospective experiment

Created before fitting on 2026-11-20. This experiment does not create a fresh independent test: all historical outcomes, including 2024, have been examined in earlier work.

## Problem and hypothesis

The existing competing-event model is calibrated on completed quarters of loans of mixed ages, then compounds quarter forecasts for new loans. Its 2024 two-year expected event count is about 2.48 times observed and its predicted dollars per expected event about 1.91 times observed. We will test direct prediction of the same 24-month origination outcomes, with borrower-held-out calibration on fully completed origination cohorts, nonlinear severity and recency weighting. No target-cohort adjustment factor is allowed.

## Data and chronology

Keep all original snapshots, models and reports unchanged. Read hour_short_development, hour_medium_holdout, and the existing ALFRED macro lineage. Recompute 24-month events/amounts from dates; do not reuse stored y. Each training/calibration loan must have its complete 24-month outcome end strictly before the virtual cutoff. Use borrower hash modulo five: 0 for calibration, others for fit; calibration covers the most recent two years of completed outcome windows. The test's borrowers have no pre-cutoff disbursement. Approval and disbursement must both be in Q3 of the chosen year.

## Fixed candidates and selection

Compare direct_base, direct_macro, recent_macro, recent_poisson as defined in src/origination_model.py. All use HGB probability and held-out sigmoid calibration. Hurdle candidates multiply calibrated probability by nonlinear Gamma severity; direct Poisson predicts exposure-weighted gross/approval ratio. Recency weighting has a fixed four-year half-life. Calibration-only amount scaling is allowed. No class balancing, SMOTE, clipping observed charge-offs to approval amounts, tuning on 2024 or fabricated company financials.

Base uses the existing eight business/loan features. Macro adds eleven approval-date macro features plus two loan-structure ratios computed from existing fields. Macro series are reconstructed through the existing as-of lookup. Do not inject future macro vintages.

Select the candidate with the lowest mean log ratio of Tweedie (power 1.5) deviance to the existing dynamic_macro baseline on 2018/2019/2020 Q3, all at exactly 24 months. Save the selection before fitting/scoring 2024. Freeze the selected model hash before scoring the 8,632-loan 2024 comparison. Report all candidates and years, final AUC/AP/Brier/log loss, dollar MAE/RMSE/deviance, total ratio, top-decile capture, low-event segments, and 500-replicate paired borrower-bootstrap uncertainty. The bootstrap conditions on the fitted models and does not establish future transportability.

## Release and limits

Research gates: 2024 gross ratio and probability ratio within [0.75,1.25], AUC not below baseline, positive lower 95% bound for paired deviance improvement. These are experiment gates, not bank policy or regulatory approval. Even if all pass, keep approved_for_lending=false. Integrate the selected result transparently as a two-year research challenger, showing baseline comparison and every failed gate; do not describe aggregate fit as individual loan accuracy.

Public approved-loan selection, gross charge-off vs default/net loss, missing balances/recoveries/cash flows/rejected applicants, revised snapshot fields and reporting latency remain unresolved. Operational use needs lender-specific and independently validated data, current calibration, performance/fairness monitoring, auditability, controls and human review. No actual approval/denial is produced here.
