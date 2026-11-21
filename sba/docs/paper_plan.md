# Paper methods experiment — fixed before new fits

Date: 2026-11-20. Scope: local/public-data research. Historical SBA snapshots and prior cohort outcomes have already been examined; this is a retrospective comparison, not a new independent validation or a lending approval model.

## Methods and attribution

1. Saavedra et al. (2024), DOI 10.1016/j.eswa.2024.123607: retain a joint competing-event structure (recorded charge-off / paid in full / still at risk). The existing model already implements this idea. Here we add a multinomial logistic discrete-time model with 24 age-quarter indicators, inspired also by Peng & Lessmann (2026), https://arxiv.org/html/2601.20533v2 §§3.3–3.5. This is NOT their CWGB / Fine–Gray implementation or full LMISO model.
2. Bellotti & Crook (2009), DOI 10.1057/jors.2008.130: add time-varying macro covariates to the quarterly risk set. Use ALFRED vintages at or before the interval START, never revised/current values. Apply the same HGB configuration as the existing model; compare approval-time static macro with interval-time macro to isolate the representation. This is an adaptation, not Cox replication. Forecast unknown macro paths using (a) persistence at approval and (b) a fixed eight-quarter half-life toward the preceding five-year vintage mean. Both are transparent assumptions, not observed future values or calibrated economic forecasts.
3. Peng & Lessmann (2026): test held-out isotonic probability calibration. Fit CO calibration and conditional PIF-given-no-CO calibration, then compose the three probabilities so they sum to one. Compare against joint multinomial log-odds calibration. Use separate borrowers and only fully completed intervals in the final 12 months before each virtual fit date. The paper's balance-based behavioural marker cannot be reproduced from these public records; no fabricated marker.

## Fixed comparison

Four fit dates: 2018-07-01, 2019-07-01, 2020-07-01, 2024-07-01. Test borrowers must have no pre-fit disbursement and both approval and disbursement within the corresponding Q3. Horizon 72 months for 2018/19/20, 24 months for 2024. Display annual sub-horizons as diagnostics. Snapshot 2026-09-30. All historical labels have previously been viewed, including the 2024 cohort.

Seven variants per cohort, report all without tuning on these outcomes:
- base HGB + joint calibration (same existing six-year configuration)
- age-indicator multinomial logistic + joint calibration
- age-indicator multinomial logistic + isotonic calibration
- approval-static macro HGB + joint calibration
- interval-dynamic macro HGB + joint calibration, persistence forecast
- interval-dynamic macro HGB + isotonic calibration, persistence forecast
- interval-dynamic macro HGB + joint calibration, mean-reverting forecast

Keep the existing Gamma severity specification and use its training-only fitted model across variants within a cohort. Re-estimate only aggregate severity scale on each variant's calibration predictions. No clipping charge-offs to approval amounts. Model gross amounts, not EAD or net LGD. No class-balancing or synthetic events. For logistic computation, retain all CO/PIF fit intervals and a seeded 15% simple random sample of no-event intervals, with inverse inclusion probability weights (1 / .15). HGB uses all fit intervals. Logistic regularization C=.1, lbfgs max_iter=600, tolerance 1e-5. HGB 80 trees, 7 leaves, learning rate .06, min leaf 200, L2 20. Fit/cal borrower partition remains hash modulo five.

## Outcomes and decision rule

Primary: cohort-level predicted/actual gross amount, Tweedie deviance (power 1.5; dollar/1000), calibration of charge-off probability. Secondary: AUC, average precision, Brier, dollar MAE and RMSE, capture by top predicted decile. Paired borrower bootstrap (500 replicates) for deviance improvement and aggregate ratio. Improvement for adoption requires all three six-year cohorts to have gross ratio within [.8,1.2], mean absolute log ratio lower than base, and pooled deviance improvement with 95% bootstrap interval above zero. Also inspect 2024 separately; do not mix different horizons. If gates fail, retain research status and show failures. These are research gates, not bank/regulatory acceptance criteria.

## Safeguards

Audit macro lookup dates, completed intervals, borrower separation, event boundaries, probability mass and monotone cumulative forecasts. Test forecast invariance to changing/deleting future macro vintages. Save plan/code/source hashes before fitting and model hashes before cohort scoring. Retain existing frozen files untouched. All source-vintage limitations of the SBA snapshot remain: charge-off amounts/current fields may be revised and original historical availability is not fully reconstructible. PIF includes all recorded payoff, not identified early prepayment. Charge-off is later than initial default. No claims of true lifetime ECL or causal macro effects.
