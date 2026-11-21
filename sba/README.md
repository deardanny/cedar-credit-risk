# SBA gross charge-off modeling & temporal data audit

**A retrospective lending-data study with a frozen model, explicit failure cases, and an audit of every model input.** Developed by Heedong (Daniel) Jang with AI-assisted tooling as an independent extension of Cedar. Published November 21, 2026; analysis through November 20.

[Portfolio case study](https://hdjang.com/projects/sba-credit-risk/) · [Korean analysis report (12-page PDF)](reports/Cedar_SBA_Analysis_2026-11-20.pdf) · [All 21 inputs](reports/feature_leakage_audit/all_21_feature_audit.csv)

## What the model estimates

For the studied U.S. SBA 7(a) non-revolving loans, Origination v2 estimates the probability of a recorded charge-off within **24 months of disbursement**, and the expected **gross charge-off amount in USD** over that horizon. It uses a separately calibrated histogram-gradient-boosting classifier and a Poisson expected-amount model with recency weights and aggregate amount calibration. The two outputs are estimated separately; the amount is not a recovered balance-based PD × LGD × EAD estimate.

The target is the gross amount in the September 2026 public snapshot, attributed to its recorded charge-off date. This is not first default, bankruptcy, cash-flow timing, recovered net lender loss, or a lifetime loss estimate. EXEMPT is not evidence of full repayment.

## Matched retrospective assessment

Same 8,632 loans approved and disbursed in 2024 Q3; 45 recorded charge-offs within 24 months. The baseline is the earlier paper-inspired competing-event model, evaluated at the same horizon. The historical outcomes had already been inspected: **this is not a fresh independent test**.

| Metric | Earlier model | Origination v2 | Observed |
|---|---:|---:|---:|
| Total gross charge-offs | $19,011,138 | $4,601,518 | $4,012,277 |
| Mean absolute amount error per loan | $2,651.80 | $991.57 | — |
| ROC-AUC (risk ordering) | 0.7091 | 0.6530 | — |
| Average charge-off probability | 1.293% | 0.404% | 0.521% |

Amount Tweedie deviance improved 21.9%, but risk ordering deteriorated. The 2020 validation cohort still has a 9.08× predicted/observed amount ratio with only six events. Final probability calibration used 36 events. A 500-replicate borrower bootstrap gives an amount-ratio interval of 0.75–2.10, conditional on the fitted model and this cohort. The full research gate fails and `approved_for_lending` is **false**.

Full selection, uncertainty, segments, and calibration: [results.json](reports/origination_v2/results.json). The selected model was chosen across 2018/2019/2020 before final 2024 scoring; that procedural freeze does not erase prior inspection of the outcomes.

## What the leakage audit found

The June/September 2026 snapshots do not provide immutable original application records. After excluding conflicting matches, 478,504 linked loans were compared. **59 loans changed at least one of eight used input columns**, including guarantee share, legal form, state, broad industry and dependent regional macro inputs. These comprise 26 fitting, one calibration, six final-assessment and 26 other project loans.

Term was already excluded from the current model. In the term-specific strict matching set, 2,409 of 478,008 loans changed term; the separate all-feature audit uses different matching coverage. A multiple-of-12 term pattern is a reason to investigate, not itself proof of leakage.

Outcome-poisoning and future-vintage controls passed; all 8,632 saved predictions reproduced. These checks do **not** establish that historical loan fields were available unchanged at approval. No observed revision is not proof of an original value, and a revision alone does not establish that an outcome caused it or quantify score inflation. The audit did **not** retrain the model: the frozen artifact still uses the affected columns. [Full audit](reports/feature_leakage_audit/README.md) · [Audit policy](reports/feature_leakage_audit/feature_audit_policy.json).

## Run the published model

Use Python 3.12 in a separate environment from the earlier UCI API. From this `sba` directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python predict.py examples/input.json
.venv/bin/python -m unittest discover -s tests -v
```

The bundled input is **fictional** and explicitly dated to the 2024 Q3 research context. No loan-level source data is required for this example. The command verifies the model hash and uses the original feature pipeline and published macro vintage table. It rejects present-day dates, unexpected fields and invalid numeric values. It produces no observed outcome or approval decision. Only load joblib artifacts you trust; a checksum establishes file identity, not trust in an unknown publisher.

## Included evidence and source

- Frozen [model](models/origination_v2_2024.joblib), [training/inference](src/origination_model.py), [experiment](scripts/origination_experiment.py), and [predeclared plan](docs/origination_v2_plan.md).
- Earlier [paper-inspired methods](scripts/paper_methods.py): discrete-time age-indicator hazards, competing recorded charge-off/paid-in-full events, historical macro paths, and held-out calibration. Specific paper algorithms such as Fine–Gray/CWGB were not reproduced; see [method attribution](docs/paper_plan.md).
- Aggregate evaluations and all-feature/term audit summaries, plus scripts for original-record verification. Loan-level prediction tables and raw borrower data are not redistributed in this package.
- Additional Census, FDIC and BLS data collection code and [coverage](reports/underwriting_public_v1/coverage.json). These are candidate sources, **not features already trained into v2**.
- The Korean PDF also documents the local Cedar integration, dated 10-company review and required validation work. Company operation evidence is not a model label or proof of bankruptcy. The local UI can compare predictions with historical outcomes and score reviewed consultation inputs; this publication does not turn it into a public lending service.

## Reproduction boundaries

Frozen inference and public fixture tests run with the included files. Full training and original-record audits require the exact June/September SBA snapshots and derived cohorts; these are not bundled. The historical source hashes and split counts remain in the experiment records. [Reproduction and data access](REPRODUCIBILITY.md) distinguishes shipped checks from full-data procedures. Original frozen research files are preserved byte-for-byte; publication-only entry points and documentation are listed in the publication manifest.

Real lending use needs application-time feature snapshots, monthly balances/delinquency/default/recovery/cost histories, an independently held-out future period and relevant lender population, and validated decision/monitoring procedures. The current study provides evidence about those gaps, not operational approval.
