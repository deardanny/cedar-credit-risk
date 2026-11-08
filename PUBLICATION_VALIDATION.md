# Publication validation — 2026-11-08

- Python 3.12 in an isolated environment installed from the recorded dependency versions.
- `python -m pytest -q`: **17 passed**. Two upstream test-client deprecation warnings; no failures.
- Existing tests verify API/model agreement, rejected invalid input, model corruption handling, aggregate counters, and grouped partition integrity.
- Original source files and saved model are preserved; README is now English with the Korean guide retained.
- Historical training metrics and Docker verification are from the original experiment described in `VERIFICATION.md`; training and Docker were not rerun for publication.
- The UCI dataset source and CC BY 4.0 attribution were checked. Original financial CSV is not bundled; saved model, public-dataset split indices, aggregate metrics and a fictitious API example are included.
