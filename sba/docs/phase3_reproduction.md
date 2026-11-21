# 금액 모델 실행 순서

기존 실행 결과를 검증할 때는 모델을 다시 학습하지 않고 아래를 실행한다. `.venv`, 원본·공개 자료 캐시, 학습 모델이 있는 현재 로컬 프로젝트를 기준으로 한다.

```sh
.venv/bin/python scripts/validate_loss.py
.venv/bin/python scripts/export_loss.py
node tests/test_loss.mjs > reports/phase3/web_validation.json
.venv/bin/python scripts/report_loss.py
```

최초 실행의 계산 순서는 아래와 같다. 원본 `data/processed/analytic.pkl`은 기존 `scripts/run_all.py` 과정에서 만든다. 새 SBA 파일을 먼저 읽거나 결과를 확인한 재실행은 새로운 독립 시험이 아니다. 이 결과를 보고 모델을 조정하면 이후 별도 시간 집단이 필요하다.

```sh
.venv/bin/python scripts/fetch_public.py
.venv/bin/python scripts/enrich_public.py
.venv/bin/python scripts/train_loss.py
.venv/bin/python scripts/freeze_loss.py
.venv/bin/python scripts/fetch_public.py --holdout
.venv/bin/python scripts/enrich_public.py --new-test
.venv/bin/python scripts/train_loss.py --evaluate
.venv/bin/python scripts/export_loss.py
.venv/bin/python scripts/validate_loss.py
node tests/test_loss.mjs > reports/phase3/web_validation.json
.venv/bin/python scripts/report_loss.py
```

`freeze_loss.py`는 기존 고정 기록을 덮어쓰거나, 새 시험 파일이 이미 있는 상태를 처음 보는 시험이라고 선언하지 않는다. 이번 첫 실행의 고정 기록은 새 시험 다운로드 전에 동일 로직으로 저장했다. 이후 도우미 파일로 분리한 것이며 학습·선정·전처리 코드와 모델은 변경하지 않았다.

이번 고정 설계와 비교 지표는 `phase3_plan.md`, 데이터별 URL·해시는 `reports/phase3/*source_manifest.json`, 모델·코드 해시는 `model_freeze_manifest.json`에 있다. 모델 및 원본/분석 데이터는 용량과 차주 자료 노출을 줄이기 위해 Git에 넣지 않았다. 웹에는 집계 통계, 파라미터, 합성 검증 사례만 포함된다.

원본 공개 파일을 다시 내려받으면 공급자의 같은 URL 내용이 수정될 수 있다. 재현에는 저장된 해시와 일치하는 파일을 사용해야 한다. ALFRED는 과거 버전을 사용하지만 SBA 원자료와 CBP의 모든 과거 정정 이력을 복원한 것은 아니다.
