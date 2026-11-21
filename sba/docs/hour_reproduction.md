# 1시간 추가 연구: 실행 및 재현

2026-11-19 02:08:55 ET부터 사용자가 요청한 1시간 동안 질문 없이 공개 자료 기반으로 실행했다. 실제 종료와 경과 시간은 `reports/hour/run_budget.json`에 기록한다. 온라인 게시·원본 업로드는 하지 않았다. 서버는 loopback 주소만 연다.

## 결과를 여는 방법

- 현재 로컬 주소: http://127.0.0.1:8813/hour.html
- 서버 재시작: `.venv/bin/python scripts/serve.py --port 8813 --no-open`
- 보고서: `reports/1시간_추가연구_결과.pdf` (6쪽), 같은 이름의 Markdown
- 결과의 수치 원본: `reports/hour/*results.json`, `long_robustness.json`
- 계획: `hour_plan.md`와 `hour_*_plan.md`; 1년에서 2년 변경은 `hour_short_amendment.md`

## 완료한 비교

| 실험 | 설정 수 | 평가와 결론 |
|---|---:|---|
| 직접 금액 Tweedie / Poisson | 24 | 기존 2023Q3 사후 비교, 오차 지표 개선과 총액 +42.8% 과대예측이 공존 |
| 24개월 확률 × 금액 | 12 | 고정 후 연 2024Q3 예약 시험, 총액 3.82배 과대예측 |
| 36개월 분기별 상각·상환 상태 | 2 | 기존 시험의 동일 부분 집단에서 이전 모델보다 나쁨 |
| 세부 지역·업종·기업 신용 통계 | 24 | 8~33개 입력, 순위 개선이 금액 정확성 개선으로 이어지지 않음 |
| 고정 72개월 분기별 상태 | 1 | 2020Q3 총액 +12.1%, 2018Q3 -10.5%, 2019Q3 +78.0%; 시점별 불안정 |

총 63개 설정이다. 연도별 검증 재학습과 최종 재학습을 설정 수에 중복 합산하지 않았다. 사후 혼합 비중 10개 비교는 별도다. 12개월 모델은 2020년 보정 사건 0건 때문에 중단했고 완료 수에 포함하지 않는다. `short_training.log`의 실패를 유지한다. 1년 예약 시험 정답은 추출하지 않았다.

## 현재 저장된 결과 검사

프로젝트 루트에서 다음을 실행한다. 기존 모델 재학습이나 시험 정답 재선정을 하지 않는다.

```sh
.venv/bin/python -m unittest discover -s tests -p test_hour.py -v
node tests/test_hour_curve.mjs
.venv/bin/python scripts/hour_validate.py
.venv/bin/python scripts/hour_final_audit.py
```

`node`는 실행 환경 PATH에 있어야 한다. 이번에는 Codex 기본 런타임의 Node 24를 사용했다. Python 패키지는 프로젝트 `.venv`와 `requirements.lock`을 따른다. PDF는 AppleGothic 폰트와 reportlab, matplotlib을 사용한다.

## 처음부터 실험하는 순서

이미 존재하는 2년 예약 시험을 지워서 다시 “처음 보는 시험”으로 만들지 않는다. 현재 작업 폴더의 고정 기록을 보존한다. 재학습이 필요하면 별도 복사본에서 원본 자료와 phase3 처리 결과를 준비한 후 아래 순서를 따른다. 지금의 시험 집단은 이미 열었으므로 어떤 재실행도 새로운 독립 시험이 아니다.

1. 기존 phase3 자료와 공식 2026-09-30 SBA 파일을 준비한다. `docs/phase3_reproduction.md` 참고.
2. `hour_fetch.py`: ALFRED 경기 통계 2023Q4~2025Q3 40개 파일. `hour_data.py`: 1년 개발 자료만 추출. 1년 모델 학습은 완료 경로에서 제외한다.
3. `hour_direct.py`: 직접 금액 24개 설정. `hour_audit.py`: paired bootstrap, 스냅샷 정정·승인시점 민감도. 공통 집단 12~72개월 관찰치는 `hour_final_audit.py`에서 재검산한다.
4. `hour_medium_data.py` (개발 자료) → `hour_medium.py` (선택·동결) → `hour_medium_data.py --holdout` → `hour_medium.py --evaluate`. 동결 파일은 시험 파일이 이미 있으면 생성하지 못하도록 보호한다.
5. `hour_hazard.py`: 완료된 분기만으로 36개월 상태 모델. `hour_long.py`: 같은 구조의 고정 72개월 모델. `hour_long_robustness.py`: 같은 설정의 2018·2019 추가 시점 점검.
6. `hour_credit_fetch.py`: 56개 분기의 3개 기업 신용 통계 과거 버전. 혼합 주기 응답은 ZIP일 수 있어 원본과 정규화 CSV를 모두 보관한다. 404 과거 버전 부재는 최신 자료로 대체하지 않는다. `hour_granular.py`: 세부 입력 복원·모델 비교. `hour_ensemble.py`: 과거 검증에서만 혼합 비중 선택. `hour_error_decomposition.py`: 금액 오차의 사건 빈도·금액 산술 분해.
7. `hour_validate.py`와 `hour_final_audit.py` → `hour_export.py` → `node tests/test_hour_curve.mjs` → 결과를 `web_validation.json`에 기록 → `hour_report.py`.
8. PDF 6쪽 렌더링 확인과 실제 브라우저의 계산·기간 조절·빈 입력·초기화 확인. 모델을 웹으로 내보낼 때 HistGradientBoosting 내부의 범주형 우선 열 순서를 보존한다. 합성 사례 48개로 Python 예측과 대조한다.

실행 스크립트는 새 결과를 같은 이름에 저장한다. 고정 시점의 원본·모델·계획 해시는 `medium_frozen.json`, 기존 `phase3/model_freeze_manifest.json`과 최종 `hour/artifact_checksums.json`에 있다. 모델 joblib 및 원본/행별 예측은 Git에서 제외되지만 로컬에 존재한다.

## 정답과 시점의 한계

- 총상각액은 단일 스냅샷의 금액을 사건일 기준 기간에 귀속한다. 해당 연도에 실제로 공개돼 있던 상각액 시계열을 복원한 것은 아니다.
- 상각일은 최초 부도일이 아니다. PIF가 찍힌 뒤에는 연구 모형에서 상환 상태로 빠지지만 EXEMPT는 정상으로 인증된 기록이 아니다.
- 신용 통계 CORBLACBS는 미국 은행의 회수 차감 상각률이다. 이는 거시 입력이며 SBA 목표 금액을 은행 순손실로 바꾸지 않는다.
- 6년 곡선은 승인 당시 정보와 경과 분기를 사용한다. 매월 새 재무·상환 정보를 반영하는 운영 모델은 아니다.
- 거절 신청자·매출·현금흐름·부채·월별 잔액·부도·회수가 없어서 은행 심사와 순손실 적합성을 확인하지 못했다. 추가 내부 자료 계약은 `internal_data_contract.md`에 있다.
- 시점별 결과와 신뢰 구간은 자료·모형 가정하의 자체 점검이다. 외부 독립 검증, 정책 공정성·적법성 승인 또는 실제 대출 승인 체계를 구현하지 않았다.

## 공식 출처

- [SBA FOIA](https://data.sba.gov/dataset/7a-504-foia)
- [ALFRED 과거 자료 안내](https://alfred.stlouisfed.org/help)
- [기업대출 연체율](https://fred.stlouisfed.org/series/DRBLACBS)
- [기업대출 상각률](https://fred.stlouisfed.org/series/CORBLACBS)
- [소기업 대출 기준 강화](https://fred.stlouisfed.org/series/DRTSCIS)
- [scikit-learn Tweedie 금액 모델 예제](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html)

완료 결과와 실패 기록을 모두 보존했다. 시험 총액에 맞춰 보정 계수를 다시 정하거나 큰 실제 손실을 지우지 않았다.
