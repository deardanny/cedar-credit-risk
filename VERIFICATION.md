# 검증 기록

2026-09-14 실행. 모델 버전: `cedar-v0-2e744ab0dfc7`.

## 학습·평가

- 공식 UCI CSV 30,000건, 정답 0=23,364 / 1=6,636, 결측값 0, 중복 ID 0.
- 고정 CSV SHA-256: `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0`.
- 학습 17,998 / 확률 보정 2,995 / 모델 선택 3,007 / 최종 평가 6,000.
- 동일 금융 프로필의 구간 간 중복 0.
- validation log loss로 `hist_gradient_boosting` 선택 후 test 평가.
- test ROC-AUC 0.787418, AP 0.561440, log loss 0.424444, Brier 0.133212.
- 단순 평균 발생률 기준 모델의 test log loss 0.528590.
- 저장 후 다시 읽은 모델의 **전체 test 6,000건 확률**이 저장 전과 절대 오차 1e-12 이내로 일치.
- manifest의 학습 코드 지문이 현재 모든 기록된 원본 파일과 일치.

## 자동 테스트

`python -m pytest -q`: **17 passed**, 1.94초.

예측/점수/API 일치, 입력 범위·개수·정수 범주 검증, 누락 값 거절, 음수 청구 잔액 허용,
최근 월부터의 순서, 간단한 운영 지표, 범위 이탈 알림, 손상/누락 모델 로드 거절,
그룹 간 분리와 seed 재현성, test 이전 모델 선택 규칙을 확인했다.

두 개의 비실패 경고: 설치된 Starlette/httpx 테스트 클라이언트와 AnyIO alias의 향후 지원 변경 안내.
기능 실패는 없었고, 실제 HTTP 검증은 아래에서 별도로 수행했다.

## Docker 로컬 배포

- 이미지: `cedar-baseline:local`.
- Docker engine: 29.7.2, Linux arm64 컨테이너.
- 기반 이미지: `python:3.12-slim`, 확인된 digest `sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea`.
- 빌드된 이미지 manifest list: `sha256:14e012ef4009878a5f8f5a92948256db9279e3f9453e3345ee09e6ee8d321672`.
- 일반 사용자(UID 10001) 실행, 로컬 `127.0.0.1:8010`에만 포트 공개.
- `/health`: ready, Docker healthcheck: healthy.
- 가상 예시의 컨테이너 확률 0.09134012091643216이 호스트 모델과 절대 오차 1e-12 이내로 일치.
- 추가 순차 요청 30회에서 같은 결과. 잘못된 요청은 HTTP 422.
- 로컬 warm 순차 호출 p50 5.79ms / p95 7.03ms. 동시 사용자 부하나 운영 성능을 입증하는 값은 아니다.
- `/metrics`에서 성공·실패·요청 수 및 평균 지연 기록 확인.
- 원본 응답 기록: `artifacts/deployment_verification.json`.

외부 클라우드에 게시하지 않았다. 실제 고객의 상환 결과를 수집하거나 운영 중 성능을 평가한 단계도 아니다.

## 실행 상태와 종료

검증 후 컨테이너를 로컬 API 체험용으로 실행해 두었다.

- API 설명: http://127.0.0.1:8010/docs
- 컨테이너 이름: `cedar-baseline-20260914`
- 종료: `docker stop cedar-baseline-20260914` (종료하면 컨테이너는 자동 제거, 이미지·모델 파일은 유지)
- 다시 실행: README의 Docker 실행 명령 사용.
