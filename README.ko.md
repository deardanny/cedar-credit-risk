# Cedar — 기본 위험 예측 모델

**차주의 과거 상환 정보로 다음 달 미상환 확률을 추정하고, 평가한 모델을 동일한 API로 실행하는 학습 프로젝트.**

이 버전은 공개 데이터로 모델링·평가·배포의 기준을 만드는 첫 실험이다.
UCI의 대만 신용카드 고객 30,000건을 사용하며, 소액금융기관에 검증된 신용평가 모델은 아니다.
기후 정보의 유용성은 위치·시점·상환 결과가 연결되는 데이터로 다음 단계에서 검증한다.

## 먼저 볼 결과

- [실제 성능과 한계](MODEL_CARD.md)
- [학습 전에 정한 평가 기준](EXPERIMENT_PLAN.md)
- [전체 지표와 확률 보정 결과](artifacts/metrics.json)
- [배포 모델 버전·데이터·코드 기록](artifacts/manifest.json)
- [분할별 원본 행 ID](artifacts/splits.csv)
- [테스트와 로컬 배포 확인](VERIFICATION.md)

## 우리가 정한 기준

1. 정답의 사건과 기간을 먼저 정의한다. 이번 정답은 UCI의 `default payment next month` 그대로다.
2. 점수는 `100 × 사건 발생 확률`로 표시한다. 의미 없는 가중 합계를 부도 확률로 부르지 않는다.
3. 예측 당시 알 수 있는 입력만 사용한다. 전처리·보정·선택·최종 평가는 구간을 분리한다.
4. 학습 구간 평균 발생률을 내는 단순 기준 모델을 반드시 비교 대상으로 둔다.
5. 모델 선택은 validation log loss로 고정하고, 최종 test 결과는 선택을 바꾸는 데 사용하지 않는다.
6. 데이터 출처, 모델 버전, 평가 결과를 저장한다. API는 그 평가에 사용한 모델을 그대로 로드한다.
7. 데이터가 부족하거나 모델을 로드하지 못하면 임의의 점수를 만들어내지 않는다.

## 설치

Python 3.12를 사용한다. 프로젝트 폴더에서:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

`requirements.lock.txt`에는 이번 실험에서 실제 사용한 버전이 고정되어 있다.
`.joblib`은 신뢰하는 이 프로젝트의 산출물만 로드해야 한다. 해시 검증은 파일 변형을 감지하며 출처 인증을 대신하지 않는다.

## API 실행

저장된 모델이 포함되어 있으므로 재학습 없이 실행할 수 있다.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uvicorn cedar.api:app --host 127.0.0.1 --port 8010
```

API 설명: http://127.0.0.1:8010/docs

```bash
curl http://127.0.0.1:8010/health
curl -X POST http://127.0.0.1:8010/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

예시 요청은 직접 만든 가상의 금융 이력이며 실제 사람의 정보가 아니다.
배열은 **최근 월부터 6개월** 순서, 금액 단위는 **대만 달러(NTD)**다.
대출금액·월소득을 이 모델의 신용한도·청구액으로 임의 대체하면 안 된다.

반환값: 사건 확률, 확률×100 점수, 모델 버전, 예측 목표, 적용 범위, 학습 범위를 벗어난 입력 알림.
`/model`에는 모델 정보가, `/metrics`에는 요청 수·오류 수·평균 응답시간·평균 출력 확률이 있다.
개별 입력은 저장하지 않는다. 지표는 프로세스 메모리에 있으며 재시작하면 초기화된다.
배포 후 실제 성능 저하는 상환 정답이 도착한 뒤 별도로 측정해야 한다.

## 재학습과 평가

```bash
python -m cedar.data --output data/uci350.csv
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m cedar.train \
  --data data/uci350.csv --output runs/reproduction
```

원본 CSV의 SHA-256을 확인하므로 데이터가 조용히 변경되어도 다른 실험으로 섞이지 않는다.
기존 평가를 덮어쓰지 않도록 새 출력 폴더를 지정한다. 테스트를 먼저 살펴본 뒤 모델을 조정하면
그 테스트는 더 이상 최종 평가용으로 사용하지 않아야 한다.

새 모델 실행:

```bash
CEDAR_ARTIFACT_DIR=runs/reproduction/artifacts uvicorn cedar.api:app --host 127.0.0.1 --port 8011
```

## 테스트

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

저장 모델과 API 예측 일치, 잘못된 입력, 누락 필드, 손상된 모델, 그룹 분리 및 모델 선택 규칙을 검증한다.

## 컨테이너로 실행

```bash
docker build -t cedar-baseline:local .
docker run --rm --name cedar-baseline -p 127.0.0.1:8010:8000 cedar-baseline:local
```

컨테이너는 일반 사용자 권한으로 실행되며, 미리 학습·평가된 모델을 로드한다.
모델 교체는 새 모델과 manifest를 함께 포함한 새 이미지로 수행한다. 이전 이미지를 유지하면 롤백할 수 있다.
현재 범위는 로컬 배포까지다. 외부 서비스에 연결하기 전에는 인증, 저장 정책, 관측 체계와 대상 고객 검증이 필요하다.

## 읽으면서 공부할 순서

1. `EXPERIMENT_PLAN.md`: 무엇을 예측하고 어떻게 검증하는지.
2. `cedar/data.py`: 데이터 계약과 중복 프로필을 고려한 분할.
3. `cedar/model.py`: 전처리와 모델을 한 파이프라인으로 묶는 방법.
4. `cedar/train.py`: 별도 확률 보정, 후보 선택, 최종 평가와 모델 저장.
5. `cedar/api.py`: 입력 계약, 모델 로딩, 버전 반환, 최소 운영 지표.
6. `Dockerfile`: 같은 환경과 모델을 재사용하는 배포 단위.

## 데이터와 참고 자료

- Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- 원본 데이터: https://archive.ics.uci.edu/static/public/350/data.csv
- 데이터 라이선스: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 데이터 열 이름을 읽기 쉽게 매핑하고 금융 입력 19개만 선택했다.
- [scikit-learn 교차 검증](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn 확률 보정](https://scikit-learn.org/stable/modules/calibration.html)
- [FastAPI 컨테이너 배포](https://fastapi.tiangolo.com/deployment/docker/)
