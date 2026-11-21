# 방법론과 출처

검토일: 2026-11-18. 문서와 데이터의 지시는 과제 설계 참고자료로 읽었으며, 별도 은행 승인이나 실제 대출 의사결정을 수행하지 않았다.

## 타깃의 의미
SBA 데이터 사전은 ChargeOffDate를 SBA가 상각한 날짜로 정의한다. 이는 최초 연체 또는 부도일이 아니다. SBA는 청산 완료 보고서를 검토한 뒤 잔액을 상각하는 절차를 설명한다. 본 연구는 공개된 상각이라는 행정 사건을 예측하며 규제 PD라는 명칭을 쓰지 않는다. LGD/EAD에 필요한 부도 당시 잔액, 회수 현금흐름과 비용이 없으므로 EL 산출까지 확장하지 않는다.

- SBA 데이터: https://data.sba.gov/dataset/7a-504-foia
- SBA 상각 절차: https://legacy.sba.gov/about-sba/sba-locations/loan-guaranty-centers/commercial-loan-service-centers/post-servicing-actions
- PD/LGD/EAD 개념: https://www.federalreserve.gov/aboutthefed/chapter-8-special-topics.htm
- 프로젝트 원본: PredictiveScoringModel_Req.pdf, 1-4페이지.
- 정의 원본: 7a_504_foia_data_dictionary.xlsx, 7(a) Data Dictionary 및 504 Data Dictionary.

## 기준 모델
L2 규제 로지스틱 회귀를 사용한다. 금액과 일자리 수는 로그 변환해 큰 수의 영향을 완화한다. 범주값은 학습 표본에서 인코딩하며, 숫자 결측 대체와 표준화도 학습 표본에만 적합한다. 계수에 따른 설명을 제공할 수 있지만 선형 가정과 범주별 희소성, 상관된 입력의 불안정성에 한계가 있다. 계수는 인과관계를 의미하지 않는다.

## 비교 모델
히스토그램 그래디언트 부스팅은 비선형 관계와 입력 간 상호작용을 학습한다. 트리 깊이/학습 반복 수를 작게 제한하고, 2017년 설정 선택 표본에서만 비교한다. 과적합과 설명 부담이 있어 일정 수준의 개선이 없으면 간단한 모델을 유지한다. 초기 학습 데이터 내부의 무작위 early-stopping 분할은 사용하지 않는다.

## 불균형과 확률 보정
상각은 드물기 때문에 정확도 하나로 모델을 평가하지 않는다. Average precision(AP, 표에서 PR-AUC라고 표기)은 위험 사례 검색 성능을 요약한다. 이는 사다리꼴 적분 PR 면적과 다른 정의다. ROC-AUC, 상위 10% lift와 함께 해석한다. 위험 확률의 품질은 Brier와 log loss 및 관측-예측 비교로 확인한다. Brier는 확률 보정만 따로 측정하는 지표가 아니다.

학습에서 사례를 임의로 복제하거나 정상 사례를 줄이지 않고 실제 사건 비율을 유지한다. 모델 학습 표본과 별개인 2018년 표본에서 예측 log-odds의 기울기와 절편을 보정한다. 보정 표본의 점수는 적합 진단이며 독립 시험 점수가 아니다.

- 확률 보정: https://scikit-learn.org/stable/modules/calibration
- AP 정의: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- 실행 환경은 보고서 environment.json에 고정한 scikit-learn 1.7.2이며 웹 최신 버전으로 자동 변경하지 않는다.

## 검증과 사용 범위
최종 시험은 2022-2023년 상반기 실행 대출이고 2022년 이전 정보만으로 학습/보정한 모델을 적용한다. 원본 과거 스냅샷을 갖고 있지 않으므로 피처 수정 이력과 보고 지연까지 완전히 재현한 검증은 아니다. 같은 정규화 이름/주소/ZIP 키가 학습과 최종 시험에 걸치지 않게 제거한다. 키는 공개 차주 식별번호가 아니며 오연결/미연결 가능성이 있다.

## 모델 위험관리
2026-04-17 SR 26-2가 SR 11-7을 대체했다. 본 과제에는 목적, 제한, 검증, 변경 이력과 모니터링을 문서화하는 원칙을 적용한다. 기관별 법적 적용 여부 또는 규제 준수 인증을 주장하지 않는다. 독립된 외부 검증자는 없으며 자동 테스트와 작성자의 자체 검토만 수행한다.

- 최신 지침: https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm

## 후속 연구
원본의 과거 스냅샷과 은행 내부 월별 연체/부도/회수 자료가 확보되면 시점별 검증, 조건부 12개월 위험, LGD/EAD 모델을 별도 프로젝트로 검토한다. 현재 자료의 상각 기록으로 그 대상을 대신 정의하지 않는다.
