"""Write a concise collection guide and empty lender-intake contracts."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/underwriting_public_v1'
DATA=ROOT/'data/processed/underwriting_public_v1'
SCHEMAS={
 'applications':{
  'grain':'One decision version per application; retain approvals and declines',
  'primary_key':['application_id','decision_version'],
  'fields':['application_id','decision_version','borrower_id','loan_id','lender_id','application_at','decision_at','decision','decision_reason_codes','country','currency','product','requested_amount','approved_amount','policy_version','available_at'],
 },
 'borrower_financials':{
  'grain':'One borrower statement period and received version; values in currency units, not thousands',
  'primary_key':['borrower_id','period_end','version'],
  'fields':['borrower_id','period_start','period_end','version','received_at','available_at','source_document_id','currency','revenue','operating_cash_flow','ebitda','cash','current_assets','current_liabilities','total_debt','scheduled_principal','scheduled_interest','debt_service_period_start','debt_service_period_end'],
 },
 'loan_performance':{
  'grain':'One loan per month-end; monthly cash flows versus end-of-month stocks explicitly distinguished',
  'primary_key':['loan_id','observation_date','version'],
  'fields':['loan_id','observation_date','version','available_at','currency','principal_balance','undrawn_commitment','scheduled_principal','scheduled_interest','principal_paid','interest_paid','days_past_due','first_default_date','default_definition_version','restructure_date','paid_in_full_date','loan_sale_date','chargeoff_date','chargeoff_amount'],
 },
 'recoveries':{
  'grain':'One dated cash-flow event; positive amounts with flow_type determining inflow/cost',
  'primary_key':['recovery_event_id','version'],
  'fields':['recovery_event_id','version','loan_id','event_date','available_at','currency','flow_type','amount','source_ledger_id','is_reversal','reverses_event_id'],
  'flow_type_values':['borrower_repayment','collateral_proceeds','guarantee_payment','legal_cost','collection_cost'],
 },
 'collateral_guarantees':{
  'grain':'One collateral/guarantee asset per loan and dated valuation version',
  'primary_key':['loan_id','asset_id','valuation_date','version'],
  'fields':['loan_id','asset_id','valuation_date','version','available_at','currency','collateral_type','appraised_value','prior_lien_balance','guarantee_type','guarantee_share','guarantee_conditions_document_id'],
 },
}

def main():
 c=json.loads((OUT/'coverage.json').read_text())
 m=json.loads((OUT/'manifest.json').read_text())
 d=ROOT/'data/private_intake_templates';d.mkdir(exist_ok=True)
 contract={'status':'empty_templates_no_private_data_obtained','missing_values':'blank/null; never treat missing as zero','date_format':'ISO 8601; available_at is the first usable timestamp, not financial period end','currency':'ISO currency code; do not mix currencies','identity':'Use verified lender borrower_id and loan_id; name similarity is insufficient','tables':SCHEMAS,
 'validation_rules':['Required primary keys unique at stated grain','Loan/borrower foreign keys reconcile across tables','Statement period end <= available_at; prediction uses available_at <= decision_at','Debt-service numerator and denominator cover the same period and currency','Ledger reversal and guarantee/recovery events must not be double counted','Do not infer decline outcomes or net loss from a missing record','Define first default, exposure at default, recoveries, costs and discounting jointly with the lender before creating labels']}
 (d/'data_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
 catalog={
  'loan_fields':{'entity':'applicant loan, public snapshot','fields':['term_months','rate_type','collateral_reported','business_age_detail','naics6'],'join':['source_file','source_row','verified borrower hash','approval date','approved amount'],'warning':'Reported collateral flag is not collateral valuation or recovery value; approval-time revision history unverified'},
  'lender_context':{'entity':'lending bank, not applicant','fields':{'bank_equity_assets':'ratio; 0.1 means 10%','bank_loans_deposits':'ratio; can exceed 1','bank_roa_percent':'annualized percent; 1 means 1%','bank_log_assets_thousands_usd':'log1p(assets in thousand USD)'},'join':['fdic_cert','latest assumed_available_date <= app'],'assumed_publication_lag_days':90,'maximum_extra_age_days':185},
  'business_dynamics':{'entity':'state and 2-digit sector','prefix':'bds_','join':['state_fips','sector','context_year = approval year - 2'],'rates':'original percentage units, not decimal probabilities','warning':'Establishment exits are not borrower default events'},
  'employment_wages':{'entity':'county and 2-digit sector, private ownership','prefix':'qcew_','join':['county_fips','sector','context_year = approval year - 2'],'level_units':'annual average establishment/employment counts and USD wages','growth_units':'percentage change','suppression':'N disclosure flags produce null, never a zero'},
  'borrower_financials_and_net_loss':{'entity':'applicant/loan','status':'not obtained','columns':['borrower_revenue','borrower_operating_cash_flow','borrower_existing_debt','borrower_monthly_debt_service','loan_balance_at_default','net_recovery_cash_flow'],'substitution_allowed':False},
  'historical_vintage_verified':False,'approved_for_training':False,'approved_for_lending':False,
 }
 (OUT/'feature_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
 for name,spec in SCHEMAS.items():
  p=d/(name+'.csv')
  # Preserve any real data the user may already have put into a template.
  if not p.exists():
   with p.open('w',newline='') as f:csv.writer(f).writerow(spec['fields'])
 groups=c['by_cohort'];table=[]
 for g in groups:
  n=g['loans'];pct=lambda k:f"{g[k]:,}건 ({g[k]/n:.1%})"
  table.append(f"| {g['cohort']} | {n:,} | {pct('county_exact_mapping')} | {pct('bank_context_matched')} | {pct('bds_sector_context_matched')} | {pct('qcew_sector_context_matched')} |")
 size=sum(s.get('bytes',0) for s in m['sources'])/1024**2
 text=f'''# 추가 데이터 수집 결과

공식 공개 원본 {len(m['sources']):,}개, 총 {size:,.1f} MiB를 보존하고 기존 대출 {c['loans']:,}건에 연결한 후보 데이터셋을 만들었다. 이 작업은 수집·정제·연결률 검증이며 모델 재학습이나 정확도 개선 검증은 아니다. 현재 Cedar 모델과 기존 평가 결과는 변경하지 않았다.

## 확보한 자료

| 자료 | 범위 | 정제된 행 수 | 무엇을 알려주는가 |
|---|---|---:|---|
| Census BDS | 1978–2023, 주 × 업종 | {c['source_table_rows']['bds_state_sector']:,} | 사업체 진입·폐업, 일자리 생성·소멸 |
| FDIC | {c['source_years']['fdic'][0]}–{c['source_years']['fdic'][1]}, 은행 × 분기 | {c['source_table_rows']['fdic_quarterly']:,} | 대출한 은행의 자산·예금·대출·자본·수익률 |
| BLS QCEW | {min(c['source_years']['qcew'])}–{max(c['source_years']['qcew'])}, 카운티/주 × 업종, 민간부문 | {c['source_table_rows']['qcew_county_state_sector']:,} | 지역 고용·사업체 수·임금 및 증감 |
| 기존 SBA 원본에서 복원 | 기존 개발·2024 평가 대출 | {c['loans']:,} | 만기, 담보 보고 여부, 고정/변동금리, 상세 업종·지역·은행 코드 |

은행 재무자료는 차주의 재무제표가 아니다. 업종 폐업률은 대출 부도 확률이 아니며, 지역 임금은 차주 매출이 아니다. 세 자료를 서로 대신 채우지 않았다.

## 실제 연결률

| 집단 | 대출 수 | 지역 코드 | 은행 과거 분기 | 주·업종 폐업률 | 카운티·업종 고용 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

은행은 FDIC 인증번호, 지역은 주와 정규화한 카운티명, 업종은 공식 산업 코드를 사용했다. 모호한 지역이나 비은행·신용조합의 누락 식별자를 임의로 연결하지 않았다. 카운티/시 구분도 유지했다. 2024 평가 원본에는 최종 모델 평가에서 제외되는 차주도 있으므로 이 연결률의 분모를 이전의 8,632건 정확도 평가와 혼동하지 않는다.

고용 자료는 과거 학습 대출에 연결할 수 있도록 2007년까지 확장했다. 비공개 셀, 역사적 지역코드·산업분류의 차이와 미연결 사례는 결측으로 남겼다. 모든 행이 연결되는 데이터라고 주장하지 않는다.

## 사용 시점과 데이터 누출

- BDS·QCEW는 승인 연도보다 2년 전 관측값, 은행은 보고일에 90일을 더한 날이 승인일 이전인 가장 최근 분기를 붙였다. 은행의 그 이후 추가 경과일은 최대 185일이다.
- 이 시차는 탐색용 가정이다. 과거 자료의 사후 정정 이력, 실제 발표일·입수일을 완전히 확인한 것은 아니다. 현재 다운로드한 개정 통계를 과거의 완전한 정보 시점 자료로 표현하지 않는다.
- 모든 행의 `historical_vintage_verified`와 `approved_for_training`은 false다. 현재 모델에는 자동 투입하지 않았다. 엄격한 과거 성능 비교에는 당시 공개 버전/입수 시점 검증이 추가로 필요하다.
- 2023–2025 자료는 현재 분석에 보존하지만 2024년 승인 대출에 미래 값으로 붙이지 않는다.
- SBA 단일 스냅샷의 만기·담보·금리 유형도 최초 승인 당시 버전인지 추가 확인해야 한다. 결과 관련 상태/상각액은 이 후보 변수 테이블로 가져오지 않았다.

## 아직 확보하지 못한 핵심 자료

차주 매출·영업현금흐름·기존 부채·원리금 부담, 월별 대출 잔액과 연체, 회수금·보증 지급·회수 비용, 거절 신청 이력은 이 수집에서 확보하지 못했다. 대응 열은 전부 결측으로 남겼다. SEC 공식 회사 목록 요청은 HTTP 403으로 실패했으며 회사별 재무제표를 연결했다고 주장하지 않는다. 이름이 비슷한 다른 회사의 재무자료를 붙이지 않았다.

금융기관/차주에게 적법하게 자료를 받으면 바로 적재할 수 있도록 `data/private_intake_templates/`에 다섯 개 빈 CSV와 `data_contract.json`을 만들었다. 헤더만 있는 양식이며 실데이터가 아니다. 외부 기관에 연락하거나 자료를 요청하지는 않았다.

## 파일

- 원본 및 개별 출처: `data/external/underwriting_public_v1/`
- 대출별 후보 연결표: `data/processed/underwriting_public_v1/loan_context_candidates.csv.gz`
- 정제 원천 테이블: 같은 폴더의 `bds_state_sector.csv.gz`, `fdic_quarterly.csv.gz`, `qcew_county_state_sector.csv.gz`
- 출처·크기·다운로드 일시·SHA-256: `reports/underwriting_public_v1/manifest.json`
- 연결률 및 한계: `coverage.json`; 미연결 식별자: `unmatched_counties.csv`, `unmatched_lender_ids.csv`
- 기관 전달용 빈 양식: `data/private_intake_templates/`

## 단위

- FDIC의 ASSET·DEP·LNLSNET·EQ·NETINC는 천 달러 단위 원본이다. NETINC는 연초부터 누적된 이익으로 별도 분기 이익처럼 합산하지 않는다. ROA는 연율화된 백분율이다. 후보 연결표의 자본/자산, 대출/예금은 백분율로 바꾸기 전 비율(0.1 = 10%)이며 대출/예금은 1을 넘을 수 있다.
- QCEW 임금은 달러, 고용은 연평균 고용, 증감률은 백분율 단위다. 집계된 사업체 수는 차주의 매출이나 재무로 취급하지 않는다.
- BDS의 진입·퇴출·일자리 증감률은 원본 백분율이다. 폐업 사건이 대출 상각 사건이라는 뜻은 아니다.
- 내부 자료 양식의 금액은 `currency`에 명시한 통화의 기본 단위로 받는다. 천 단위 원천과 섞지 않는다.

재현: `.venv/bin/python scripts/collect_underwriting_data.py fetch` 다음 `prepare`; 설명 재생성: `.venv/bin/python scripts/write_collection_catalog.py`. 다운로드 캐시를 체크섬으로 검증하며 재사용한다. 실행 시점의 최신 API를 새로 받으면 원천 값이 바뀔 수 있으므로 기존 파일·출처 기록을 보존한다.

검증: `.venv/bin/python -m unittest discover -s tests -p 'test_underwriting_collection.py' -v`.

## 공식 출처

- Census BDS: https://www.census.gov/data/datasets/time-series/econ/bds/bds-datasets.html
- BLS QCEW: https://www.bls.gov/cew/downloadable-data-files.htm
- QCEW 개정 정책: https://www.bls.gov/cew/revisions/
- FDIC API: https://api.fdic.gov/banks/docs
- FDIC 필드 사전: https://api.fdic.gov/banks/docs/risview_properties.yaml
- 은행 공시 단위·기간: https://cdr.ffiec.gov/public/HelpFiles/FAQ.htm
- SEC 회사 재무 API 범위: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- SBA 필드 정의: 사용자 제공 `7a_504_foia_data_dictionary.xlsx`, `7(a) Data Dictionary` A11:B11, A24:B25, A42:B42.
'''
 (OUT/'README.md').write_text(text)
 print('Saved collection guide and five empty lender intake templates.')

if __name__=='__main__':main()
