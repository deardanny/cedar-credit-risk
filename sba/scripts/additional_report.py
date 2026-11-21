"""Readable follow-up report and locally generated Korean PDF."""
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from make_pdf import p,table,footer
from reportlab.platypus import SimpleDocTemplate,PageBreak,Spacer

def main():
 a=json.loads((ROOT/'web/dist/analysis.json').read_text());h=a['additional_holdout'];m=h['metrics'];r=a['capacity'][9];old=a['original_test'];selection=json.loads((ROOT/'reports/additional/selection_before_holdout.json').read_text());ci=h['confidence_intervals']
 def percent(v):return f'{v*100:.2f}%'
 text=f'''# 로컬 전체 실행 및 추가 검증 결과

작성일: 2026-11-19. 기준 자료: 2026-06-30. 외부 사이트에 게시하지 않고 로컬에서 실행했다.

## 결론
위험 순위를 이용한 과거 자료 검토에는 정보가 있지만, 확률을 실제 심사에 쓰기에는 보정이 불안정하다. 최초 시험의 과소예측을 바로잡았다고 주장하지 않는다. 추가 시험에서는 반대로 과대예측했다. 모델 선택·보정에 시험 정답을 다시 사용하지 않았다.

## 10%만 골라보면
최초 시험 전체 {old['n']:,}건 중 점수가 높은 {r['review_n']:,}건을 선택하면, 36개월 내 상각 {old['events']:,}건 중 {r['events_found']:,}건({r['recall']*100:.1f}%)을 찾았다. 같은 수를 무작위로 고르면 기대 포착 건수는 약 {r['random_expected_events']:.0f}건이다. 약 {r['lift']:.2f}배의 차이다.

선택 집단의 상각 비율은 {percent(r['precision'])}다. 선택한 {r['review_n']:,}건 중 {r['non_events_reviewed']:,}건은 상각이 기록되지 않았으며, 전체 상각 중 {r['events_missed']:,}건은 이 10% 밖에 남아 있다. 실제 검토 비용, 승인 기준 또는 손실액 절감 효과를 계산한 것은 아니다. 1~100% 전체 결과는 additional/review_capacity.csv와 로컬 화면에서 확인한다.

## 다른 시기에 추가 시험
2017년 평가로 고른 {h['selected_model']} / {h['selected_adjustment']} 후보를 최종 적합한 후 2021년 대출 {m['n']:,}건으로 평가했다. 이 집단의 상각은 {m['events']:,}건이다.

| 지표 | 추가 시험 결과 |
|---|---|
| 평균 예상 상각 비율 | {percent(m['predicted_rate'])} |
| 실제 상각 비율 | {percent(m['observed_rate'])} |
| 예상 / 실제 비율 | {m['predicted_rate']/m['observed_rate']:.2f}배 |
| ROC-AUC | {m['roc_auc']:.4f} |
| AUC 차주 부트스트랩 95% 구간 | {ci['roc_auc'][0]:.4f} ~ {ci['roc_auc'][1]:.4f} |
| Average precision | {m['pr_auc']:.4f} |
| 상위 10% lift | {m['top_decile_lift']:.2f}배 |
| Brier | {m['brier']:.6f} |
| 연구 통과 기준 | {sum(h['gates'].values())} / {len(h['gates'])} |

평균 예측은 실제보다 {(m['predicted_rate']/m['observed_rate']-1)*100:.1f}% 높았다. 최초 시험에서는 낮게, 이번 추가 시험에서는 높게 예측했다. 학습 시기와 모델이 다르므로 순수한 모델 개선 비교는 아니다. 원인이 경기, 정책, 표본 선택 중 무엇인지 이 비교만으로 확정할 수 없다. 원자료 없이 단순히 확률을 반으로 조정하는 것은 새 시험을 통과한 개선이 아니다.

## 미리 고정한 순서
설정 비교는 2010~2012 학습, 2013 보정, 2017 평가다. 로지스틱과 부스팅 각각에 무보정, 기울기·절편 보정, 절편만 보정을 적용해 여섯 후보를 비교했다. 연구 기준을 통과한 후보 중 log loss가 가장 낮은 것을 골랐다.

최종 적합은 2010~2015 학습, 2016~2017 보정, 2021 시험이다. 모든 학습·보정의 36개월 결과는 2021년 이전에 완료된다. 차주가 각 분할에 겹치지 않게 했다. 선택 파일을 먼저 저장한 뒤 시험을 계산했고 해시로 변경 여부를 확인했다.

2021년은 이전 모델의 학습·보정·설정 선택·예측 성능평가에는 쓰지 않았지만, 전체 점검에서 연도별 사건 비율은 이미 확인했다. 완전 블라인드 검증이나 외부 검증이 아니다. 기존 2022~2023 결과는 이미 본 시험이며, 후속 설계의 독립 검증으로 재사용하지 않았다.

## 원인 조사와 실행 근거
금액·금리별 확률 오차는 additional/calibration_segments.csv, 후보 비교는 additional/selection_before_holdout.json, 추가 시험은 additional/holdout_2021.json에 있다. 원본 전체 실행 기록은 local_full_run.log다. 추가 검증 코드와 기준은 scripts/investigate.py와 docs/additional_validation_plan.md에 고정했다. 새로운 점수 모델은 models/additional_2021.joblib에 연구용으로 보관하며 기존 웹 계산기를 교체하지 않았다.

## 다음에 필요한 것
현재 의미 있는 성과는 위험 순위와 검토 범위의 관계를 정량화하고, 다른 시기에서 확률 오차가 반복됨을 확인한 것이다. 실제 운영 개선을 확인하려면 새 평가 집단 또는 시점별 내부 자료가 필요하다. 이 시험 결과에 맞춰 반복 조정한 다음 독립 검증이라고 부르지 않는다.
'''
 (ROOT/'reports/추가_검증결과.md').write_text(text)
 rows=[['항목','기존 시험의 상위 10% 결과'],['살펴본 대출',f"{r['review_n']:,} / {old['n']:,}건"],['찾은 상각',f"{r['events_found']:,} / {old['events']:,}건 ({r['recall']*100:.1f}%)"],['무작위 선택 시 기대 상각',f"약 {r['random_expected_events']:.0f}건"],['선택했지만 상각 미기록',f"{r['non_events_reviewed']:,}건"],['선택 집단의 상각 비율',percent(r['precision'])]]
 story=[p('로컬 실행 완료\n추가 검증에서 확인한 것','TitleKR'),p('위험 순서는 도움이 되지만 확률은 아직 불안정합니다','H2KR'),p('원본 정리부터 모델 학습, 검증, 보고서 생성까지 다시 실행했습니다. 로컬 화면에서 예시 대출 계산과 위험 순위별 포착률을 직접 확인할 수 있습니다.'),table(rows,[220,290]),Spacer(1,18),p('상위 10%를 살펴보면 전체 상각의 약 31%를 찾았습니다. 같은 수의 무작위 선택보다 약 3.12배입니다. 하지만 선택한 대출 대부분은 상각되지 않았고, 상각 704건은 이 범위 밖에 있었습니다.'),p('이 수치는 과거 시험의 비교입니다. 실제 승인 기준이나 업무 비용, 손실액 절감 효과를 계산한 것은 아닙니다.'),p('추가 검증의 핵심','H2KR'),p('기존 시험은 평균 위험을 낮게 예측했습니다. 다른 시기의 추가 시험에서는 반대로 높게 예측했습니다. 위험 순위와 확률 숫자의 정확성을 구분해서 봐야 합니다.'),PageBreak(),p('2021년 대출로 추가 시험','TitleKR'),table([['항목','결과'],['추가 시험 대출',f"{m['n']:,}건, 상각 {m['events']:,}건"],['평균 예상 / 실제',f"{percent(m['predicted_rate'])} / {percent(m['observed_rate'])}"],['위험 순위 구분 AUC',f"{m['roc_auc']:.3f}"],['상위 10% lift',f"{m['top_decile_lift']:.2f}배"],['연구 기준',f"{sum(h['gates'].values())} / {len(h['gates'])} 통과. 평균 확률 오차 실패"]],[220,290]),Spacer(1,18),p('어떻게 시험했나요?','H2KR'),p('여섯 후보를 과거 2017년 평가로 비교하고 후보 선택을 저장했습니다. 이후 2010~2015년 학습, 2016~2017년 보정으로 적합한 모델을 2021년 대출에 시험했습니다. 모든 학습·보정 결과는 2021년 이전에 성숙했고 차주가 겹치지 않게 했습니다.'),p('2021년 시험을 본 뒤 후보를 다시 고르거나 확률을 맞추지 않았습니다. 기존 웹 계산기도 교체하지 않았습니다. 학습 시기와 모델이 달라 최초 시험과 직접적인 성능 개선 비교는 아닙니다.'),p('확인한 한계와 다음 단계','H2KR'),p('2021년은 이전 모델 평가에 사용하지 않았지만 자료 점검 중 연도별 사건 비율은 이미 확인했습니다. 완전 블라인드 외부 검증은 아닙니다. 단일 스냅샷의 과거 정보 한계와 승인된 대출만 있다는 한계도 유지됩니다.'),p('현재 자료에서 시험 결과를 좋게 만들기 위해 반복 조정하지 않았습니다. 실제 운영 개선을 검증하려면 새 평가 집단이나 시점별 은행 내부 자료가 필요합니다.'),p('자세한 수치와 실행 근거: 추가_검증결과.md, additional/ 폴더, local_full_run.log. 작성일 2026-11-19.','SmallKR')]
 path=ROOT/'reports/추가_검증결과.pdf';SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=48,title='SBA 로컬 실행 및 추가 검증').build(story,onFirstPage=footer,onLaterPages=footer)
 print(path)
if __name__=='__main__':main()
