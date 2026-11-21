"""Generate evidence figures, readable Korean findings, and a technical report."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/sba_mpl')
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve,precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'reports'/name).read_text())
def pct(v):return f'{v*100:.2f}%'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'

def main():
    results=read('model_results.json');split=read('split_audit.json');flow=read('exclusion_flow.json');stability=read('stability.json');sensitivity=read('sensitivity.json');coverage=read('demo_coverage.json')
    chosen=results['selected_before_test'];m=results['test'][chosen];scores=pd.read_csv(ROOT/'data/processed/test_predictions.csv');bands=pd.read_csv(ROOT/'reports/risk_bands.csv');segments=pd.read_csv(ROOT/'reports/segments.csv');inventory=read('source_inventory.json')
    figdir=ROOT/'reports/figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.titlepad':14,'figure.facecolor':'white','axes.labelcolor':'#243c48','text.color':'#243c48','savefig.dpi':160})
    colors={'logistic':'#7b8c99','boosting':'#176879'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for kind in colors:
        fpr,tpr,_=roc_curve(scores.y,scores[kind]);axes[0].plot(fpr,tpr,label=f'{kind}: {results["test"][kind]["roc_auc"]:.3f}',color=colors[kind],lw=2)
        precision,recall,_=precision_recall_curve(scores.y,scores[kind]);axes[1].plot(recall,precision,label=f'{kind}: AP {results["test"][kind]["pr_auc"]:.4f}',color=colors[kind],lw=2)
    axes[0].plot([0,1],[0,1],ls='--',color='#b6c0c7');axes[0].set(title='Held-out discrimination',xlabel='False positive rate',ylabel='True positive rate');axes[0].legend()
    axes[1].axhline(scores.y.mean(),ls='--',color='#b8682b',label='Event prevalence');axes[1].set(title='Rare-event retrieval',xlabel='Recall',ylabel='Precision',ylim=(0,.25));axes[1].legend();fig.savefig(figdir/'discrimination.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].plot(bands.predicted,bands.observed,'o-',color='#176879',lw=2);mx=max(bands.observed.max(),bands.predicted.max())*1.15;axes[0].plot([0,mx],[0,mx],'--',color='#a6b2ba');axes[0].set(title='Predictions underestimate events',xlabel='Mean predicted probability',ylabel='Observed event rate',xlim=(0,mx),ylim=(0,mx));axes[0].xaxis.set_major_formatter(PercentFormatter(1));axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].bar(np.arange(len(bands)),bands.observed,color='#176879');axes[1].set(title='Observed risk across reference bands',xlabel='Risk band (2018 calibration cutoffs)',ylabel='Observed event rate');axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_xticks(np.arange(len(bands)),np.arange(1,len(bands)+1));fig.savefig(figdir/'calibration.png');plt.close(fig)
    cohort=pd.read_csv(ROOT/'reports/cohorts.csv');fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.plot(cohort.disb,cohort.event_rate,marker='o',color='#176879');ax.set(title='36-month recorded charge-off by disbursement year',xlabel='Disbursement year (2023 through June only)',ylabel='Observed event rate');ax.yaxis.set_major_formatter(PercentFormatter(1));fig.savefig(figdir/'vintages.png');plt.close(fig)
    summary=f'''# 이번 프로젝트에서 확인한 것

## 무엇을 만들었나
과거 SBA 7(a) 대출을 이용해, 실행 후 3년 안에 SBA가 손실 처리할 위험을 계산하는 연구용 모델과 웹 데모를 만들었다. 실제 대출 승인이나 금리 결정에 쓰는 도구는 아니다.

## 자료를 얼마나 사용했나
원본 6개 CSV의 2,190,504행을 점검했다. 한 종류의 대출로 범위를 좁히고 날짜·상태·중복·관찰기간을 정리한 분석 자료는 {flow[-1]['n']:,}행이다. 최종 시험에는 학습 때 본 차주와 분리된 {m['n']:,}건을 사용했고, 그중 {m['events']:,}건이 3년 안에 상각됐다.

## 무엇이 잘됐나
선택된 모델이 위험하다고 본 상위 10% 집단에서는 전체 평균보다 약 {m['top_decile_lift']:.2f}배 높은 비율로 상각이 발생했다. 위험 수준을 구분하는 데 도움이 된다는 결과다.

## 무엇이 부족했나
모델의 평균 예상치는 {pct(m['predicted_rate'])}, 실제는 {pct(m['observed_rate'])}였다. 평균 위험을 약 {(1-m['predicted_rate']/m['observed_rate'])*100:.1f}% 낮게 예측했다. 미리 정한 연구 기준 5개 중 평균 확률 오차 기준을 통과하지 못했다. 위험 순서를 구분하는 능력과 확률 숫자가 정확한지는 다르다.

시험 시기의 금리 분포가 학습 시기와 크게 달라졌다. 이는 추가 조사할 근거지만 과소예측의 원인이라고 입증한 것은 아니다. 모델을 최종 시험에 맞춰 다시 조정하지 않았고, 더 유리해 보이는 모델로 사후 교체하지 않았다.

## 데모를 어떻게 읽어야 하나
표시되는 퍼센트는 연구용 계산값이다. 실제 부도확률 또는 현재 고객에게 검증된 확률이 아니다. 입력값은 브라우저 안에서 계산하고 외부로 전송하지 않는다. 학습 범위 밖의 값에는 점수를 제공하지 않는다. 최종 시험 전체의 {coverage['coverage']*100:.1f}%가 이 입력 지원 범위에 들어온다.

## 실제 사용을 위해 더 필요한 것
Northwest의 실제 고객 자료, 최초 연체/부도 시점, 과거 시점별 입력값과 회수 자료가 필요하다. 이를 확보한 후 새 평가 자료를 분리하고 확률 보정과 은행 적용성을 다시 검증해야 한다. 현 결과는 연구와 포트폴리오 시연 단계까지 완성한 것이다.
'''
    (ROOT/'reports/쉬운_결과설명.md').write_text(summary)
    rows=[]
    for k,v in results['test'].items():rows.append([k,f"{v['roc_auc']:.4f}" if v['roc_auc'] is not None else '-',f"{v['pr_auc']:.4f}" if v['pr_auc'] is not None else '-',f"{v['brier']:.6f}",pct(v['predicted_rate']),pct(v['observed_rate'])])
    report='# SBA 7(a) 36개월 상각 위험 모델 개발 및 검증\n\n'
    report+='## 결론\n\n선택 모델: '+chosen+'. 연구 기준 전체 통과: '+str(results['research_gate_pass'])+'. 운영 사용 승인: 없음. 구분력은 확인됐으나 평균 확률 과소예측 기준을 실패했다.\n\n'
    report+='## 문제와 타깃\n\n최초 실행 시점의 승인 조건을 이용해 36개월 내 공개된 ChargeOffDate 사건을 예측한다. 대출 수준의 관측 누적 상각 위험이며 최초 부도, 연체, 회계 LGD 또는 EL을 추정하지 않는다. 완제는 실제 상환 경로의 종료로 다루며 가상의 계속 대출 상태를 추정하지 않는다. 원본은 승인된 SBA 대출만 포함하므로 일반 신청자 전체에 대한 대표성이 없다.\n\n'
    report+='## 원본과 재현성\n\n기준일은 2026-06-30이다. 압축파일/데이터 사전/PDF 및 각 CSV의 SHA-256은 source_inventory.json에 저장했다. 원본을 수정하지 않는다. 전체 CSV 행 수는 '+str(sum(i['rows'] for i in inventory['csv_inventory']))+'이다.\n\n'
    report+='## 정리 내역\n\n'+table(['순차 제외 규칙','제외 전','제외 행','제외 후'],[[v['rule'],f"{v['before']:,}",f"{v['removed']:,}",f"{v['after']:,}"] for v in flow if 'removed' in v])+'\n'
    report+='공개 고유 대출번호가 없으므로 전체 값이 같은 행을 동일 공개 기록으로 간주하고 첫 행을 남겼다. 실제 별도 대출의 우연한 일치 가능성이 있으므로 확인된 중복 대출이라고 단정하지 않는다. 정규화 이름/주소/ZIP의 해시는 가명 연결 키이며 익명성을 보장하지 않는다. 원본 개인·차주 이름은 데모나 공개 보고서에 포함하지 않는다.\n\n'
    report+='## 입력 변수\n\n'+table(['변수','변환/주의'],[['GrossApproval','양수 확인 후 자연로그'],['SBAGuaranteedApproval / GrossApproval','0~1 범위 확인'],['InitialInterestRate','승인 당시 총금리. 0/불가능한 값은 결측으로 처리. 금리 결정 전 사용 불가'],['JobsSupported','음수 제외 후 log(1+x). 신청서상 미감사 추정'],['NaicsCode','대분류 통합. 31~33/44~45/48~49를 각각 통합'],['ProjectState','프로젝트 주. 편향/대리변수 가능성 검토 필요'],['BusinessAge','New / Existing / Unknown으로 보수적 통합. 소유권 변경은 Unknown'],['BusinessType','범주 인코딩']])+'\n'
    report+='현재 은행, 이차시장 매각 여부, 대출 상태/사건일/상각액, 완제일은 입력에서 제외했다. 최초값 여부가 확인되지 않은 기간/담보도 제외했다. 결측 대체, 표준화, 범주 인코딩은 각 모델의 학습 데이터에만 적합했다.\n\n'
    report+='## 시간순 실험 설계\n\n'+table(['단계','실행일 범위','대출 수','상각 수','36개월 결과 완료'],[[name,v['start']+' ~ '+v['end'],f"{v['n']:,}",f"{v['events']:,}",v['label_complete_by']] for name,v in split.items() if isinstance(v,dict)])+'\n'
    report+='최종 모델의 가상 동결일은 2022-01-01이다. 모든 학습과 보정 결과가 동결 이전에 성숙한다. 보정 집단은 파라미터 적합용이며 과거 2018년 시점의 독립 배포 시험이 아니다. 설정 선택용 2017년 집단은 초기 학습 이후 충분한 결과 관찰 간격을 두었다. 최종 시험은 학습/보정 차주와 분리한다. 2019~2021년 대출은 이 동결 설계의 최종 학습 또는 시험에 넣지 않았다.\n\n'
    report+='## 모델 비교\n\n'+table(['모델','ROC-AUC','AP','Brier','평균 예상','실제'],rows)+'\n'
    report+='가족별 설정은 2017년 AP로 선택했다. 부스팅은 기준 모델 대비 AP 상대 개선 10% 이상과 AUC 비하락 기준을 충족해 최종 시험 이전에 선택됐다. 최종 시험에서 로지스틱 AP가 더 높아도 선택을 바꾸지 않았다. constant_calibration_rate는 2018년 실제 비율을 고정한 비교값이며 최종 시험의 실제 비율을 미리 쓰지 않는다.\n\n'
    report+='AP는 sklearn average_precision_score이며 사다리꼴 PR 면적과 다르다. ROC-AUC는 순위 구분 지표이고 77% 정확도라는 의미가 아니다. Brier와 log loss는 보정과 구분을 함께 반영하는 적절한 확률 점수다.\n\n'
    report+=table(['연구 통과 기준','결과'],[[k,'통과' if v else '실패'] for k,v in results['research_gates'].items()])+'\n'
    report+='## 불확실성과 세부 결과\n\n'+table(['지표','차주 그룹 부트스트랩 95% 구간'],[[k,f'{v[0]:.5f} ~ {v[1]:.5f}'] for k,v in results['bootstrap_95_ci'].items()])+'\n'
    report+='부트스트랩 200회, 고정 난수. 구간은 시험 표본의 불확실성을 나타내며 미래 시점으로의 일반화를 보장하지 않는다. 산업/지역/연령/금액/연도별 결과는 segments.csv에 있다. 사건 수 30건 미만인 집단은 저표본 표시를 했으며 강한 결론을 내리지 않는다.\n\n'
    yr=segments[segments.dimension=='year'];report+=table(['실행 연도','시험 수','ROC-AUC','예상','실제'],[[r.segment,int(r.n),f'{r.roc_auc:.3f}',pct(r.predicted_rate),pct(r.observed_rate)] for r in yr.itertuples()])+'\n'
    report+='## 입력 변화와 민감도\n\n'+table(['입력','PSI'],[[r['feature'],f"{r['psi']:.3f}"] for r in stability])+'\n'
    report+='PSI는 학습과 시험의 분포 비교 진단이다. 경계는 학습 자료에서 정했고 희소 빈도는 1e-6으로 안정화했다. 금리 분포 변화가 크지만 그것만으로 과소예측의 원인이라고 결론내리지 않는다.\n\n'
    report+=table(['고정 로지스틱 민감도 모델','ROC-AUC','AP','평균 예상'],[[k,f"{v['roc_auc']:.4f}",f"{v['pr_auc']:.4f}",pct(v['predicted_rate'])] for k,v in sensitivity.items()])+'\n'
    report+='이 민감도 결과는 사후 진단이며 챔피언 재선택이나 시험에 맞춘 조정에 사용하지 않았다.\n\n'
    if (ROOT/'reports/global_permutation.csv').exists():
        imp=pd.read_csv(ROOT/'reports/global_permutation.csv')
        report+='## 모델이 의존하는 정보\n\n'+table(['입력','섞었을 때 AUC 감소','AP 감소'],[[v.feature,f'{v.auc_drop_mean:.4f}',f'{v.ap_drop_mean:.4f}'] for v in imp.itertuples()])+'\n'
        report+='최종 시험의 입력 한 종류를 3회 무작위로 섞은 사후 설명 진단이다. 모델 선택이나 재학습에는 사용하지 않았다. 감소가 클수록 해당 시험에서 그 정보에 더 의존했다는 뜻이며 인과 효과가 아니다. 상관된 입력과 비현실적 조합 때문에 중요도가 왜곡될 수 있다. 섞기 표준편차는 신뢰구간이 아니다. baseline_coefficients.csv에는 기준 모델의 인코딩별 계수를 저장했다. 수치 계수는 로그 변환 및 표준화 이후 단위이며 범주 전체를 포함한 정규화 회귀이므로 원자료 단위의 효과로 읽으면 안 된다.\n\n'
    report+='## 데모와 입력 지원 범위\n\n선택된 모델과 확률 보정기를 JavaScript로 내보내 브라우저 안에서 계산한다. 트리 노드 직렬화는 scikit-learn 1.7.2에 고정하며 Python과 JavaScript의 예측 일치 테스트로 검증한다. 설명은 입력 하나만 학습 대표값으로 대체했을 때의 확률 차이이며 SHAP 또는 인과 효과가 아니다. 조합이 실제 존재하는 차주를 의미하지 않는다.\n\n'
    report+=f"데모 지원 범위: 시험 {coverage['supported_n']:,}행 / 전체 {m['n']:,}행 ({coverage['coverage']*100:.2f}%). 누락/학습범위 밖 값은 점수 미제공이다. 본문 성능은 전체 시험 집단이며 지원 범위 부분집단 성능은 demo_coverage.json에 별도로 저장했다.\n\n"
    report+='## 제한과 남은 의사결정\n\n원본 과거 스냅샷 부재로 피처의 과거 수정과 보고 지연을 완전히 검증할 수 없다. 현재값으로 기록된 비회전형 여부도 최초 시점과 동일하다는 가정이다. 사업연령 기록 체계, NAICS 버전, 경제 상황 변화가 남아 있다. protected attribute가 없어 공정성 검증을 완료했다고 주장하지 않는다. 외부 독립 검증은 없고 자동 테스트와 작성자 자체 검토만 수행했다. Northwest 모집단과 실제 부도/연체 자료, 내부 승인 체계가 확인돼야 운영 적합성을 다시 평가할 수 있다.\n\n'
    report+='## 추가 문서\n\n- ../docs/project_charter.md: 사전 정의와 연구 통과 기준\n- ../docs/research_summary.md: 방법론과 공식 출처\n- ../docs/implementation_monitoring.md: 적용·예외·모니터링 개념\n- verification.json 및 browser_parity.json: 자동 검증 근거\n- figures/: 판별·확률·연도별 그래프\n'
    (ROOT/'reports/technical_report.md').write_text(report)
    web={'chosen':chosen,'metrics':m,'gates':results['research_gates'],'scope':'SBA 7(a) term loans, 36-month recorded charge-off','asof':'2026-06-30','coverage':coverage}
    (ROOT/'web/dist/results.json').write_text(json.dumps(web,indent=2))
    print('Reports and figures generated',flush=True)

if __name__=='__main__':main()
