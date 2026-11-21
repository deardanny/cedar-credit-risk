"""Korean paper-application findings, generated from verified aggregate results."""
import json,sys,html
from pathlib import Path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
ROOT=Path(__file__).resolve().parents[1]
R=json.loads((ROOT/'reports/papers/evaluated_results.json').read_text())
pdfmetrics.registerFont(TTFont('AppleGothic','/System/Library/Fonts/Supplemental/AppleGothic.ttf'))
INK=colors.HexColor('#183f3a');TEAL=colors.HexColor('#087e73');LIGHT=colors.HexColor('#e9f0ea')
styles={k:ParagraphStyle(k,fontName='AppleGothic',fontSize=size,leading=leading,textColor=INK,spaceAfter=space,wordWrap='CJK') for k,size,leading,space in [('title',26,35,18),('h',16,23,12),('p',10,17,10),('small',8,13,8),('cell',8.5,13,0)]}
items=[]
def p(text,style='p'):items.append(Paragraph(text,styles[style]))
def table(rows,widths):
    t=Table([[Paragraph(html.escape(str(c)),styles['cell']) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#d3dfd4')),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9),('VALIGN',(0,0),(-1,-1),'TOP')]))
    items.extend([t,Spacer(1,14)])
def signed(x):return f'{x:+.1%}'
def million(x):return f'${x/1e6:.2f}M'
def footer(c,doc):
    c.setStrokeColor(TEAL);c.line(44,801,551,801);c.setFont('AppleGothic',8);c.setFillColor(INK);c.drawString(44,814,'SBA LOSS LAB | PAPER METHODS | 2026-11-20');c.drawString(44,29,'공개 자료의 과거 재분석 · 총상각액 예측 · 실제 심사 사용 미승인');c.drawRightString(551,29,str(doc.page))

p('논문 기법을 적용한<br/>대출 손실 모델 실험','title')
p('실제로 학습하고 비교했지만, 아직 실무 심사에 사용할 정확성은 확보하지 못했다.','h')
p('7가지 방식을 4개 시점에서 비교했다. 2018·2019·2020년 대출은 6년, 2024년 대출은 2년 동안 관찰했다. 현재 데이터에 기록된 총상각액을 예측하며, 회수액·보증금·회수 비용을 반영한 은행 순손실을 예측하는 모델은 아니다.')
rows=[['대출 집단 / 기간','건수 / 상각','실제액','기존 오차','경기 변화','경기 + 보정']]
for c in R['cohorts']:
    vals=[c['variants'][v][-1] for v in ['base','dynamic_macro','dynamic_macro_iso']]
    rows.append([f"{c['year']} Q3 / {c['months']//12}년",f"{c['n']:,} / {c['events']}",million(vals[0]['amount']['actual_total']),*[signed(v['amount']['predicted_actual_ratio']-1) for v in vals]])
table(rows,[101,78,76,74,82,96])
p('오차는 (예측액 ÷ 실제액 - 1). +는 과대예측, -는 과소예측. 각 행은 같은 대출 집단을 비교한다. 2024년은 기간과 차주 제외 범위가 달라 이전 연구의 2년 시험과 직접 비교하지 않는다.','small')
p('어디까지 좋아졌나?','h')
s=R['summary']['dynamic_macro'];u=s['pooled_uncertainty']
p(f"경기 변화 모델은 2020년 총액 오차를 +12.1%에서 +3.6%로 줄였다. 세 6년 집단의 평균 절대 총액 오차는 기존 {R['summary']['base']['mean_absolute_relative_error']:.1%}에서 {s['mean_absolute_relative_error']:.1%}로 감소했다. 하지만 2019년에는 +59.2% 과대예측이 남았다.")
p(f"세 집단을 합친 개별 금액 오차(Tweedie deviance)는 {u['deviance_improvement']:.2%} 감소했다. 차주 단위 재표집 95% 구간은 {u['deviance_improvement_95_ci'][0]:.2%}~{u['deviance_improvement_95_ci'][1]:.2%}였다. 시간 변화에 대한 안정성이나 새 대출에 대한 정확성을 보장하는 구간은 아니다.")
p('최근 집단에서는 실패','h')
p('2024년의 경기 변화 + 단조 보정 모델은 실제 $4.01M에 대해 $12.18M을 예측했다(3.03배). 실제 상각 비율 0.52%를 1.16%로 높게 예측했고, 사건당 금액도 과대평가했다. 사전에 정한 연구 통과 기준을 모두 만족한 새 모델은 없었다.')
items.append(PageBreak())
p('어떤 논문 기법을 어떻게 적용했나?','title')
table([
['근거','이 프로젝트의 구현','원 논문과 다른 점'],
['Peng & Lessmann (2026)','24개 대출 경과 분기 지표 + 다항 로지스틱 위험 모델. 별도 차주의 최근 12개월 완결 구간으로 isotonic 보정.','월별 실제 잔액과 예정 상환액의 차이를 이용한 행동 지표는 자료가 없어 미구현. 전체 LMISO 재현이 아님.'],
['Bellotti & Crook (2009)','각 위험 구간 시작 전의 ALFRED 발표 버전으로 11개 경기 관련 입력을 갱신. 승인 당시 고정 방식과 비교.','Cox 대신 기존 트리 모델에 적용. 미래 실제 경기를 넣지 않고 유지 또는 과거 5년 평균으로 이동한다고 가정.'],
['Saavedra 외 (2024)','기존의 상각·상환·유지 경쟁 사건 구조를 유지. 누적 상각 확률과 기간별 금액을 계산.','CWGB 또는 Fine-Gray 알고리즘은 구현하지 않음. 기록된 상각과 모든 상환 완료를 사용하므로 부도·조기상환과 다름.']
],[105,201,201])
p('미리 고정한 7가지 방식의 결과','h')
labels={'base':'기존 방식','age_logit_joint':'기간별 로지스틱','age_logit_iso':'기간별 로지스틱 + 단조 보정','static_macro':'승인 당시 경기 고정','dynamic_macro':'경기 변화 / 미래 유지','dynamic_macro_iso':'경기 변화 + 단조 보정','dynamic_macro_revert':'경기 변화 / 과거 평균 이동'}
rows=[['방식','2018 오차','2019 오차','2020 오차','2024 오차']]
for v,label in labels.items():rows.append([label,*[signed(c['variants'][v][-1]['amount']['predicted_actual_ratio']-1) for c in R['cohorts']]])
table(rows,[187,80,80,80,80])
p('로지스틱은 모든 사건 구간과 무사건 구간의 15% 표본을 사용하고, 표본추출 확률의 역수로 가중했다. 트리 모델은 모든 학습 구간을 사용했다. 각 시점의 금액 모형(Gamma) 사양은 동일하게 유지하고 보정 자료에서 총액 계수만 다시 추정했다. 모델을 저장·해시 기록한 뒤 비교 집단을 점수화했다.','small')
p('단조 보정의 세 상태 확률','h')
p('먼저 상각 확률을 보정하고, 상각이 아닌 경우의 조건부 상환 확률을 별도로 보정했다. 두 값을 결합하여 상각·상환·유지 확률이 항상 100%가 되도록 했다. 분기별 남은 확률에 상각 확률과 조건부 금액을 곱해 더하면 누적 예상 상각액이 된다.')
items.append(PageBreak())
p('검증, 한계, 재실행','title')
p('검증 완료','h')
parity=json.loads((ROOT/'reports/papers/browser_parity.json').read_text())
p(f"19개 단위 검사(새 검사 9개 + 기존 경계 검사 10개)를 통과했다. 미래 경기 값을 크게 바꾸거나 삭제한 전체 예측 {R['validation']['forecast_invariance_cases']:,}개 시점 비교에서 차이는 0이었다. 합성 입력 48개 × 4개 방식 × 6개 연도의 브라우저·Python 계산 {parity['variant_horizon_cases']:,}개를 대조했고 최대 상대 차이는 {parity['max_relative_error']:.2e}였다.")
p('학습·보정 차주 중복 0, 학습 기준일 이전 완결 구간만 사용, 사건 이후 위험 집합 제외, 확률 합 1, 누적 금액 비감소를 확인했다. 이전에 고정한 파일 18개를 보존했고 기존 6년 예측을 정확히 재현했다.')
p('숫자를 해석할 때 남는 한계','h')
p('1. 새 독립 시험이 아닌 이미 본 과거 집단의 재분석이다. 모델 설정을 이번 학습 전에 고정했어도 이전에 본 결과를 모르게 만들 수는 없다.<br/>2. 500회 차주 단위 재표집 구간은 고정한 모델·선택한 시점에 조건부이다. 모델 재학습, 새 경기 충격, 다중 비교를 모두 반영한 신뢰 보장은 아니다.<br/>3. SBA 최신 스냅샷에서 과거 자료를 복원했으므로 사후 수정된 입력과 상각 금액의 원래 발표 시점을 완전히 재현할 수 없다. 경기 통계만 발표 당시 버전을 사용했다.<br/>4. 승인된 SBA 7(a) 비회전형 대출만 포함한다. 거절 신청자나 다른 상품으로의 적용은 검증하지 않았다.<br/>5. 상각은 최초 부도와 다르다. 월별 잔액·회수 이력이 없으므로 순 LGD, 실제 EAD, 만기 전체 ECL을 학습했다고 볼 수 없다.')
p('재실행 파일','h')
p('계획: docs/paper_plan.md<br/>학습: scripts/paper_methods.py<br/>통계 평가·누출 검사: scripts/paper_evaluate.py<br/>브라우저 내보내기: scripts/paper_export.py<br/>모델: models/paper_2018.joblib, paper_2019.joblib, paper_2020.joblib, paper_2024.joblib<br/>집계 결과: reports/papers/evaluated_results.json<br/>로컬 화면: http://127.0.0.1:8813/paper.html','small')
p('논문 및 데이터 출처','h')
for title,url in [
('Saavedra 외 (2024), Probability of default for lifetime credit loss for IFRS 9 using machine learning competing risks survival analysis models','https://doi.org/10.1016/j.eswa.2024.123607'),
('Bellotti & Crook (2009), Credit scoring with macroeconomic variables using survival analysis','https://doi.org/10.1057/jors.2008.130'),
('Peng & Lessmann (2026), Incorporating data drift to perform survival analysis on credit risk, arXiv v2','https://arxiv.org/html/2601.20533v2'),
('SBA FOIA (snapshot 2026-09-30) / ALFRED historical vintages','https://data.sba.gov/dataset/7a-504-foia')]:
    p(html.escape(title)+f'<br/><link href="{url}" color="#087e73">{url}</link>','small')
path=ROOT/'reports/논문기법_적용결과.pdf'
SimpleDocTemplate(str(path),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=57,bottomMargin=49,title='SBA 논문 기법 적용 결과',author='SBA Loss Lab').build(items,onFirstPage=footer,onLaterPages=footer)
print(path)
