"""Compare SBA FOIA term values in two snapshots; never match on outcomes/term."""
from pathlib import Path
import argparse,csv,hashlib,io,json,zipfile
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--output',type=Path)
parser.add_argument('--keep-matches',action='store_true')
args=parser.parse_args()
ROOT=args.root
OUT=args.output or ROOT/'reports/term_snapshot_audit'
OUT.mkdir(parents=True,exist_ok=True)
FIELDS=['AsOfDate','Program','BorrName','BorrStreet','BorrCity','BorrState','BorrZip','BankName','BankFDICNumber','BankNCUANumber','GrossApproval','ApprovalDate','FirstDisbursementDate','TermInMonths','LoanStatus','PaidInFullDate','ChargeOffDate','GrossChargeOffAmount','RevolverStatus']
KEYS=['borrower_key','app','disb','amount','program']
DATE_MAP={'ApprovalDate':'app','FirstDisbursementDate':'disb','PaidInFullDate':'pif','ChargeOffDate':'co'}

def load(f,name,expected_date,model_rows=None):
    d=pd.read_csv(f,usecols=FIELDS,dtype=str,keep_default_na=False)
    assert set(d.AsOfDate)=={expected_date},set(d.AsOfDate)
    d['source_file']=name;d['source_row']=np.arange(2,len(d)+2)
    norm=lambda s:s.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)
    b=norm(d.BorrName)+'|'+norm(d.BorrStreet)+'|'+d.BorrZip.str[:5]
    d['borrower_key']=b.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    d['program']=d.Program.str.strip()
    d['amount']=pd.to_numeric(d.GrossApproval,errors='coerce')
    d['term']=pd.to_numeric(d.TermInMonths,errors='coerce')
    d['status']=d.LoanStatus.str.replace(' ','',regex=False).str.upper()
    d['loss']=pd.to_numeric(d.GrossChargeOffAmount,errors='coerce')
    for raw,out in DATE_MAP.items():d[out]=pd.to_datetime(d[raw],errors='coerce')
    valid=d.app.notna()&d.disb.notna()&d.amount.gt(0)&d.BorrName.str.strip().ne('')&d.BorrStreet.str.strip().ne('')
    ambiguous=d.duplicated(KEYS,keep=False)
    d['model_cohort']=d.source_row.isin(model_rows.get(name,set())) if model_rows is not None else False
    info={'rows':len(d),'valid_identity_rows':int(valid.sum()),'ambiguous_valid_rows':int((valid&ambiguous).sum()),'eligible_unique_rows':int((valid&~ambiguous).sum())}
    return d.loc[valid&~ambiguous].copy(),info

def summarize(d):
    valid=d.term_june.notna()&d.term_sept.notna()
    changed=~(d.term_june.eq(d.term_sept)|(d.term_june.isna()&d.term_sept.isna()))
    transition=[]
    for (a,b),s in d.groupby(['status_june','status_sept']):
        c=changed.loc[s.index]
        transition.append({'june_status':a,'september_status':b,'matched':len(s),'term_changed':int(c.sum()),'term_changed_pct':float(c.mean()*100),'term_shortened':int((s.term_sept<s.term_june).sum()),'term_extended':int((s.term_sept>s.term_june).sum()),'multiple12_to_nonmultiple':int((s.term_june.gt(0)&s.term_sept.gt(0)&s.term_june.mod(12).eq(0)&s.term_sept.mod(12).ne(0)).sum())})
    return {'matched_loans':len(d),'term_changed':int(changed.sum()),'term_changed_pct':float(changed.mean()*100) if len(d) else None,'both_terms_present':int(valid.sum()),'shortened':int((d.term_sept<d.term_june).sum()),'extended':int((d.term_sept>d.term_june).sum()),'status_transitions':transition}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify_examples(d):
    names=['The Block Distilling Company LLC','GREATER SERVICES WACO LLC','PMPZ Realty, LLC']
    ex=d[d.strict_identity&d.model_cohort_sept&d.BorrName_sept.isin(names)&d.status_june.eq('EXEMPT')&d.status_sept.eq('CHGOFF')&d.term_june.ne(d.term_sept)].copy()
    assert len(ex)==3 and set(ex.BorrName_sept)==set(names)
    original_rows={}
    with zipfile.ZipFile(ROOT/'data/raw/Archive.zip') as z:
        for suffix in ['june','sept']:
            for name,g in ex.groupby('source_file_'+suffix):
                needed=set(g['source_row_'+suffix])
                source=z.open(name) if suffix=='june' else open(ROOT/'data/external/sba_new'/name,'rb')
                with source,io.TextIOWrapper(source,encoding='utf-8-sig') as f:
                    for record,row in enumerate(csv.DictReader(f),2):
                        if record in needed:original_rows[(suffix,name,record)]=row
                        if record>=max(needed):break
    for _,r in ex.iterrows():
        old=original_rows[('june',r.source_file_june,r.source_row_june)]
        new=original_rows[('sept',r.source_file_sept,r.source_row_sept)]
        for col in ['BorrName','BorrStreet','BorrCity','BorrState','BorrZip','BankName','BankFDICNumber','BankNCUANumber','ApprovalDate','FirstDisbursementDate','GrossApproval','Program']:
            assert old[col].strip().upper()==new[col].strip().upper(),(r.BorrName_sept,col)
        assert float(old['TermInMonths'])==r.term_june and float(new['TermInMonths'])==r.term_sept
        assert old['LoanStatus'].replace(' ','')=='EXEMPT' and new['LoanStatus'].replace(' ','')=='CHGOFF'
        assert old['ChargeOffDate']=='' and pd.Timestamp(new['ChargeOffDate'])==r.co_sept
    cols=['BorrName_sept','BorrState_sept','app','disb','amount','term_june','term_sept','status_june','status_sept','co_sept','source_file_june','source_row_june','source_file_sept','source_row_sept']
    ex[cols].to_csv(OUT/'verified_examples.csv',index=False)
    return ex,{'verified_examples':len(ex),'independently_read_raw_records':len(original_rows),'independent_parser':'csv.DictReader','source_row_convention':'CSV record ordinal plus 1 header record; first data record=2, not physical text line if quoted newlines exist','selection':'Illustrative cases selected after inspection, not a random sample; aggregate statistics use every eligible match.'}

def write_evidence(d,output):
    import joblib
    from openpyxl import load_workbook
    current=d[d.strict_identity&d.model_cohort_sept]
    charged=current[current.status_june.eq('EXEMPT')&current.status_sept.eq('CHGOFF')]
    changed=charged[charged.term_june.ne(charged.term_sept)].copy()
    elapsed=(changed.co_sept-changed.disb).dt.days/30.4375
    output['elapsed_duration_diagnostic']={'population':'strict current model cohort, EXEMPT to CHGOFF, term changed','n':len(changed),'days_per_month':30.4375,'new_term_within_3_months_of_elapsed':int((changed.term_sept-elapsed).abs().le(3).sum()),'term_reduction_within_3_months_of_elapsed':int((changed.term_june-changed.term_sept-elapsed).abs().le(3).sum()),'interpretation':'The field is not uniformly actual elapsed time to charge-off. A remaining-term-like pattern is exploratory, not proof of an SBA calculation rule.'}
    assert charged.co_june.isna().all()
    assert charged.co_sept.between('2026-07-01','2026-09-30').all()
    output['validation']={'all_new_chargeoffs_dated_in_snapshot_interval':True,'new_chargeoffs_checked':len(charged),'ambiguous_linkage_keys_excluded':True,'outcome_and_term_not_used_in_linkage':True}
    ex,verified=verify_examples(d);output['validation'].update(verified)
    wb=load_workbook(ROOT/'data/raw/7a_504_foia_data_dictionary.xlsx',read_only=True,data_only=True)
    ws=wb['7(a) Data Dictionary'];assert ws['A25'].value=='TermInMonths'
    output['dictionary']={'file':'data/raw/7a_504_foia_data_dictionary.xlsx','sheet':ws.title,'range':'A25:B25','definition':ws['B25'].value,'original_immutable_term_explicitly_stated':False};wb.close()
    model=ROOT/'models/origination_v2_2024.joblib';before=sha(model);bundle=joblib.load(model)
    selected={}
    for kind in ['probability','severity']:
        selected[kind]=[c for name,t,cols in bundle[kind].named_steps['prep'].transformers_ if name!='remainder' for c in cols]
        assert not any('term' in c.lower() for c in selected[kind])
    assert before==sha(model)
    output['model_check']={'artifact':str(model.relative_to(ROOT)),'sha256_before':before,'sha256_after':sha(model),'term_used':False,'selected_features':selected,'model_modified':False}
    output['decision']={'TermInMonths':'Exclude from origination training unless an original-contract value with decision-time provenance is obtained.','term_months':'Same exclusion applies to the derived copy.','confirmed':'The published field changes after origination; changes also occur in the interval when new charge-offs are recorded.','not_proven':'Two snapshots do not establish the exact update timestamp, the causal mechanism, or the original contractual value of any loan. The June value is not assumed original.','scope':'7(a) FY2010 onward; current project cohort and broader raw-source matched sample. No conclusion about every 504 row.'}
    output['official_sources']=[{'url':'https://data.sba.gov/dataset/7a-504-foia','supports':'Quarterly FOIA updates and source downloads.'},{'url':'https://data.sba.gov/sites/default/files/uploaded_resources/7a_504_foia_data_dictionary.xlsx','supports':'Field dictionary; local supplied workbook inspected.'},{'url':'https://legacy.sba.gov/about-sba/sba-locations/loan-guaranty-centers/commercial-loan-service-centers/7a-express-servicing','supports':'E-Tran permits changes to maturity date/months; it does not specify the FOIA field computation.'}]
    paths=[ROOT/'data/raw/Archive.zip',ROOT/'data/raw/7a_504_foia_data_dictionary.xlsx',ROOT/'data/external/sba_new/FOIA_7a_FY2010_FY2019_asof_260930.csv',ROOT/'data/external/sba_new/FOIA_7a_FY2020_Present_asof_260930.csv',ROOT/'data/processed/hour_short_development.pkl',ROOT/'data/processed/hour_medium_holdout.pkl',ROOT/'src/origination_model.py']
    output['source_hashes']={str(p.relative_to(ROOT)):sha(p) for p in paths}
    (OUT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    stats=output['current_model_cohort_strict_matched']
    lines=['# SBA 대출 기간 원본 대조 감사','', '**결론: `TermInMonths`를 최초 계약 당시의 불변 값으로 취급할 수 없다. 실제로 사후 변경된다.**','',
    '2026-06-30 Archive.zip 원본과 2026-09-30 SBA 공식 갱신본을 비교했다. 대출 승인 시점의 원본 계약서는 두 자료에 포함되지 않는다. 따라서 6월 값도 최초 계약 기간이라고 가정하지 않았다.','',
    '## 우리 모델 데이터에서 확인한 결과','',f'- 대상 데이터: 개발/평가 후보 494,386건. 그중 엄격하게 연결된 {stats["matched_loans"]:,}건을 비교했다.',f'- 기간 변경: {stats["term_changed"]:,}건 ({stats["term_changed_pct"]:.3f}%). 모두 2024년까지 승인된 대출이므로 신규 승인 과정의 갱신만으로 설명되지 않는다.',
    '- 상태 비공개(EXEMPT) → 상각(CHGOFF): 1,034건 중 317건(30.66%)에서 기간이 변경됐다. 312건은 짧아지고 5건은 길어졌다. 212건은 12의 배수에서 비배수로 바뀌었다.',
    '- 상태 비공개(EXEMPT) → 완납(PIF): 5,988건 중 123건(2.05%)에서 기간이 변경됐다.',
    '- 두 시점 모두 완납(PIF): 297,321건 중 기간 변경 0건. 이는 이 3개월 동안의 관측일 뿐, 완납 이전에 한 번도 바뀌지 않았음을 뜻하지 않는다.',
    '- EXEMPT는 정상 상환을 뜻하지 않는다. 상환 중·연체 등 상세 상태가 비공개인 대출을 포함할 수 있다.','',
    '## 원본 행을 별도 CSV 파서로 재검증한 사례','',
    '| 회사 | 승인일 | 승인 금액(USD) | 6월 기록(개월) | 9월 기록(개월) | 9월에 기록된 상각일 |','|---|---|---:|---:|---:|---|']
    for _,r in ex.iterrows():lines.append(f'| {r.BorrName_sept} | {r.app:%Y-%m-%d} | {r.amount:,.0f} | {r.term_june:g} | {r.term_sept:g} | {r.co_sept:%Y-%m-%d} |')
    lines += ['', '세 사례 모두 6월 EXEMPT에서 9월 CHGOFF로 바뀌었다. 이름·주소·우편번호·은행 정보·승인일·최초 지급일·승인 금액이 같은 원본 행인지 재확인했다. 상각 기록은 회사 폐업 여부와 같지 않다. 사례는 설명을 위해 선택했으며 무작위 표본이 아니다. 파일명과 CSV 레코드 번호는 `verified_examples.csv`에 남겼다.','',
    '## 확인한 사실과 아직 알 수 없는 점','',
    '확인: 이 필드는 대출 이후 바뀌며, 상각이 새로 기록되는 분기에도 변경된다. 따라서 현재 공개 파일의 기간을 과거 심사 시점의 입력으로 쓰면 미래 정보를 섞을 위험이 있다. 친구가 지적한 12개월 배수 패턴은 실제 변경 사례로 뒷받침된다.','',
    '미확인: 상각 처리 순간에 자동으로 덮어쓰는지, 그 전에 만기 변경·연체 관리 과정에서 바뀌는지, 단순 정정인지까지는 두 스냅샷으로 판별할 수 없다. 이 자료에는 필드 변경 이력이 없다.','',
    '특히 “상각될 때까지 실제 걸린 개월 수로 바뀐다”는 설명은 일반 규칙으로 확인되지 않았다. 기간이 바뀐 신규 상각 317건 중 9월 기간이 지급~상각 경과기간과 ±3개월 이내인 건은 15건뿐이다. 반면 기간 감소량이 경과기간과 ±3개월 이내인 건은 267건이어서 잔여 기간과 비슷한 패턴이 관측된다. 이것 역시 공식 산식의 증명은 아니다.','',
    '## 공식 문서와 모델 조치','',
    '- 첨부 데이터 사전 `7(a) Data Dictionary!A25:B25`는 `Length of loan term`이라고만 정의한다. 최초 계약값·수정 불가라고 명시하지 않는다.',
    '- [SBA 공식 공개 페이지](https://data.sba.gov/dataset/7a-504-foia)는 분기별 갱신을 설명한다.',
    '- [SBA 공식 E-Tran 설명](https://legacy.sba.gov/about-sba/sba-locations/loan-guaranty-centers/commercial-loan-service-centers/7a-express-servicing)은 만기일 또는 개월 수를 변경할 수 있다고 안내한다. 이 문서 자체가 FOIA의 산식을 설명하는 것은 아니다.',
    '- 현재 Cedar `origination_v2_2024.joblib`의 확률·손실 예측 파이프라인을 직접 읽어 `TermInMonths`/`term_months`가 입력에 없는 것을 확인했다. 모델은 변경하지 않았다.',
    '- 이 필드와 복사본 `term_months`는 최초 계약값의 출처와 시점이 검증될 때까지 신규 대출 심사 모델에서 계속 제외한다. 이 결론이 다른 모든 입력의 누수 부재를 보증하지는 않는다.','',
    '## 연결 방법과 한계','',
    '공개된 고유 대출 ID가 없으므로 차주 이름·주소·우편번호로 만든 해시 + 승인일 + 최초 지급일 + 승인 금액 + 프로그램으로 연결했다. 어느 한쪽에 중복 키가 있으면 전부 제외했다. 여기에 원문 이름·주소·도시·주·우편번호·은행명·FDIC/NCUA 번호 일치까지 요구한 엄격 표본을 본문에 사용했다. 기간과 결과는 연결 키에 사용하지 않았다. 동일 조건의 별도 대출 가능성을 수학적으로 배제할 수는 없지만 중복 및 불일치 사례를 보수적으로 제거했다.','',
    '더 넓은 원본(7(a), FY2010 이후)에서도 엄격 연결 771,167건 중 4,050건이 변경됐다. 이 수치를 우리 모델 표본 478,008건과 혼용하면 안 된다. 연결되지 않은 대출은 변경 여부를 판단하지 않았다. 504 또는 FY2010 이전 자료 전체를 감사한 것은 아니다.','',
    '## 재현','', '```sh', '.venv/bin/python scripts/audit_term_snapshots.py', '```','',
    '`summary.json`에는 모집단별 전체 집계·입력 해시·실제 모델 입력 목록·검증 결과가 있고, `term_changes.csv`에는 전체 원본의 변경 건과 모델 표본 포함 여부가 있다. 원본과 모델은 읽기 전용으로 검사했다.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')

def main():
    cohort=pd.concat([pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl')[['source_file','source_row']],pd.read_pickle(ROOT/'data/processed/hour_medium_holdout.pkl')[['source_file','source_row']]])
    model_rows={name:set(g.source_row) for name,g in cohort.groupby('source_file')}
    joined=[];sources=[]
    with zipfile.ZipFile(ROOT/'data/raw/Archive.zip') as z:
        for period in ['FY2010_FY2019','FY2020_Present']:
            oldname=f'FOIA_7a_{period}_asof_260630.csv';newname=f'FOIA_7a_{period}_asof_260930.csv'
            with z.open(oldname) as f:old,oi=load(f,oldname,'2026-06-30')
            new,ni=load(ROOT/'data/external/sba_new'/newname,newname,'2026-09-30',model_rows)
            m=old.merge(new,on=KEYS,suffixes=('_june','_sept'),validate='one_to_one')
            m['strict_identity']=True
            for c in ['BorrName','BorrStreet','BorrCity','BorrState','BorrZip','BankName','BankFDICNumber','BankNCUANumber']:
                m['strict_identity']&=m[c+'_june'].str.strip().str.upper().eq(m[c+'_sept'].str.strip().str.upper())
            sources.append({'old_file':oldname,'new_file':newname,'old':oi,'new':ni,'matched':len(m),'strict_matched':int(m.strict_identity.sum())})
            # No borrower street addresses are retained in audit output.
            m.drop(columns=[c for c in m if c.startswith(('BorrStreet','BorrCity','BorrZip'))],inplace=True)
            joined.append(m)
            print(json.dumps(sources[-1]),flush=True)
    d=pd.concat(joined,ignore_index=True)
    if args.keep_matches:d.to_pickle(OUT/'matched.pkl')
    changed=~(d.term_june.eq(d.term_sept)|(d.term_june.isna()&d.term_sept.isna()))
    output={'snapshots':['2026-06-30','2026-09-30'],'sources':sources,'linkage_is_not_public_loan_id':True,'identity_keys':KEYS,'strict_extra_fields':['BorrName','BorrStreet','BorrCity','BorrState','BorrZip','BankName','BankFDICNumber','BankNCUANumber'],'all_matched':summarize(d),'strict_matched':summarize(d[d.strict_identity]),'current_model_cohort_matched':summarize(d[d.model_cohort_sept]),'current_model_cohort_strict_matched':summarize(d[d.model_cohort_sept&d.strict_identity])}
    write_evidence(d,output)
    cols=['borrower_key','BorrName_sept','BorrState_sept','app','disb','amount','term_june','term_sept','status_june','status_sept','co_june','co_sept','pif_june','pif_sept','loss_june','loss_sept','RevolverStatus_sept','strict_identity','model_cohort_sept','source_file_june','source_row_june','source_file_sept','source_row_sept']
    d.loc[changed,cols].to_csv(OUT/'term_changes.csv',index=False)
    print(json.dumps({'output':str(OUT),'current_model_cohort_strict_matched':output['current_model_cohort_strict_matched'],'validation':output['validation'],'model_term_used':output['model_check']['term_used']},indent=2),flush=True)

if __name__=='__main__':main()
