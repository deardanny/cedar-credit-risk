"""Audit frozen Cedar origination features without using audited fields as match keys."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();ROOT=args.root;OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from features import transform_raw,RAW
from origination_model import split_training
from origination_experiment import cohort

IDENTITY=['Program','BorrName','BorrStreet','BorrCity','BorrState','BorrZip','BankName','BankFDICNumber','BankNCUANumber']
DATES=['ApprovalDate','FirstDisbursementDate']
OTHER=['AsOfDate','TermInMonths','RevolverStatus','LoanStatus','PaidInFullDate','ChargeOffDate','GrossChargeOffAmount']
NUM_RAW=['GrossApproval','SBAGuaranteedApproval','InitialInterestRate','JobsSupported']
KEYS=['borrower_key','app','disb','program'] # Deliberately omit amount, all predictor fields and outcomes.
STRICT=['BorrName','BorrStreet','BorrCity','BorrState','BorrZip','BankName','BankFDICNumber','BankNCUANumber']
FEATURES=['log_amount','guarantee_share','initial_rate','log_jobs','industry','state','business_age','business_type','log_amount_per_job','log_unguaranteed_amount']

def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
def diff(a,b):
    if pd.api.types.is_numeric_dtype(a):return ~pd.Series(np.isclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True),index=a.index)
    return ~(a.eq(b)|(a.isna()&b.isna()))

def read(f,name,asof,roles):
    d=pd.read_csv(f,usecols=list(dict.fromkeys(IDENTITY+DATES+OTHER+RAW)),dtype=str,keep_default_na=False)
    assert set(d.AsOfDate)=={asof}
    d['source_file']=name;d['source_row']=np.arange(2,len(d)+2)
    norm=lambda s:s.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)
    d['borrower_key']=(norm(d.BorrName)+'|'+norm(d.BorrStreet)+'|'+d.BorrZip.str[:5]).map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    d['program']=d.Program.str.strip()
    for c,n in [('ApprovalDate','app'),('FirstDisbursementDate','disb'),('PaidInFullDate','pif'),('ChargeOffDate','co')]:d[n]=pd.to_datetime(d[c],errors='coerce')
    d['status']=d.LoanStatus.str.replace(r'\s+','',regex=True).str.upper()
    x=transform_raw(d)
    for c in NUM_RAW+['TermInMonths','GrossChargeOffAmount']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d['amount']=d.GrossApproval
    d=pd.concat([d,x],axis=1)
    d['log_amount_per_job']=d.log_amount-d.log_jobs
    d['log_unguaranteed_amount']=np.log1p(d.amount*(1-d.guarantee_share))
    d['role']=d.source_row.map(roles.get(name,{})).fillna('outside_project')
    d['valid_identity']=d.BorrName.str.strip().ne('')&d.BorrStreet.str.strip().ne('')&d.BorrZip.str.strip().ne('')&d.app.notna()&d.disb.notna()
    return d

def link(old,new,keys,strict):
    a=old.valid_identity&~old.duplicated(keys,keep=False);b=new.valid_identity&~new.duplicated(keys,keep=False)
    m=old.loc[a].merge(new.loc[b],on=keys,how='inner',suffixes=('_june','_sept'),validate='one_to_one')
    good=pd.Series(True,index=m.index)
    for c in strict:
        if c not in keys:good&=m[c+'_june'].astype(str).str.strip().str.upper().eq(m[c+'_sept'].astype(str).str.strip().str.upper())
    info={'old_rows':len(old),'new_rows':len(new),'old_eligible_unique':int(a.sum()),'new_eligible_unique':int(b.sum()),'old_ambiguous_rows':int((old.valid_identity&old.duplicated(keys,keep=False)).sum()),'new_ambiguous_rows':int((new.valid_identity&new.duplicated(keys,keep=False)).sum()),'unique_matches':len(m),'strict_matches':int(good.sum())}
    return m[good].copy(),info

def aggregates(d,fields):
    groups={'all_raw_matches':pd.Series(True,index=d.index),'project_matches':d.role_sept.ne('outside_project'),'fit':d.role_sept.eq('fit'),'calibration':d.role_sept.eq('calibration'),'assessment':d.role_sept.eq('assessment')}
    result=[]
    for field in fields:
        changed=diff(d[field+'_june'],d[field+'_sept'])
        for population,mask in groups.items():
            n=int(mask.sum());c=int((mask&changed).sum())
            result.append({'field':field,'population':population,'n':n,'changed':c,'changed_pct':c/n*100 if n else None,'old_missing':int(d.loc[mask,field+'_june'].isna().sum()),'new_missing':int(d.loc[mask,field+'_sept'].isna().sum())})
    return result

def main():
    dev=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl');hold=pd.read_pickle(ROOT/'data/processed/hour_medium_holdout.pkl')
    fit,cal=split_training(dev,'2024-07-01');test=cohort(hold,dev,2024)
    data=pd.concat([dev,hold],ignore_index=True);data['role']='other_project';data.loc[data.row_hash.isin(fit.row_hash),'role']='fit';data.loc[data.row_hash.isin(cal.row_hash),'role']='calibration';data.loc[data.row_hash.isin(test.row_hash),'role']='assessment'
    roles={name:g.set_index('source_row').role.to_dict() for name,g in data.groupby('source_file')}
    metadata={'project_population':len(data),'roles':data.role.value_counts().to_dict(),'main_key':KEYS,'strict_equality_fields':STRICT,'no_public_loan_id':True,'audited_raw_fields':RAW,'audited_transformed_loan_fields':FEATURES,'sources':[]}
    matched=[];date_rows=[];relaxed_rows=[];date_matches=[];relaxed_changes=[]
    with zipfile.ZipFile(ROOT/'data/raw/Archive.zip') as z:
        for period in ['FY2010_FY2019','FY2020_Present']:
            on=f'FOIA_7a_{period}_asof_260630.csv';nn=f'FOIA_7a_{period}_asof_260930.csv'
            with z.open(on) as f:old=read(f,on,'2026-06-30',{})
            new=read(ROOT/'data/external/sba_new'/nn,nn,'2026-09-30',roles)
            m,info=link(old,new,KEYS,STRICT);info['period']=period;metadata['sources'].append(info)
            # Context/target date audit must not condition on those dates.
            dates,di=link(old,new,['borrower_key','amount','program'],STRICT)
            date_matches.append(dates[dates.role_sept.ne('outside_project')].drop(columns=[c for c in dates if c.startswith(('BorrStreet','BorrCity','BorrZip'))]))
            for f in ['app','disb']:
                for pop,mask in [('all',pd.Series(True,index=dates.index)),('project',dates.role_sept.ne('outside_project'))]:
                    changed=diff(dates[f+'_june'],dates[f+'_sept'])
                    date_rows.append({'period':period,'field':f,'population':pop,'n':int(mask.sum()),'changed':int((mask&changed).sum())})
            # Sensitivity: omit borrower address/ZIP and lender equality to expose changes hidden by strict matching.
            relaxed,ri=link(old,new,['BorrName','app','disb','program'],[])
            rc=relaxed[relaxed.role_sept.ne('outside_project')&diff(relaxed.ProjectState_june,relaxed.ProjectState_sept)].copy()
            rc['same_bank_and_amount']=rc.GrossApproval_june.eq(rc.GrossApproval_sept)
            for col in ['BankName','BankFDICNumber','BankNCUANumber']:rc['same_bank_and_amount']&=rc[col+'_june'].eq(rc[col+'_sept'])
            relaxed_changes.append(rc.drop(columns=[c for c in rc if c.startswith(('BorrStreet','BorrCity','BorrZip'))]))
            for f in RAW:
                mask=relaxed.role_sept.ne('outside_project');c=diff(relaxed[f+'_june'],relaxed[f+'_sept'])
                relaxed_rows.append({'period':period,'field':f,'n':int(mask.sum()),'changed':int((mask&c).sum()),'interpretation':'Sensitivity only; weaker identity, unique exact name/date/program match, excludes predictor amount.'})
            keep=[c for c in m if not c.startswith(('BorrStreet','BorrCity','BorrZip'))]
            matched.append(m[keep]);print(json.dumps(info),flush=True)
            del old,new,dates,relaxed,m
    d=pd.concat(matched,ignore_index=True);d[d.role_sept.ne('outside_project')].to_pickle(OUT/'project_matches.pkl')
    dd=pd.concat(date_matches,ignore_index=True);dd.to_pickle(OUT/'date_project_matches.pkl')
    rr=pd.concat(relaxed_changes,ignore_index=True);rr.to_pickle(OUT/'relocation_matches.pkl')
    pd.DataFrame(aggregates(dd,[c for c in RAW+FEATURES if c not in ['GrossApproval']])).to_csv(OUT/'date_linked_feature_summary.csv',index=False)
    rows=aggregates(d,RAW+FEATURES+['TermInMonths']);pd.DataFrame(rows).to_csv(OUT/'snapshot_field_summary.csv',index=False)
    pd.DataFrame(date_rows).to_csv(OUT/'date_audit.csv',index=False);pd.DataFrame(relaxed_rows).to_csv(OUT/'relaxed_identity_summary.csv',index=False)
    changes=[]
    for field in RAW+FEATURES:
        mask=diff(d[field+'_june'],d[field+'_sept'])&d.role_sept.ne('outside_project')
        cols=['BorrName_sept','app','disb','role_sept','status_june','status_sept','co_sept','source_file_june','source_row_june','source_file_sept','source_row_sept']
        a=d.loc[mask,cols].copy();a['field']=field;a['old_value']=d.loc[mask,field+'_june'];a['new_value']=d.loc[mask,field+'_sept'];changes.append(a)
    pd.concat(changes,ignore_index=True).to_csv(OUT/'project_field_changes.csv',index=False)
    metadata['date_linkage_keys']=['borrower_key','amount','program'];metadata['strict_project_matches']=int(d.role_sept.ne('outside_project').sum());metadata['strict_all_matches']=len(d)
    save('snapshot_metadata.json',metadata)
    print(pd.DataFrame(rows).query("population=='project_matches'")[['field','n','changed']].to_string(index=False),flush=True)

if __name__=='__main__':main()
