"""Read frozen user inputs, profile every CSV, and prepare anonymous analysis rows."""
import sys, json, zipfile, hashlib, collections, platform
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from features import transform_raw, assign_target, RAW

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    cfg=json.loads((ROOT/'config.json').read_text())
    for key in ['archive','dictionary','requirements_pdf']:
        if not Path(cfg[key]).is_absolute():cfg[key]=str(ROOT/cfg[key])
    for folder in ['data/processed','reports','models','tests','web/dist','docs']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
    inventory=[]; frames=[]; manifest={'sources':[], 'snapshot':cfg['as_of'], 'python':platform.python_version()}
    for key in ['archive','dictionary','requirements_pdf']:
        p=Path(cfg[key]);manifest['sources'].append({'role':key,'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    with zipfile.ZipFile(cfg['archive']) as z:
      for name in z.namelist():
        if not name.endswith('.csv') or name.startswith('__MACOSX'):continue
        s={'file':name,'rows':0,'missing':collections.Counter(),'status':collections.Counter(),'asof':collections.Counter(),'missing_by_fy':{},'schema':[], 'sha256':None}
        h=hashlib.sha256()
        with z.open(name) as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        s['sha256']=h.hexdigest();offset=0
        for d in pd.read_csv(z.open(name),dtype=str,keep_default_na=False,chunksize=100000):
            s['schema']=list(d.columns);s['rows']+=len(d)
            s['missing'].update({c:int(d[c].str.strip().eq('').sum()) for c in d})
            s['status'].update(d.LoanStatus.value_counts().to_dict());s['asof'].update(d.AsOfDate.value_counts().to_dict())
            for fy,g in d.groupby('ApprovalFY'):
                row=s['missing_by_fy'].setdefault(fy,{'n':0, 'missing':collections.Counter()})
                row['n']+=len(g);row['missing'].update({c:int(g[c].eq('').sum()) for c in RAW if c in g})
            if '7a' in name and ('2010' in name or '2020' in name):
                base=d[RAW].copy()
                # Full published-row fingerprints identify indistinguishable repeats; no real loan ID is available.
                base['row_hash']=pd.util.hash_pandas_object(d,index=False).astype(str)
                base['source_file']=name;base['source_row']=np.arange(offset+2,offset+len(d)+2)
                borrower=(d.BorrName.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)+'|'+d.BorrStreet.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)+'|'+d.BorrZip.str[:5])
                base['borrower_key']=borrower.map(lambda v:hashlib.sha256(v.encode()).hexdigest())
                base['borrower_missing']=d.BorrName.str.strip().eq('')|d.BorrStreet.str.strip().eq('')|d.BorrZip.str.strip().eq('')
                for src,dst in [('ApprovalDate','app'),('FirstDisbursementDate','disb'),('ChargeOffDate','co'),('PaidInFullDate','pif')]:base[dst]=pd.to_datetime(d[src],format='%Y-%m-%d',errors='coerce')
                base['status']=d.LoanStatus.str.replace(r'\s+','',regex=True).str.upper()
                base['revolver']=d.RevolverStatus;base['fy']=d.ApprovalFY
                base['loss']=pd.to_numeric(d.GrossChargeOffAmount,errors='coerce')
                base['asof']=d.AsOfDate
                frames.append(base)
            offset+=len(d)
        inventory.append(s);print('Profiled',name,s['rows'],flush=True)
    manifest['csv_inventory']=inventory
    (ROOT/'reports/source_inventory.json').write_text(json.dumps(manifest,indent=2))
    d=pd.concat(frames,ignore_index=True);asof=pd.Timestamp(cfg['as_of']);flow=[]
    def remove(reason,condition):
        nonlocal d
        flow.append({'rule':reason,'before':len(d),'removed':int(condition.sum()),'removed_chgoff':int((condition & d.status.eq('CHGOFF')).sum()),'after':int((~condition).sum())})
        d=d.loc[~condition].copy()
    remove('snapshot_mismatch',d['asof'].ne(cfg['as_of']))
    remove('published_row_duplicate_keep_first',d.row_hash.duplicated())
    remove('not_term_loan',d.revolver.ne('N'))
    remove('cancelled_unfunded_or_unsupported_status',~d.status.isin(['CHGOFF','PIF','EXEMPT']))
    remove('missing_borrower_link_fields',d.borrower_missing)
    amount=pd.to_numeric(d.GrossApproval,errors='coerce');guarantee=pd.to_numeric(d.SBAGuaranteedApproval,errors='coerce')
    remove('invalid_approval_amounts',amount.isna()|amount.le(0)|guarantee.isna()|guarantee.lt(0)|guarantee.gt(amount))
    d['y']=assign_target(d,asof,cfg['horizon_months'])
    remove('invalid_dates_unknown_event_or_incomplete_horizon',d.y.isna())
    remove('outside_study_disbursement_window',d.disb.lt('2010-01-01')|d.disb.gt('2023-06-30'))
    x=transform_raw(d)
    keep=['source_file','source_row','row_hash','borrower_key','app','disb','co','pif','status','fy','y','loss']
    a=pd.concat([d[keep],x],axis=1).reset_index(drop=True);a['y']=a.y.astype(int)
    a.to_pickle(ROOT/'data/processed/analytic.pkl')
    flow.append({'rule':'final_analytic','n':len(a),'events':int(a.y.sum()),'event_rate':float(a.y.mean())})
    (ROOT/'reports/exclusion_flow.json').write_text(json.dumps(flow,indent=2))
    a.groupby(a.disb.dt.year).agg(n=('y','size'),events=('y','sum'),event_rate=('y','mean')).to_csv(ROOT/'reports/cohorts.csv')
    print('Analytic sample',len(a),'events',int(a.y.sum()),flush=True)

if __name__=='__main__':main()
