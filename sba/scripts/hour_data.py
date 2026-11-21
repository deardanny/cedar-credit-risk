"""Separate 12-month development/holdout extraction with approval-time vintages."""
import sys,json,hashlib,argparse
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from features import transform_raw,assign_target
from public_features import STATE_FIPS,MACRO
OUT=ROOT/'reports/hour'

def public_tables():
 manifest=json.loads((OUT/'new_public_sources.json').read_text());assert len(manifest['sources'])==40 and not manifest['failures'],'Finish all additional vintage downloads first'
 old=pd.read_csv(ROOT/'reports/phase3/macro_lineage.csv');rows=[]
 for f in sorted((ROOT/'data/external/hour/alfred').glob('*.csv')):
  vintage=pd.Timestamp(f.name[:10]);d=pd.read_csv(f);date=pd.to_datetime(d.iloc[:,0]);stamp=vintage.strftime('%Y%m%d')
  for col in d.columns[1:]:
   assert col.endswith('_'+stamp)
   s=pd.Series(pd.to_numeric(d[col],errors='coerce').to_numpy(),index=date).dropna().sort_index();assert len(s) and s.index.max()<vintage
   latest=s.index.max();v=float(s.iloc[-1]);past=s.get(latest-pd.DateOffset(months=12),np.nan)
   rows.append({'quarter':str(vintage.date()),'series':col[:-(len(stamp)+1)],'value':v,'year_change':v-past,'year_growth':(v/past-1)*100 if pd.notna(past) and past else np.nan,'observation_date':str(latest.date()),'age_days':int((vintage-latest).days)})
 all_series=pd.concat([old,pd.DataFrame(rows)],ignore_index=True);assert not all_series.duplicated(['quarter','series']).any();assert all_series.groupby('quarter').size().eq(58).all();national=[];regional=[]
 for quarter,g in all_series.groupby('quarter'):
  g=g.set_index('series');n={'quarter':quarter}
  for series,name in [('UNRATE','us_unemployment'),('FEDFUNDS','fed_funds'),('MPRIME','prime_rate')]:n[name]=float(g.loc[series,'value'])
  n['us_unemployment_change']=float(g.loc['UNRATE','year_change']);n['inflation_yoy']=float(g.loc['CPIAUCSL','year_growth']);n['industrial_growth_yoy']=float(g.loc['INDPRO','year_growth']);n['yield_curve']=float(g.loc['GS10','value']-g.loc['TB3MS','value']);national.append(n)
  for state in STATE_FIPS:
   series=state+'UR'
   if series in g.index:regional.append({'quarter':quarter,'state':state,'state_unemployment':float(g.loc[series,'value']),'state_unemployment_change':float(g.loc[series,'year_change'])})
 all_series.to_csv(OUT/'macro_lineage.csv',index=False)
 return pd.DataFrame(national),pd.DataFrame(regional)

def extract(holdout=False):
 if holdout:assert (OUT/'short_frozen.json').exists(),'Freeze before reading reserved targets'
 frames=[];counts=[]
 for path in sorted((ROOT/'data/external/sba_new').glob('*.csv')):
  offset=0
  for raw in pd.read_csv(path,dtype=str,keep_default_na=False,chunksize=100000):
   raw['source_row']=np.arange(offset+2,offset+2+len(raw));offset+=len(raw)
   disb=pd.to_datetime(raw.FirstDisbursementDate,errors='coerce');app=pd.to_datetime(raw.ApprovalDate,errors='coerce')
   window=disb.between('2025-07-01','2025-09-30')&app.ge('2025-07-01') if holdout else disb.between('2010-01-01','2024-06-30')
   raw=raw[window].copy()
   if raw.empty:continue
   d=pd.DataFrame(index=raw.index);d['source_file']=path.name;d['source_row']=raw.source_row;d['row_hash']=pd.util.hash_pandas_object(raw.drop(columns='source_row'),index=False).astype(str)
   key=raw.BorrName.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)+'|'+raw.BorrStreet.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)+'|'+raw.BorrZip.str[:5]
   d['borrower_key']=key.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
   for source,target in [('ApprovalDate','app'),('FirstDisbursementDate','disb'),('ChargeOffDate','co'),('PaidInFullDate','pif')]:d[target]=pd.to_datetime(raw[source],errors='coerce')
   d['status']=raw.LoanStatus.str.replace(r'\s+','',regex=True).str.upper();d['loss']=pd.to_numeric(raw.GrossChargeOffAmount,errors='coerce');d['amount']=pd.to_numeric(raw.GrossApproval,errors='coerce');guarantee=pd.to_numeric(raw.SBAGuaranteedApproval,errors='coerce')
   good=raw.RevolverStatus.eq('N')&raw.AsOfDate.eq('2026-09-30')&d.amount.gt(0)&guarantee.ge(0)&guarantee.le(d.amount)
   good&=raw.BorrName.str.strip().ne('')&raw.BorrStreet.str.strip().ne('')&raw.BorrZip.str.strip().ne('')
   d['y']=assign_target(d,pd.Timestamp('2026-09-30'),12);good&=d.y.notna();d=pd.concat([d,transform_raw(raw)],axis=1)
   d['processing_method']=raw.ProcessingMethod.str.strip().replace('','Unknown');d['naics3']=raw.NaicsCode.str[:3].where(raw.NaicsCode.str.match(r'^\d{3}'),'Unknown');d=d[good].copy();d.y=d.y.astype(int)
   d['target_gross_amount']=d.loss.where(d.y.eq(1),0);assert d.target_gross_amount.notna().all() and d.target_gross_amount.ge(0).all()
   counts.append({'file':path.name,'window_rows':len(raw),'eligible_rows':len(d)});frames.append(d)
 d=pd.concat(frames,ignore_index=True);dup=int(d.row_hash.duplicated().sum());d=d.drop_duplicates('row_hash');label='short_holdout' if holdout else 'short_development'
 (OUT/(label+'_exclusions.json')).write_text(json.dumps({'chunks':counts,'duplicate_rows':dup},indent=2)+'\n')
 return d

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--holdout',action='store_true');args=parser.parse_args();n,r=public_tables();a=extract(args.holdout);a['quarter']=a.app.dt.to_period('Q').dt.start_time.dt.strftime('%Y-%m-%d')
 a=a.merge(n,on='quarter',how='left',validate='many_to_one').merge(r,on=['quarter','state'],how='left',validate='many_to_one');a['state_unemployment_gap']=a.state_unemployment-a.us_unemployment;a['rate_over_prime']=a.initial_rate-a.prime_rate;a['log_amount_per_job']=a.log_amount-a.log_jobs;a['log_unguaranteed_amount']=np.log1p(a.amount*(1-a.guarantee_share))
 assert a.us_unemployment.notna().all();assert (pd.to_datetime(a.quarter)<=a.app).all();assert (a.disb+pd.DateOffset(months=12)).max()<=pd.Timestamp('2026-09-30')
 label='short_holdout' if args.holdout else 'short_development';a.to_pickle(ROOT/f'data/processed/hour_{label}.pkl')
 audit={'rows':len(a),'events':int(a.y.sum()),'horizon_months':12,'from':str(a.disb.min().date()),'through':str(a.disb.max().date()),'national_matched':float(a.us_unemployment.notna().mean()),'state_matched':float(a.state_unemployment.notna().mean()),'missing_by_feature':{c:int(a[c].isna().sum()) for c in MACRO},'event_amount_over_approval':int((a.target_gross_amount>a.amount+.01).sum()),'reserved_holdout':args.holdout,'source_snapshot':'2026-09-30','all_app_after_freeze':bool(a.app.ge('2025-07-01').all()) if args.holdout else None}
 (OUT/(label+'_audit.json')).write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
