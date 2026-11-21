"""Join public series by approval-time vintage, state, industry and conservative year."""
import sys,json,zipfile,argparse,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from features import transform_raw,assign_target,INDUSTRIES
from public_features import STATE_FIPS,MACRO_IDS,MACRO,CBP,GROUPS

def external_tables():
 rows=[]
 for f in sorted((ROOT/'data/external/alfred').glob('*.csv')):
  vintage=pd.Timestamp(f.name[:10]);d=pd.read_csv(f);d['date']=pd.to_datetime(d.iloc[:,0]);stamp=vintage.strftime('%Y%m%d')
  for c in d.columns[1:-1]:
   assert c.endswith('_'+stamp),c
   s=pd.Series(pd.to_numeric(d[c],errors='coerce').to_numpy(),index=d.date).dropna().sort_index()
   assert len(s) and s.index.max()<vintage,(c,s.index.max(),vintage)
   latest=s.index.max();v=float(s.iloc[-1]);past=s.get(latest-pd.DateOffset(months=12),np.nan)
   rows.append({'quarter':str(vintage.date()),'series':c[:-(len(stamp)+1)],'value':v,'year_change':v-past,'year_growth':(v/past-1)*100 if pd.notna(past) and past!=0 else np.nan,'observation_date':str(latest.date()),'age_days':int((vintage-latest).days)})
 series=pd.DataFrame(rows);assert not series.duplicated(['quarter','series']).any()
 national=[];regional=[]
 for q,g in series.groupby('quarter'):
  g=g.set_index('series');n={'quarter':q}
  for s,col in [('UNRATE','us_unemployment'),('FEDFUNDS','fed_funds'),('MPRIME','prime_rate')]:n[col]=float(g.loc[s,'value'])
  n['us_unemployment_change']=float(g.loc['UNRATE','year_change']);n['inflation_yoy']=float(g.loc['CPIAUCSL','year_growth']);n['industrial_growth_yoy']=float(g.loc['INDPRO','year_growth']);n['yield_curve']=float(g.loc['GS10','value']-g.loc['TB3MS','value'])
  national.append(n)
  for state in STATE_FIPS:
   s=state+'UR'
   if s in g.index:regional.append({'quarter':q,'state':state,'state_unemployment':float(g.loc[s,'value']),'state_unemployment_change':float(g.loc[s,'year_change'])})
 cbps=[];inverse={v:k for k,v in STATE_FIPS.items()}
 for f in sorted((ROOT/'data/external/cbp').glob('*.zip')):
  year=2000+int(f.name[3:5])
  with zipfile.ZipFile(f) as z:d=pd.read_csv(z.open(z.namelist()[0]),dtype=str,keep_default_na=False)
  d.columns=d.columns.str.lower()
  if 'lfo' in d:d=d[d.lfo.eq('-')].copy()
  d=d[d.naics.str.match(r'^\d{2}----$|^\d{2}-\d{2}$')].copy();d['industry']=d.naics.str[:2].map(INDUSTRIES);d['state']=d.fipstate.str.zfill(2).map(inverse)
  for c in ['emp','ap','est']:
   d[c]=pd.to_numeric(d[c],errors='coerce')
   if c+'_nf' in d:d.loc[d[c+'_nf'].isin(['N','S','D']),c]=np.nan
  if 'empflag' in d:
   bad=d.empflag.str.strip().ne('');d.loc[bad,['emp','ap']]=np.nan
  d=d.dropna(subset=['state','industry']);d=d.groupby(['state','industry'],as_index=False)[['emp','ap','est']].sum(min_count=1);d['cbp_year']=year;cbps.append(d)
 cbp=pd.concat(cbps,ignore_index=True);assert not cbp.duplicated(['cbp_year','state','industry']).any()
 prev=cbp.copy();prev.cbp_year+=1;prev=prev.rename(columns={'emp':'prev_emp','est':'prev_est','ap':'prev_ap'})
 cbp=cbp.merge(prev,on=['cbp_year','state','industry'],how='left',validate='one_to_one')
 cbp['cbp_log_establishments']=np.log1p(cbp.est.where(cbp.est.ge(0)))
 cbp['cbp_employees_per_establishment']=cbp.emp/cbp.est.where(cbp.est.gt(0))
 cbp['cbp_log_pay_per_employee']=np.log((cbp.ap*1000/cbp.emp.where(cbp.emp.gt(0))).where(cbp.ap.gt(0)))
 cbp['cbp_establishment_growth']=cbp.est/cbp.prev_est.where(cbp.prev_est.gt(0))-1
 cbp['cbp_employment_growth']=cbp.emp/cbp.prev_emp.where(cbp.prev_emp.gt(0))-1
 out=ROOT/'reports/phase3';out.mkdir(exist_ok=True)
 series.to_csv(out/'macro_lineage.csv',index=False)
 return pd.DataFrame(national),pd.DataFrame(regional),cbp[['cbp_year','state','industry']+CBP]

def augment(a,national,regional,cbp):
 a=a.copy();a['quarter']=a.app.dt.to_period('Q').dt.start_time.dt.strftime('%Y-%m-%d');a['cbp_year']=a.app.dt.year-3
 assert (pd.to_datetime(a.quarter)<=a.app).all()
 a=a.merge(national,on='quarter',how='left',validate='many_to_one').merge(regional,on=['quarter','state'],how='left',validate='many_to_one').merge(cbp,on=['cbp_year','state','industry'],how='left',validate='many_to_one')
 a['state_unemployment_gap']=a.state_unemployment-a.us_unemployment;a['rate_over_prime']=a.initial_rate-a.prime_rate
 a['log_amount_per_job']=a.log_amount-a.log_jobs
 a['log_unguaranteed_amount']=np.log1p(a.amount*(1-a.guarantee_share))
 a['target_gross_amount']=a.loss.where(a.y.eq(1),0.0)
 assert a.target_gross_amount.notna().all() and a.target_gross_amount.ge(0).all()
 assert not a.row_hash.duplicated().any()
 return a

def original():
 a=pd.read_pickle(ROOT/'data/processed/analytic.pkl');extras=[]
 with zipfile.ZipFile(ROOT/'data/raw/Archive.zip') as z:
  for name in a.source_file.unique():
   need=set(a.loc[a.source_file.eq(name),'source_row']);offset=0
   for raw in pd.read_csv(z.open(name),dtype=str,keep_default_na=False,usecols=['ProcessingMethod','NaicsCode','GrossApproval'],chunksize=100000):
    raw['source_row']=np.arange(offset+2,offset+len(raw)+2);offset+=len(raw);raw=raw[raw.source_row.isin(need)].copy();raw['source_file']=name;extras.append(raw)
 extra=pd.concat(extras,ignore_index=True);a=a.merge(extra,on=['source_file','source_row'],how='left',validate='one_to_one');assert a.GrossApproval.notna().all()
 a['amount']=pd.to_numeric(a.GrossApproval);a['processing_method']=a.ProcessingMethod.str.strip().replace('','Unknown');a['naics3']=a.NaicsCode.str[:3].where(a.NaicsCode.str.match(r'^\d{3}'),'Unknown')
 return a.drop(columns=['GrossApproval','ProcessingMethod','NaicsCode'])

def new_test():
 assert (ROOT/'reports/phase3/selection_before_new_test.json').exists();frames=[];counts=[];snapshot=pd.Timestamp('2026-09-30')
 for f in sorted((ROOT/'data/external/sba_new').glob('*.csv')):
  offset=0
  for raw in pd.read_csv(f,dtype=str,keep_default_na=False,chunksize=100000):
   raw['source_row']=np.arange(offset+2,offset+len(raw)+2);offset+=len(raw)
   disb=pd.to_datetime(raw.FirstDisbursementDate,errors='coerce');raw=raw[disb.between('2023-07-01','2023-09-30')].copy()
   if raw.empty:continue
   d=pd.DataFrame(index=raw.index);d['source_file']=f.name;d['source_row']=raw.source_row
   d['row_hash']=pd.util.hash_pandas_object(raw.drop(columns='source_row'),index=False).astype(str)
   b=(raw.BorrName.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)+'|'+raw.BorrStreet.str.upper().str.replace(r'[^A-Z0-9]','',regex=True)+'|'+raw.BorrZip.str[:5]);d['borrower_key']=b.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
   for src,dst in [('ApprovalDate','app'),('FirstDisbursementDate','disb'),('ChargeOffDate','co'),('PaidInFullDate','pif')]:d[dst]=pd.to_datetime(raw[src],errors='coerce')
   d['status']=raw.LoanStatus.str.replace(r'\s+','',regex=True).str.upper();d['loss']=pd.to_numeric(raw.GrossChargeOffAmount,errors='coerce');d['amount']=pd.to_numeric(raw.GrossApproval,errors='coerce');guarantee=pd.to_numeric(raw.SBAGuaranteedApproval,errors='coerce')
   good=raw.RevolverStatus.eq('N')&raw.AsOfDate.eq('2026-09-30')&d.amount.gt(0)&guarantee.ge(0)&guarantee.le(d.amount)
   good&=raw.BorrName.str.strip().ne('')&raw.BorrStreet.str.strip().ne('')&raw.BorrZip.str.strip().ne('')
   d['y']=assign_target(d,snapshot,36);good&=d.y.notna();d=pd.concat([d,transform_raw(raw)],axis=1)
   d['processing_method']=raw.ProcessingMethod.str.strip().replace('','Unknown');d['naics3']=raw.NaicsCode.str[:3].where(raw.NaicsCode.str.match(r'^\d{3}'),'Unknown')
   counts.append({'file':f.name,'in_new_disbursement_window':len(raw),'after_cohort_rules':int(good.sum())});frames.append(d[good])
 d=pd.concat(frames,ignore_index=True).drop_duplicates('row_hash');d['y']=d.y.astype(int)
 (ROOT/'reports/phase3/new_cohort_exclusions.json').write_text(json.dumps(counts,indent=2)+'\n')
 return d

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--new-test',action='store_true');args=parser.parse_args()
 tables=external_tables();a=augment(new_test() if args.new_test else original(),*tables)
 label='new_test' if args.new_test else 'development';a.to_pickle(ROOT/f'data/processed/phase3_{label}.pkl')
 audit={'rows':len(a),'events':int(a.y.sum()),'macro_national_matched_fraction':float(a.us_unemployment.notna().mean()),'macro_state_matched_fraction':float(a.state_unemployment.notna().mean()),'cbp_matched_fraction':float(a.cbp_log_establishments.notna().mean()),'unmatched_state_counts':a.loc[a.state_unemployment.isna(),'state'].value_counts().to_dict(),'missing_by_feature':{c:int(a[c].isna().sum()) for c in MACRO+CBP},'macro_vintage_not_after_approval':bool((pd.to_datetime(a.quarter)<=a.app).all()),'cbp_reference_at_least_3_years_prior':bool((a.cbp_year<=a.app.dt.year-3).all()),'cbp_vintage_verified':False,'target_gross_amount_total':float(a.target_gross_amount.sum()),'event_amount_over_approval_count':int((a.y.eq(1)&a.loss.gt(a.amount+0.01)).sum()),'feature_counts':{k:len(n)+len(c) for k,(n,c) in GROUPS.items()}}
 (ROOT/f'reports/phase3/{label}_join_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(label,json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
