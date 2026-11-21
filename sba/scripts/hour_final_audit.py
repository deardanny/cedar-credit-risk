"""Independent source, long-horizon target and categorical-fit audits."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from hour_export import curve
from hour_granular import FEATURES
OUT=ROOT/'reports/hour'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 checks={};m=json.loads((OUT/'credit_sources.json').read_text());raw_count=0
 assert not m['failures'] and len(m['quarters'])==56
 for q in m['quarters']:
  f=ROOT/q['canonical_file'];assert sha(f)==q['canonical_sha256'];d=pd.read_csv(f)
  for s in q['unavailable_series']:assert pd.to_numeric(d[s+'_'+q['quarter'].replace('-','')],errors='coerce').isna().all()
  for r in q['raw_sources']:
   if r['file'] is not None:assert sha(ROOT/r['file'])==r['sha256'];raw_count+=1
 lineage=pd.read_csv(OUT/'credit_lineage.csv');observed=lineage.dropna(subset='observation_date');assert (pd.to_datetime(observed.observation_date)<pd.to_datetime(observed.quarter)).all();checks['credit_quarters_verified']=56;checks['credit_raw_sources_verified']=raw_count;checks['missing_vintages_preserved']=True
 a=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl');cutoff=pd.Timestamp('2020-07-01');known=set(a.loc[a.disb.lt(cutoff),'borrower_key']);d=a[a.disb.between('2020-07-01','2020-09-30')&a.app.ge(cutoff)&~a.borrower_key.isin(known)].copy();assert not set(d.borrower_key)&known;assert (d.disb+pd.DateOffset(months=72)).max()<=pd.Timestamp('2026-09-30')
 result=json.loads((OUT/'long_results.json').read_text());saved=pd.read_csv(ROOT/'data/processed/hour_long_predictions.csv',dtype={'row_hash':str});assert len(d)==result['n']==len(saved);assert set(d.row_hash)==set(saved.row_hash)
 b=joblib.load(ROOT/'models/hour_long.joblib');pred=curve(b,d);assert b['horizon_months']==72;assert result['plan_sha256']==sha(ROOT/'docs/hour_long_plan.md')
 for row,metric in zip(pred,result['curve']):
  y=d.co.ge(d.disb)&d.co.le(d.disb+pd.DateOffset(months=row['months']));actual=d.loss.where(y,0);assert np.isclose(actual.sum(),metric['amount']['actual_total']);assert int(y.sum())==metric['probability']['events'];assert np.isclose(row['expected_amount'].sum(),metric['amount']['predicted_total']);np.testing.assert_allclose(row['probability']+row['paid_probability']+row['remaining_probability'],1,atol=1e-10)
 assert (np.diff(np.stack([r['expected_amount'] for r in pred]),axis=0)>=0).all();checks['six_year_all_targets_and_forecasts_recomputed']=True;checks['six_year_borrower_and_date_separation']=True
 # Reproduce the common-cohort observed horizon curve without using different populations.
 cohort=a[(a.disb+pd.DateOffset(months=72)).le(pd.Timestamp('2026-09-30'))].copy();coverage=json.loads((OUT/'horizon_coverage.json').read_text())
 for row in coverage['curves']:
  target=cohort.co.ge(cohort.disb)&cohort.co.le(cohort.disb+pd.DateOffset(months=row['months']));assert len(cohort)==row['n'];assert int(target.sum())==row['events'];assert np.isclose(cohort.loss.where(target,0).sum(),row['gross_amount'])
 checks['same_cohort_horizon_coverage_recomputed']=True
 a=pd.read_pickle(ROOT/'data/processed/hour_granular_development.pkl');fit=a[a.disb.between('2010-01-01','2018-12-31')];assert (pd.to_datetime(a.quarter)<=a.app).all()
 for group,(_,cats) in FEATURES.items():
  b=joblib.load(ROOT/f'models/hour_granular_{group}.joblib');encoder=b['probability'].named_steps['prep'].named_transformers_['cat']
  for c,levels in zip(cats,encoder.categories_):assert set(levels)==set(fit[c].unique())
  z=encoder.transform(fit[cats]);assert np.nanmax(z)<250
 checks['granular_categories_learned_from_fit_only']=True
 for name,plan in [('granular_selection','hour_granular_plan.md'),('hazard_selection','hour_hazard_plan.md'),('long_results','hour_long_plan.md')]:assert json.loads((OUT/(name+'.json')).read_text())['plan_sha256']==sha(ROOT/'docs'/plan)
 checks['exploratory_plans_unchanged']=True;web=json.loads((OUT/'web_validation.json').read_text());assert web['passed'] and web['max_relative_error']<1e-9;checks['web_288_annual_points_match_python']=True
 (OUT/'final_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
