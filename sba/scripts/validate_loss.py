"""Recompute temporal, borrower, target and source integrity checks independently."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from train_loss import amounts
from public_features import GROUPS

def main():
 out=ROOT/'reports/phase3';checks={}
 a=pd.read_pickle(ROOT/'data/processed/phase3_development.pkl');d=pd.read_pickle(ROOT/'data/processed/phase3_new_test.pkl');selection=json.loads((out/'selection_before_new_test.json').read_text());report=json.loads((out/'new_test_results.json').read_text());line=pd.read_csv(out/'macro_lineage.csv')
 selection_hash=hashlib.sha256((out/'selection_before_new_test.json').read_bytes()).hexdigest()
 assert selection_hash==report['selection_sha256'];checks['selection_unchanged']=True
 assert selection['plan_sha256']==hashlib.sha256((ROOT/'docs/phase3_plan.md').read_bytes()).hexdigest();checks['plan_unchanged']=True
 frozen=json.loads((out/'model_freeze_manifest.json').read_text());download=json.loads((out/'new_test_source_manifest.json').read_text())
 assert frozen['before_new_test_download'] and pd.Timestamp(frozen['frozen_at_utc'])<pd.Timestamp(download['retrieved_at_utc'])
 for name,sha in frozen['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
 checks['frozen_models_and_code_unchanged']=True
 assert (pd.to_datetime(line.observation_date)<pd.to_datetime(line.quarter)).all();checks['macro_observations_precede_vintage']=len(line)
 for label,df in [('development',a),('new_test',d)]:
  reconstructed=((df.co>=df.disb)&(df.co<=df.disb+pd.DateOffset(months=36))&df.status.eq('CHGOFF')).astype(int)
  assert np.array_equal(reconstructed,df.y);assert np.array_equal(df.loss.where(reconstructed.eq(1),0),df.target_gross_amount)
  assert (pd.to_datetime(df.quarter)<=df.app).all();assert (df.cbp_year==df.app.dt.year-3).all();assert not df.row_hash.duplicated().any()
  checks[label]={'rows':len(df),'target_recomputed':True,'non_event_amount_zero':bool(df.loc[df.y.eq(0),'target_gross_amount'].eq(0).all()),'amounts_above_approval_preserved':int((df.target_gross_amount>df.amount+.01).sum())}
 fit=a[a.disb.between('2010-01-01','2018-12-31')];cal=a[a.disb.between('2019-01-01','2019-12-31')&~a.borrower_key.isin(set(fit.borrower_key))];known=set(fit.borrower_key)|set(cal.borrower_key);test=d[~d.borrower_key.isin(known)].copy()
 assert not set(fit.borrower_key)&set(cal.borrower_key);assert not set(test.borrower_key)&known
 assert (cal.disb+pd.DateOffset(months=36)).max()<pd.Timestamp('2023-01-01');assert (test.disb+pd.DateOffset(months=36)).max()<=pd.Timestamp('2026-09-30');assert len(test)==report['new_test']['n']
 checks['borrower_separation_and_complete_followup']=True
 for name in ['public_source_manifest.json','new_test_source_manifest.json']:
  manifest=json.loads((out/name).read_text());assert not manifest['failures']
  for f in manifest['sources']:assert hashlib.sha256((ROOT/f['file']).read_bytes()).hexdigest()==f['sha256']
  checks[name]=len(manifest['sources'])
 chosen=report['selected_group'];b=joblib.load(ROOT/f'models/phase3_{chosen}.joblib');p,c,e=amounts(b,test);assert np.allclose(p*c,e);assert np.all(np.isfinite(e)) and (e>0).all();assert np.isclose(e.sum(),report['results'][chosen]['amount']['predicted_total'])
 checks['expected_amount_recomputed']=True
 # Drift and descriptive slices do not select or tune any model.
 numeric=GROUPS['macro_vintage'][0];drift=[]
 for key in numeric:
  train=fit[key];new=test[key];lo,hi=train.min(),train.max()
  drift.append({'feature':key,'fit_median':float(train.median()),'test_median':float(new.median()),'fit_missing':float(train.isna().mean()),'test_missing':float(new.isna().mean()),'test_outside_fit_min_max':float(((new<lo)|(new>hi)).mean())})
 pd.DataFrame(drift).to_csv(out/'feature_drift.csv',index=False)
 test['prediction_amount']=e;test['prediction_probability']=p
 slices=[]
 for key in ['state','industry','business_age','processing_method']:
  for value,g in test.groupby(key):
   slices.append({'dimension':key,'value':value,'n':len(g),'events':int(g.y.sum()),'actual_gross':float(g.target_gross_amount.sum()),'predicted_gross':float(g.prediction_amount.sum()),'observed_probability':float(g.y.mean()),'predicted_probability':float(g.prediction_probability.mean()),'low_event_count':int(g.y.sum())<30})
 pd.DataFrame(slices).to_csv(out/'diagnostic_segments.csv',index=False)
 checks['diagnostics_used_for_reselection']=False
 (out/'validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
