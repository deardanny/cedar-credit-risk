"""Additional historical windows for the unchanged six-year model configuration."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib,gc,time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from hour_hazard import intervals,fit_model
from hour_export import curve
from hour_metrics import money_metrics
from train import metrics
OUT=ROOT/'reports/hour'
def main():
 start=time.time();a=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl');rows=[]
 for year in [2018,2019]:
  cutoff=f'{year}-07-01';risk=intervals(a,cutoff,quarters=24);b,audit=fit_model(risk,cutoff,'base');b['horizon_months']=72;joblib.dump(b,ROOT/f'models/hour_long_{year}.joblib');del risk;gc.collect();known=set(a.loc[a.disb.lt(cutoff),'borrower_key']);d=a[a.disb.between(cutoff,f'{year}-09-30')&a.app.ge(cutoff)&~a.borrower_key.isin(known)].copy();assert (d.disb+pd.DateOffset(months=72)).max()<=pd.Timestamp('2026-09-30');pred=curve(b,d)[-1];y=d.co.ge(d.disb)&d.co.le(d.disb+pd.DateOffset(months=72));actual=d.loss.where(y,0);row={'test_year':year,'n':len(d),'events':int(y.sum()),'amount':money_metrics(actual,pred['expected_amount']),'probability':metrics(y.astype(int),pred['probability']),'training_audit':audit};rows.append(row);print(json.dumps(row),flush=True)
  pd.DataFrame({'row_hash':d.row_hash,'borrower_key':d.borrower_key,'actual':actual,'prediction':pred['expected_amount'],'probability':pred['probability']}).to_csv(ROOT/f'data/processed/hour_long_{year}_predictions.csv',index=False)
 old=json.loads((OUT/'long_results.json').read_text());rows.append({'test_year':2020,'n':old['n'],'events':old['events'],'amount':old['curve'][-1]['amount'],'probability':old['curve'][-1]['probability']});result={'same_configuration_different_time_windows':True,'retrospective_exploration':True,'not_new_independent_evaluation':True,'used_for_reselection':False,'rows':rows,'plan_sha256':hashlib.sha256((ROOT/'docs/hour_long_robustness_plan.md').read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start};(OUT/'long_robustness.json').write_text(json.dumps(result,indent=2)+'\n');print('ADDITIONAL LONG WINDOWS COMPLETE',flush=True)
if __name__=='__main__':main()
