"""Fixed 72-month cumulative gross-amount diagnostic with competing repayment."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
from hour_hazard import intervals,fit_model,calibration_inputs
from hour_metrics import money_metrics
from train import metrics
OUT=ROOT/'reports/hour'
def main():
 start=time.time();a=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl');cutoff='2020-07-01';risk=intervals(a,cutoff,quarters=24);print('LONG RISK INTERVALS',len(risk),flush=True);bundle,audit=fit_model(risk,cutoff,'base');bundle['horizon_months']=72;joblib.dump(bundle,ROOT/'models/hour_long.joblib')
 (OUT/'long_training_audit.json').write_text(json.dumps(audit,indent=2)+'\n');known=set(a.loc[a.disb.lt(cutoff),'borrower_key']);d=a[a.disb.between('2020-07-01','2020-09-30')&a.app.ge(cutoff)&~a.borrower_key.isin(known)].copy();assert (d.disb+pd.DateOffset(months=72)).max()<=pd.Timestamp('2026-09-30')
 survival=np.ones(len(d));co=np.zeros(len(d));paid=np.zeros(len(d));expected=np.zeros(len(d));curve=[];max_error=0
 for q in range(24):
  x=d.copy();x['age_quarter']=q;hp=bundle['calibrator'].predict_proba(calibration_inputs(bundle['hazard'].predict_proba(x)));severity=np.maximum(bundle['severity'].predict(x),1e-8)*d.amount.to_numpy()*bundle['scale'];co+=survival*hp[:,1];paid+=survival*hp[:,2];expected+=survival*hp[:,1]*severity;survival*=hp[:,0];max_error=max(max_error,float(np.max(np.abs(co+paid+survival-1))))
  if (q+1)%4==0:
   months=(q+1)*3;y=(d.co.ge(d.disb)&d.co.le(d.disb+pd.DateOffset(months=months))).astype(int);actual=d.loss.where(y.eq(1),0);curve.append({'months':months,'probability':metrics(y,co),'amount':money_metrics(actual,expected),'mean_paid_probability':float(paid.mean()),'mean_remaining_probability':float(survival.mean())})
 y=(d.co.ge(d.disb)&d.co.le(d.disb+pd.DateOffset(months=72))).astype(int);actual=d.loss.where(y.eq(1),0)
 eligible=a[(a.disb+pd.DateOffset(months=72)).lt(pd.Timestamp(cutoff))].copy();past_y=eligible.co.ge(eligible.disb)&eligible.co.le(eligible.disb+pd.DateOffset(months=72));baseline_rate=float(eligible.loss.where(past_y,0).sum()/eligible.amount.sum());baseline=money_metrics(actual,d.amount*baseline_rate)
 rng=np.random.default_rng(7202026);groups=list(pd.Series(np.arange(len(d))).groupby(d.borrower_key.to_numpy()).apply(list));ratios=[];actual_array=actual.to_numpy()
 for _ in range(500):
  ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]);den=actual_array[ix].sum()
  if den>0:ratios.append(expected[ix].sum()/den)
 assert max_error<1e-10;assert np.all(np.diff([v['amount']['predicted_total'] for v in curve])>=0)
 result={'horizon_months':72,'retrospective_exploration':True,'not_a_new_independent_test':True,'virtual_freeze':cutoff,'test_window':['2020-07-01','2020-09-30'],'n':len(d),'events':int(y.sum()),'curve':curve,'baseline':baseline,'baseline_training_n':len(eligible),'predicted_actual_ratio_95_ci':np.quantile(ratios,[.025,.975]).tolist(),'bootstrap_replicates':len(ratios),'competing_mass_max_error':max_error,'cumulative_prediction_nondecreasing':True,'plan_sha256':hashlib.sha256((ROOT/'docs/hour_long_plan.md').read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start,'not_lifetime_net_loss':True,'approved_for_lending':False}
 (OUT/'long_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');pd.DataFrame({'row_hash':d.row_hash,'borrower_key':d.borrower_key,'actual':actual,'prediction':expected,'probability':co,'paid_probability':paid,'remaining_probability':survival}).to_csv(ROOT/'data/processed/hour_long_predictions.csv',index=False);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
