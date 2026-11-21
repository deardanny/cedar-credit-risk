"""Locked medium-horizon experiment, distinct from the 36-month target."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib,argparse,time
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from train_loss import severity_model,severity_predict,probability_pipeline,fit_bundle,amounts
from hour_metrics import money_metrics
from train import metrics,calibrate,predict
OUT=ROOT/'reports/hour'
GROUPS=['base','loan_extended','macro_vintage']
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def describe(d):return {'n':len(d),'events':int(d.y.sum()),'from':str(d.disb.min().date()),'through':str(d.disb.max().date()),'labels_mature_by':str((d.disb+pd.DateOffset(months=24)).max().date())}
def split(a,year):
 cutoff=pd.Timestamp(f'{year}-01-01');fit=a[a.disb.between('2010-01-01',f'{year-4}-12-31')].copy();cal=a[a.disb.between(f'{year-3}-01-01',f'{year-3}-12-31')&~a.borrower_key.isin(set(fit.borrower_key))].copy();known=set(fit.borrower_key)|set(cal.borrower_key);val=a[a.disb.between(f'{year}-01-01',f'{year}-12-31')&a.app.ge(cutoff)&~a.borrower_key.isin(known)].copy()
 assert (fit.disb+pd.DateOffset(months=24)).max()<cutoff;assert (cal.disb+pd.DateOffset(months=24)).max()<cutoff;assert not set(val.borrower_key)&known
 return fit,cal,val

def train():
 start=time.time();a=pd.read_pickle(ROOT/'data/processed/hour_medium_development.pkl');accum={};ys=[];losses=[];basepred=[];folds=[]
 for year in [2018,2019,2020,2021]:
  fit,cal,val=split(a,year);ys.append(val.y.to_numpy());losses.append(val.target_gross_amount.to_numpy());basepred.append(val.amount.to_numpy()*cal.target_gross_amount.sum()/cal.amount.sum());folds.append({'year':year,'fit':describe(fit),'cal':describe(cal),'validation':describe(val)})
  for group in GROUPS:
   sev={s:severity_model(group,s,fit) for s in ['constant','gamma']}
   for pk in ['logistic','boosting']:
    model=probability_pipeline(group,pk);model.fit(fit,fit.y);c=calibrate(model,cal);pc=predict(model,c,cal);pv=predict(model,c,val)
    for sk,m in sev.items():
     scale=float(cal.target_gross_amount.sum()/(pc*severity_predict(m,cal)*cal.amount.to_numpy()).sum());p=pv*severity_predict(m,val)*val.amount.to_numpy()*scale;key=group+'/'+pk+'/'+sk;row=accum.setdefault(key,{'group':group,'probability_kind':pk,'severity_kind':sk,'ps':[],'es':[],'folds':[]});row['ps'].append(pv);row['es'].append(p);row['folds'].append({'year':year,'probability':metrics(val.y,pv),'amount':money_metrics(val.target_gross_amount,p)})
    print('MEDIUM FOLD',year,group,pk,'elapsed',round(time.time()-start),flush=True)
 candidates=[];y=np.concatenate(ys);loss=np.concatenate(losses)
 for row in accum.values():row['probability']=metrics(y,np.concatenate(row.pop('ps')));row['amount']=money_metrics(loss,np.concatenate(row.pop('es')));candidates.append(row)
 chosen=min(candidates,key=lambda v:v['amount']['tweedie_deviance']);best={g:min([c for c in candidates if c['group']==g],key=lambda v:v['amount']['tweedie_deviance']) for g in GROUPS}
 selection={'selected_at_utc':datetime.now(timezone.utc).isoformat(),'horizon_months':24,'chosen':chosen,'best_by_group':best,'candidates':candidates,'folds':folds,'baseline':money_metrics(loss,np.concatenate(basepred)),'reserved_targets_not_opened':not (ROOT/'data/processed/hour_medium_holdout.pkl').exists(),'plan_sha256':hashlib.sha256((ROOT/'docs/hour_plan.md').read_bytes()).hexdigest(),'amendment_sha256':hashlib.sha256((ROOT/'docs/hour_short_amendment.md').read_bytes()).hexdigest()};assert selection['reserved_targets_not_opened'];save('medium_selection.json',selection)
 fit=a[a.disb.between('2010-01-01','2021-06-30')].copy();cal=a[a.disb.between('2021-07-01','2022-06-30')&~a.borrower_key.isin(set(fit.borrower_key))].copy();assert (cal.disb+pd.DateOffset(months=24)).max()<pd.Timestamp('2024-07-01')
 for g,row in best.items():
  b=fit_bundle(fit,cal,g,row['probability_kind'],row['severity_kind']);b['selection_sha256']=hashlib.sha256((OUT/'medium_selection.json').read_bytes()).hexdigest();b['baseline_amount_rate']=float(cal.target_gross_amount.sum()/cal.amount.sum());b['baseline_event_rate']=float(cal.y.mean());b['horizon_months']=24;joblib.dump(b,ROOT/f'models/hour_medium_{g}.joblib');print('MEDIUM FROZEN',g,flush=True)
 files=[ROOT/'docs/hour_plan.md',ROOT/'docs/hour_short_amendment.md',ROOT/'scripts/hour_medium_data.py',ROOT/'scripts/hour_metrics.py',ROOT/'scripts/hour_medium.py',OUT/'medium_selection.json',*sorted(ROOT.glob('models/hour_medium_*.joblib'))];save('medium_frozen.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'reserved_targets_not_opened':not (ROOT/'data/processed/hour_medium_holdout.pkl').exists(),'virtual_freeze':'2024-07-01','fit':describe(fit),'cal':describe(cal),'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
 print('READY TO EXTRACT RESERVED 2024 Q3 24-MONTH TARGETS',chosen['group'],chosen['probability_kind'],chosen['severity_kind'],flush=True)

def evaluate():
 start=time.time();s=json.loads((OUT/'medium_selection.json').read_text());f=json.loads((OUT/'medium_frozen.json').read_text());assert f['reserved_targets_not_opened']
 for name,sha in f['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
 a=pd.read_pickle(ROOT/'data/processed/hour_medium_development.pkl');d=pd.read_pickle(ROOT/'data/processed/hour_medium_holdout.pkl');known=set(a.loc[a.disb.le('2022-06-30'),'borrower_key']);before=len(d);d=d[~d.borrower_key.isin(known)].copy();assert d.app.ge('2024-07-01').all();assert (d.disb+pd.DateOffset(months=24)).max()<=pd.Timestamp('2026-09-30');results={};pred=d[['row_hash','borrower_key','app','disb','y','amount','target_gross_amount','state','industry']].copy();chosen=s['chosen']['group']
 for g in GROUPS:
  b=joblib.load(ROOT/f'models/hour_medium_{g}.joblib');p,c,e=amounts(b,d);results[g]={'probability':metrics(d.y,p),'amount':money_metrics(d.target_gross_amount,e),'probability_kind':b['probability_kind'],'severity_kind':b['severity_kind'],'amount_scale':b['amount_scale']};pred[g+'_probability']=p;pred[g+'_amount']=e
  if g==chosen:baseline=money_metrics(d.target_gross_amount,d.amount*b['baseline_amount_rate']);baseline_event=b['baseline_event_rate']
 m=results[chosen];gates={'amount_within_25_percent':abs(m['amount']['predicted_actual_ratio']-1)<=.25,'tweedie_beats_baseline':m['amount']['tweedie_deviance']<baseline['tweedie_deviance'],'auc_at_least_065':m['probability']['roc_auc']>=.65,'probability_within_25_percent':abs(m['probability']['predicted_rate']/m['probability']['observed_rate']-1)<=.25}
 rng=np.random.default_rng(20261119);groups=list(pd.Series(np.arange(len(d))).groupby(d.borrower_key.to_numpy()).apply(list));boot=[];actual=d.target_gross_amount.to_numpy();estimate=pred[chosen+'_amount'].to_numpy();p=pred[chosen+'_probability'].to_numpy();y=d.y.to_numpy()
 from sklearn.metrics import roc_auc_score
 for _ in range(500):
  ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
  if len(np.unique(y[ix]))<2 or actual[ix].sum()<=0:continue
  boot.append([estimate[ix].sum()/actual[ix].sum(),p[ix].mean()/y[ix].mean(),roc_auc_score(y[ix],p[ix])])
 result={'horizon_months':24,'selected_group':chosen,'cohort':describe(d),'rows_before_borrower_purge':before,'repeat_borrowers_removed':before-len(d),'results':results,'baseline':baseline,'baseline_event_rate':baseline_event,'gates':gates,'all_gates_pass':all(gates.values()),'confidence_intervals':{name:np.quantile(np.asarray(boot)[:,i],[.025,.975]).tolist() for i,name in enumerate(['amount_ratio','probability_ratio','roc_auc'])},'bootstrap_replicates':500,'low_event_count':int(d.y.sum())<100,'selected_before_reserved_targets_opened':True,'used_for_reselection':False,'approved_for_lending':False,'not_the_36_month_target':True,'evaluation_utc':datetime.now(timezone.utc).isoformat(),'evaluation_seconds':time.time()-start};save('medium_results.json',result);pred.to_csv(ROOT/'data/processed/hour_medium_predictions.csv',index=False);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');args=parser.parse_args();evaluate() if args.evaluate else train()
