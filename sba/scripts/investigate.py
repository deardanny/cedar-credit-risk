"""Locked follow-up validation and retrospective portfolio interpretation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import logit,expit
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train import pipeline,metrics,calibrate

def fit_adjustment(model,d,method):
 raw=np.clip(model.predict_proba(d)[:,1],1e-8,1-1e-8)
 if method=='none':return {'intercept':0.,'slope':1.}
 if method=='intercept':return {'intercept':float(brentq(lambda b:expit(logit(raw)+b).mean()-d.y.mean(),-30,30)),'slope':1.}
 c=calibrate(model,d)
 return {'intercept':float(c.intercept_[0]),'slope':float(c.coef_[0,0])}

def predict_adjusted(model,adj,d):
 p=np.clip(model.predict_proba(d)[:,1],1e-8,1-1e-8)
 return expit(adj['intercept']+adj['slope']*logit(p))

def gates(m,constant):
 return {'auc_at_least_065':m['roc_auc']>=.65,'ap_at_least_1_5_prevalence':m['pr_auc']>=1.5*m['observed_rate'],'top_decile_lift_at_least_2':m['top_decile_lift']>=2,'brier_better_than_frozen_constant':m['brier']<constant['brier'],'mean_prediction_within_25_percent':abs(m['predicted_rate']/m['observed_rate']-1)<=.25}

def split(a,fit_start,fit_end,cal_start,cal_end,test_start,test_end):
 fit=a[a.disb.between(fit_start,fit_end)].copy()
 cal=a[a.disb.between(cal_start,cal_end)&~a.borrower_key.isin(set(fit.borrower_key))].copy()
 test=a[a.disb.between(test_start,test_end)&~a.borrower_key.isin(set(fit.borrower_key)|set(cal.borrower_key))].copy()
 assert (fit.disb+pd.DateOffset(months=36)).max()<test.disb.min()
 assert (cal.disb+pd.DateOffset(months=36)).max()<test.disb.min()
 assert not set(fit.borrower_key)&set(cal.borrower_key)
 assert not set(test.borrower_key)&(set(fit.borrower_key)|set(cal.borrower_key))
 return fit,cal,test

def describe(d):return {'n':len(d),'events':int(d.y.sum()),'from':str(d.disb.min().date()),'through':str(d.disb.max().date()),'labels_mature_by':str((d.disb+pd.DateOffset(months=36)).max().date())}

def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
 out=ROOT/'reports/additional';out.mkdir(exist_ok=True)
 a=pd.read_pickle(ROOT/'data/processed/analytic.pkl')
 plan=(ROOT/'docs/additional_validation_plan.md').read_bytes();planhash=hashlib.sha256(plan).hexdigest()
 fit,cal,tune=split(a,'2010-01-01','2012-12-31','2013-01-01','2013-12-31','2017-01-01','2017-12-31')
 configs={'logistic':{'C':.1},'boosting':{'max_iter':120,'learning_rate':.06,'max_leaf_nodes':7,'l2_regularization':10,'min_samples_leaf':100}}
 experiments=[];constant=metrics(tune.y,np.full(len(tune),cal.y.mean()))
 for kind,params in configs.items():
  model=pipeline(kind,params);model.fit(fit,fit.y)
  for method in ['none','sigmoid','intercept']:
   adj=fit_adjustment(model,cal,method);m=metrics(tune.y,predict_adjusted(model,adj,tune));g=gates(m,constant)
   experiments.append({'model':kind,'adjustment':method,'parameters':params,'calibration':adj,'metrics':m,'gates':g,'pass':all(g.values())})
 eligible=[e for e in experiments if e['pass']]
 chosen=min(eligible or experiments,key=lambda e:(e['metrics']['log_loss'],e['model']!='logistic',e['adjustment']))
 selection={'plan_sha256':planhash,'chosen':chosen,'candidates':experiments,'selection_uses':'2017 only','test_year':2021,'tuning_split':{'fit':describe(fit),'calibration':describe(cal),'tuning':describe(tune)}}
 save(out/'selection_before_holdout.json',selection)
 selection_hash=hashlib.sha256((out/'selection_before_holdout.json').read_bytes()).hexdigest()
 print('Selection locked before holdout:',chosen['model'],chosen['adjustment'],flush=True)
 fit,cal,test=split(a,'2010-01-01','2015-12-31','2016-01-01','2017-12-31','2021-01-01','2021-12-31')
 model=pipeline(chosen['model'],chosen['parameters']);model.fit(fit,fit.y);adj=fit_adjustment(model,cal,chosen['adjustment']);p=predict_adjusted(model,adj,test)
 m=metrics(test.y,p);const=metrics(test.y,np.full(len(test),cal.y.mean()));g=gates(m,const)
 y=test.y.to_numpy();groups=list(pd.Series(np.arange(len(test))).groupby(test.borrower_key.to_numpy()).apply(list));rng=np.random.default_rng(19831119)
 boots=[]
 from sklearn.metrics import roc_auc_score,average_precision_score
 for _ in range(200):
  ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);boots.append([roc_auc_score(y[ix],p[ix]),average_precision_score(y[ix],p[ix]),p[ix].mean()/y[ix].mean()])
 ci={name:np.quantile(np.asarray(boots)[:,i],[.025,.975]).tolist() for i,name in enumerate(['roc_auc','average_precision','predicted_observed_ratio'])}
 holdout={'selection_sha256':selection_hash,'plan_sha256':planhash,'selected_model':chosen['model'],'selected_adjustment':chosen['adjustment'],'metrics':m,'constant':const,'gates':g,'pass':all(g.values()),'calibration':adj,'split':{'fit':describe(fit),'calibration':describe(cal),'holdout':describe(test)},'confidence_intervals':ci,'bootstrap_replicates':200,'previously_profiled_aggregate_counts':True,'previously_used_for_model_fitting_or_performance_selection':False,'not_used_for_reselection':True,'original_web_model_replaced':False}
 save(out/'holdout_2021.json',holdout)
 pd.DataFrame({'row_hash':test.row_hash,'borrower_key':test.borrower_key,'y':y,'prediction':p}).to_csv(ROOT/'data/processed/additional_2021_predictions.csv',index=False)
 joblib.dump({'pipeline':model,'adjustment':adj,'selection_sha256':selection_hash},ROOT/'models/additional_2021.joblib')
 assert hashlib.sha256((out/'selection_before_holdout.json').read_bytes()).hexdigest()==selection_hash
 # Interpret existing test rankings. These quantities are retrospective, not policy thresholds.
 s=pd.read_csv(ROOT/'data/processed/test_predictions.csv',dtype={'row_hash':str});old=json.loads((ROOT/'reports/model_results.json').read_text());kind=old['selected_before_test'];s=s.sort_values(kind,ascending=False,kind='stable')
 total=int(s.y.sum());n=len(s);capacity=[]
 for percentage in range(1,101):
  k=int(np.ceil(n*percentage/100));events=int(s.y.iloc[:k].sum())
  capacity.append({'review_percent':percentage,'review_n':k,'events_found':events,'events_missed':total-events,'non_events_reviewed':k-events,'recall':events/total,'precision':events/k,'lift':(events/k)/(total/n),'random_expected_events':k*total/n})
 pd.DataFrame(capacity).to_csv(out/'review_capacity.csv',index=False)
 t=s.merge(a[['row_hash','log_amount','initial_rate']],on='row_hash',how='left',validate='one_to_one',suffixes=('','_analytic'))
 segments=[]
 for field,bins,labels in [('amount',[-np.inf,100000,500000,np.inf],['≤$100k','$100k–500k','>$500k']),('rate',[-np.inf,5,7.5,10,np.inf],['≤5%','5–7.5%','7.5–10%','>10%'])]:
  value=np.exp(t.log_amount).round(2) if field=='amount' else t.initial_rate
  cat=pd.cut(value,bins,labels=labels).astype(str);cat=cat.where(value.notna(),'Missing')
  for name,idx in cat.groupby(cat).groups.items():
   d=t.loc[idx];segments.append({'dimension':field,'segment':name,'n':len(d),'events':int(d.y.sum()),'predicted_events':float(d[kind].sum()),'observed_rate':float(d.y.mean()),'predicted_rate':float(d[kind].mean()),'low_event_count':int(d.y.sum())<30})
 pd.DataFrame(segments).to_csv(out/'calibration_segments.csv',index=False)
 summary={'original_test':{'n':n,'events':total,'metrics':old['test'][kind]},'capacity':capacity,'segments':segments,'additional_holdout':holdout,'selection_candidates':[{k:e[k] for k in ['model','adjustment','metrics','pass']} for e in experiments],'local_only':True}
 save(ROOT/'web/dist/analysis.json',summary)
 save(out/'checks.json',{'time_maturity_assertions':True,'borrower_separation_assertions':True,'selection_saved_before_holdout':True,'selection_hash_unchanged':True,'capacity_total_reconciles':capacity[-1]['events_found']==total and capacity[-1]['review_n']==n,'calibration_segments_reconcile':all(sum(v['n'] for v in segments if v['dimension']==d)==n for d in ['amount','rate']),'original_web_model_replaced':False})
 print(json.dumps({'holdout_2021':m,'gates':g,'top10':capacity[9]},indent=2),flush=True)

if __name__=='__main__':main()
