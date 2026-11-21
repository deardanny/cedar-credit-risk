"""Frozen ablations and two-part recorded gross-charge-off amount models."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit,logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,GammaRegressor
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.metrics import mean_tweedie_deviance,mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from public_features import GROUPS
from train import metrics,calibrate,predict
from investigate import split,describe
OUT=ROOT/'reports/phase3'

def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def preprocessing(group):
 nums,cats=GROUPS[group]
 return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)

def probability_pipeline(group,kind):
 m=LogisticRegression(C=.1,max_iter=1500) if kind=='logistic' else HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,learning_rate=.06,l2_regularization=10,min_samples_leaf=100,early_stopping=False,random_state=271828)
 return Pipeline([('prep',preprocessing(group)),('model',m)])

def severity_model(group,kind,fit):
 d=fit[fit.y.eq(1)];ratio=d.loss/d.amount
 if kind=='constant':return float(ratio.mean())
 m=GammaRegressor(alpha=1,max_iter=1500) if kind=='gamma' else HistGradientBoostingRegressor(loss='poisson',max_iter=100,max_leaf_nodes=7,learning_rate=.05,l2_regularization=10,min_samples_leaf=30,early_stopping=False,random_state=271828)
 model=Pipeline([('prep',preprocessing(group)),('model',m)]);model.fit(d,ratio);return model

def severity_predict(model,d):return np.full(len(d),model) if isinstance(model,float) else np.maximum(model.predict(d),1e-8)

def amounts(b,d):
 prob=predict(b['probability'],b['calibrator'],d);conditional=severity_predict(b['severity'],d)*d.amount.to_numpy()*b['amount_scale']
 return prob,conditional,prob*conditional

def money_metrics(y,p):
 y=np.asarray(y,dtype=float);p=np.clip(np.asarray(p,dtype=float),1e-8,None)
 order=np.argsort(-p,kind='stable');top=order[:int(np.ceil(len(p)/10))]
 return {'n':len(y),'actual_total':float(y.sum()),'predicted_total':float(p.sum()),'predicted_actual_ratio':float(p.sum()/y.sum()),'actual_mean':float(y.mean()),'predicted_mean':float(p.mean()),'mae_dollars':float(mean_absolute_error(y,p)),'rmse_dollars':float(np.sqrt(mean_squared_error(y,p))),'tweedie_deviance':float(mean_tweedie_deviance(y/1000,p/1000,power=1.5)),'top_decile_amount_capture':float(y[top].sum()/y.sum())}

def fit_bundle(fit,cal,group,pk,sk):
 model=probability_pipeline(group,pk);model.fit(fit,fit.y);c=calibrate(model,cal);s=severity_model(group,sk,fit)
 pc=predict(model,c,cal);raw=pc*severity_predict(s,cal)*cal.amount.to_numpy();scale=float(cal.target_gross_amount.sum()/raw.sum())
 return {'group':group,'probability_kind':pk,'severity_kind':sk,'probability':model,'calibrator':c,'severity':s,'amount_scale':scale,'features':sum((list(v) for v in GROUPS[group]),[])}

def tune_and_fit():
 OUT.mkdir(exist_ok=True);a=pd.read_pickle(ROOT/'data/processed/phase3_development.pkl')
 fold_specs=[('2016','2011-12-31','2012-01-01','2012-12-31'),('2017','2012-12-31','2013-01-01','2013-12-31'),('2018','2013-12-31','2014-01-01','2014-12-31')]
 accum={};fold_audit=[];all_y=[];all_amount=[];all_baseline=[]
 for year,fit_end,cal_start,cal_end in fold_specs:
  fit,cal,tune=split(a,'2010-01-01',fit_end,cal_start,cal_end,year+'-01-01',year+'-12-31')
  baseline_rate=float(cal.target_gross_amount.sum()/cal.amount.sum());all_y.append(tune.y.to_numpy());all_amount.append(tune.target_gross_amount.to_numpy());all_baseline.append(tune.amount.to_numpy()*baseline_rate)
  fold_audit.append({'year':year,'fit':describe(fit),'calibration':describe(cal),'tuning':describe(tune)})
  for group in GROUPS:
   severities={k:severity_model(group,k,fit) for k in ['constant','gamma','poisson']}
   for pk in ['logistic','boosting']:
    model=probability_pipeline(group,pk);model.fit(fit,fit.y);c=calibrate(model,cal);pc=predict(model,c,cal);pt=predict(model,c,tune);pm=metrics(tune.y,pt)
    for sk,sev in severities.items():
     scale=float(cal.target_gross_amount.sum()/(pc*severity_predict(sev,cal)*cal.amount.to_numpy()).sum());pred=pt*severity_predict(sev,tune)*tune.amount.to_numpy()*scale
     mm=money_metrics(tune.target_gross_amount,pred);key=group+'/'+pk+'/'+sk
     row=accum.setdefault(key,{'group':group,'probability_kind':pk,'severity_kind':sk,'p':[],'e':[],'folds':[],'eligible_for_selection':group!='cbp_exploratory'})
     row['p'].append(pt);row['e'].append(pred);row['folds'].append({'year':year,'probability_metrics':pm,'amount_metrics':mm,'amount_scale':scale})
    print('TUNING FOLD',year,group,pk,flush=True)
 candidates=[];y=np.concatenate(all_y);loss=np.concatenate(all_amount)
 for row in accum.values():
  row['probability_metrics']=metrics(y,np.concatenate(row.pop('p')));row['amount_metrics']=money_metrics(loss,np.concatenate(row.pop('e')));candidates.append(row)
 best_by_group={g:min([c for c in candidates if c['group']==g],key=lambda c:c['amount_metrics']['tweedie_deviance']) for g in GROUPS}
 chosen=min([c for c in candidates if c['eligible_for_selection']],key=lambda c:c['amount_metrics']['tweedie_deviance'])
 selection={'chosen':chosen,'best_by_group':best_by_group,'candidates':candidates,'tuning_baseline':money_metrics(loss,np.concatenate(all_baseline)),'plan_sha256':hashlib.sha256((ROOT/'docs/phase3_plan.md').read_bytes()).hexdigest(),'folds':fold_audit,'new_test_seen':False,'selection_rule':'Lowest pooled 2016/2017/2018 chronological-validation Tweedie deviance among non-CBP candidates; no new-test reselection'}
 save('selection_before_new_test.json',selection)
 # Fit frozen choices with labels fully mature before 2023.
 fit=a[a.disb.between('2010-01-01','2018-12-31')].copy();cal=a[a.disb.between('2019-01-01','2019-12-31')&~a.borrower_key.isin(set(fit.borrower_key))].copy()
 assert (cal.disb+pd.DateOffset(months=36)).max()<pd.Timestamp('2023-01-01');assert (fit.disb+pd.DateOffset(months=36)).max()<pd.Timestamp('2023-01-01')
 for group,row in best_by_group.items():
  b=fit_bundle(fit,cal,group,row['probability_kind'],row['severity_kind']);b['selection_sha256']=hashlib.sha256((OUT/'selection_before_new_test.json').read_bytes()).hexdigest();b['baseline_amount_rate']=float(cal.target_gross_amount.sum()/cal.amount.sum());b['baseline_event_rate']=float(cal.y.mean());joblib.dump(b,ROOT/f'models/phase3_{group}.joblib')
  print('FROZEN MODEL',group,row['probability_kind'],row['severity_kind'],flush=True)
 save('final_fit_audit.json',{'fit':describe(fit),'calibration':describe(cal),'borrower_overlap':0,'model_freeze':'2023-01-01'})
 print('SELECTION SAVED. New test may now be downloaded.',chosen['group'],chosen['probability_kind'],chosen['severity_kind'],flush=True)

def evaluate_new():
 selection=json.loads((OUT/'selection_before_new_test.json').read_text());selection_hash=hashlib.sha256((OUT/'selection_before_new_test.json').read_bytes()).hexdigest();d=pd.read_pickle(ROOT/'data/processed/phase3_new_test.pkl');a=pd.read_pickle(ROOT/'data/processed/phase3_development.pkl')
 known=set(a.loc[a.disb.between('2010-01-01','2019-12-31'),'borrower_key']);before=len(d);d=d[~d.borrower_key.isin(known)].copy()
 assert d.disb.between('2023-07-01','2023-09-30').all();assert (d.disb+pd.DateOffset(months=36)).max()<=pd.Timestamp('2026-09-30')
 chosen=selection['chosen']['group'];results={};predictions=d[['row_hash','borrower_key','y','amount','target_gross_amount','app','disb']].copy()
 for group in GROUPS:
  b=joblib.load(ROOT/f'models/phase3_{group}.joblib');assert b['selection_sha256']==selection_hash
  p,c,e=amounts(b,d);results[group]={'probability':metrics(d.y,p),'amount':money_metrics(d.target_gross_amount,e),'probability_kind':b['probability_kind'],'severity_kind':b['severity_kind'],'amount_scale':b['amount_scale']};predictions[group+'_probability']=p;predictions[group+'_amount']=e
  if group==chosen:selected_bundle=b
 baseline=money_metrics(d.target_gross_amount,d.amount*selected_bundle['baseline_amount_rate']);m=results[chosen]
 gates={'gross_total_within_25_percent':abs(m['amount']['predicted_actual_ratio']-1)<=.25,'tweedie_better_than_constant_rate':m['amount']['tweedie_deviance']<baseline['tweedie_deviance'],'auc_at_least_065':m['probability']['roc_auc']>=.65,'probability_mean_within_25_percent':abs(m['probability']['predicted_rate']/m['probability']['observed_rate']-1)<=.25}
 rng=np.random.default_rng(271828);groups=list(pd.Series(np.arange(len(d))).groupby(d.borrower_key.to_numpy()).apply(list));actual=d.target_gross_amount.to_numpy();est=predictions[chosen+'_amount'].to_numpy();boots=[]
 for _ in range(300):
  ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]);boots.append(est[ix].sum()/actual[ix].sum())
 below=d.target_gross_amount.le(d.amount+.01);sensitivity=money_metrics(actual[below],est[below])
 report={'selected_group':chosen,'new_test_asof':'2026-09-30','new_test_period':['2023-07-01','2023-09-30'],'new_test_rows_before_borrower_purge':before,'repeat_borrower_rows_removed':before-len(d),'new_test':describe(d),'results':results,'baseline':baseline,'research_gates':gates,'all_gates_pass':all(gates.values()),'amount_ratio_95_ci':np.quantile(boots,[.025,.975]).tolist(),'bootstrap_replicates':300,'sensitivity_excluding_amount_over_approval':sensitivity,'amount_over_approval_rows':int((~below).sum()),'selection_sha256':selection_hash,'selected_before_new_data_download':True,'new_test_used_for_reselection':False,'approved_for_lending':False}
 save('new_test_results.json',report);predictions.to_csv(ROOT/'data/processed/phase3_new_predictions.csv',index=False)
 # Only encoded inputs, synthetic/default examples and aggregate metrics are exported later.
 print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
 evaluate_new() if args.evaluate else tune_and_fit()
