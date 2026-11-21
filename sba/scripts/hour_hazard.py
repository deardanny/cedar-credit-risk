"""Exploratory quarterly competing-event model using completed partial follow-up."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib,time,gc
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression,GammaRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from public_features import GROUPS
from train_loss import money_metrics,amounts
from train import metrics
OUT=ROOT/'reports/hour'

def intervals(a,cutoff,quarters=12):
 cutoff=pd.Timestamp(cutoff);d=a[a.disb.lt(cutoff)].copy();d['is_cal']=d.borrower_key.map(lambda k:int(k[:8],16)%5==0);features=sum((list(v) for v in GROUPS['macro_vintage']),[]);frames=[]
 for q in range(quarters):
  start=d.disb+pd.DateOffset(months=3*q);end=d.disb+pd.DateOffset(months=3*(q+1));alive=(d.co.isna()|d.co.gt(start))&(d.pif.isna()|d.pif.gt(start))
  if q==0:alive=(d.co.isna()|d.co.ge(start))&(d.pif.isna()|d.pif.ge(start))
  good=end.lt(cutoff)&alive
  piece=d.loc[good,features+['amount','loss','borrower_key','is_cal']].copy();piece['age_quarter']=q;piece['interval_end']=end[good]
  co=d.co.le(end)&d.co.ge(start);pif=d.pif.le(end)&d.pif.ge(start);piece['event']=np.where(co[good],1,np.where(pif[good],2,0));piece['gross']=piece.loss.where(piece.event.eq(1),0);frames.append(piece)
 risk=pd.concat(frames,ignore_index=True);assert risk.interval_end.lt(cutoff).all();assert risk.loc[risk.event.eq(1),'gross'].notna().all();assert risk.empty or risk.groupby('borrower_key').is_cal.nunique().max()==1
 return risk

def hazard_pipeline(group):
 nums,cats=GROUPS[group];nums=nums+['age_quarter'];prep=ColumnTransformer([('num','passthrough',nums),('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan),cats)],sparse_threshold=0)
 model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,learning_rate=.06,min_samples_leaf=200,l2_regularization=20,early_stopping=False,categorical_features=[False]*len(nums)+[True]*len(cats),random_state=1618033)
 return Pipeline([('prep',prep),('model',model)])

def severity_pipeline():
 nums,cats=GROUPS['base'];nums=nums+['age_quarter'];prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
 return Pipeline([('prep',prep),('model',GammaRegressor(alpha=1,max_iter=1000))])

def calibration_inputs(p):
 p=np.clip(p,1e-10,1);return np.log(p[:,1:]/p[:,[0]])

def fit_model(risk,cutoff,group):
 cut=pd.Timestamp(cutoff);fit=risk[~risk.is_cal];cal=risk[risk.is_cal&risk.interval_end.ge(cut-pd.DateOffset(months=12))];assert not set(fit.borrower_key)&set(cal.borrower_key)
 model=hazard_pipeline(group);model.fit(fit,fit.event);assert np.array_equal(model.classes_,[0,1,2]);p=model.predict_proba(cal);adjustment=LogisticRegression(C=.1,max_iter=500).fit(calibration_inputs(p),cal.event)
 positive=fit[fit.event.eq(1)];severity=severity_pipeline();ratio=positive.gross/positive.amount;severity.fit(positive,np.maximum(ratio,1e-8));pc=adjustment.predict_proba(calibration_inputs(p));raw=pc[:,1]*severity.predict(cal)*cal.amount.to_numpy();scale=float(cal.gross.sum()/raw.sum())
 bundle={'group':group,'hazard':model,'calibrator':adjustment,'severity':severity,'scale':scale,'virtual_freeze':cutoff,'hazard_months':3,'horizon_months':36}
 audit={'group':group,'cutoff':cutoff,'fit_intervals':len(fit),'cal_intervals':len(cal),'fit_borrowers':fit.borrower_key.nunique(),'cal_borrowers':cal.borrower_key.nunique(),'fit_event_counts':fit.event.value_counts().to_dict(),'cal_event_counts':cal.event.value_counts().to_dict(),'fit_last_interval':str(fit.interval_end.max().date()),'cal_last_interval':str(cal.interval_end.max().date()),'borrower_overlap':0,'severity_zero_floor_count':int(ratio.le(0).sum()),'amount_scale':scale}
 return bundle,audit

def forecast(bundle,d):
 survival=np.ones(len(d));co=np.zeros(len(d));paid=np.zeros(len(d));expected=np.zeros(len(d));hazards=[]
 for q in range(12):
  x=d.copy();x['age_quarter']=q;raw=bundle['hazard'].predict_proba(x);p=bundle['calibrator'].predict_proba(calibration_inputs(raw));conditional=np.maximum(bundle['severity'].predict(x),1e-8)*d.amount.to_numpy()*bundle['scale']
  co+=survival*p[:,1];paid+=survival*p[:,2];expected+=survival*p[:,1]*conditional;survival*=p[:,0];hazards.append(p)
 assert np.allclose(co+paid+survival,1,atol=1e-10);assert (co>=0).all() and (co<=1).all();assert np.isfinite(expected).all() and (expected>0).all()
 return co,paid,survival,expected

def target36(d):
 d=d.copy();d['y']=(d.co.ge(d.disb)&d.co.le(d.disb+pd.DateOffset(months=36))).astype(int);d['target_gross_amount']=d.loss.where(d.y.eq(1),0.0);return d

def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
 started=time.time();a=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl');groups=['base','macro_vintage'];all_y=[];all_loss=[];accum={g:{'prob':[],'amount':[],'folds':[]} for g in groups};audits=[]
 for year in [2016,2017,2018]:
  cutoff=f'{year}-01-01';risk=intervals(a,cutoff);known=set(a.loc[a.disb.lt(cutoff),'borrower_key']);val=target36(a[a.disb.between(cutoff,f'{year}-12-31')&a.app.ge(cutoff)&~a.borrower_key.isin(known)].copy());all_y.append(val.y.to_numpy());all_loss.append(val.target_gross_amount.to_numpy());print('HAZARD DATA',year,len(risk),'intervals',len(val),'validation loans',flush=True)
  for g in groups:
   b,audit=fit_model(risk,cutoff,g);p,paid,alive,e=forecast(b,val);audits.append(audit);accum[g]['prob'].append(p);accum[g]['amount'].append(e);accum[g]['folds'].append({'year':year,'n':len(val),'probability':metrics(val.y,p),'amount':money_metrics(val.target_gross_amount,e)});print('HAZARD VALIDATED',year,g,'elapsed',round(time.time()-started),flush=True)
  del risk;gc.collect()
 candidates=[]
 for g,rows in accum.items():candidates.append({'group':g,'probability':metrics(np.concatenate(all_y),np.concatenate(rows['prob'])),'amount':money_metrics(np.concatenate(all_loss),np.concatenate(rows['amount'])),'folds':rows['folds']})
 chosen=min(candidates,key=lambda c:c['amount']['tweedie_deviance']);selection={'chosen':chosen,'candidates':candidates,'plan_sha256':hashlib.sha256((ROOT/'docs/hour_hazard_plan.md').read_bytes()).hexdigest(),'retrospective_exploration':True};save('hazard_selection.json',selection)
 cutoff='2023-07-01';risk=intervals(a,cutoff);bundle,audit=fit_model(risk,cutoff,chosen['group']);audits.append(audit);joblib.dump(bundle,ROOT/'models/hour_hazard.joblib');save('hazard_training_audit.json',audits);del risk;gc.collect()
 d=pd.read_pickle(ROOT/'data/processed/phase3_new_test.pkl');known=set(a.loc[a.disb.lt(cutoff),'borrower_key']);d=target36(d[d.app.ge(cutoff)&~d.borrower_key.isin(known)].copy());p,paid,alive,e=forecast(bundle,d);base=joblib.load(ROOT/'models/phase3_base.joblib');bp,bc,be=amounts(base,d)
 result={'retrospective_exploration':True,'not_a_new_independent_test':True,'selected_group':chosen['group'],'virtual_freeze':cutoff,'horizon_months':36,'n':len(d),'events':int(d.y.sum()),'probability':metrics(d.y,p),'amount':money_metrics(d.target_gross_amount,e),'same_subset_phase3_base':{'probability':metrics(d.y,bp),'amount':money_metrics(d.target_gross_amount,be)},'competing_probability_mass_max_error':float(np.max(np.abs(p+paid+alive-1))),'mean_paid_probability':float(paid.mean()),'mean_remaining_probability':float(alive.mean()),'elapsed_seconds':time.time()-started,'approved_for_lending':False};save('hazard_results.json',result)
 pd.DataFrame({'row_hash':d.row_hash,'borrower_key':d.borrower_key,'y':d.y,'actual':d.target_gross_amount,'hazard_probability':p,'paid_probability':paid,'remaining_probability':alive,'hazard_amount':e,'baseline_amount':be}).to_csv(ROOT/'data/processed/hour_hazard_predictions.csv',index=False);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
