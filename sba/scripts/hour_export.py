"""Export the 72-month research curve and aggregate experiment evidence locally."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from export_loss import clean,export_pipeline
from hour_hazard import calibration_inputs
OUT=ROOT/'reports/hour'
def save(p,v):p.write_text(json.dumps(clean(v),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def native_export(pipe):
 prep=pipe.named_steps['prep'];model=pipe.named_steps['model'];encoder=prep.named_transformers_['cat'];trees=[]
 for step in model._predictors:
  group=[]
  for tree in step:
   nodes=[{'leaf':bool(n['is_leaf']),'value':float(n['value']),'feature':int(n['feature_idx']),'threshold':float(n['num_threshold']),'left':int(n['left']),'right':int(n['right']),'missing_left':bool(n['missing_go_to_left']),'categorical':bool(n['is_categorical']),'bitset':int(n['bitset_idx'])} for n in tree.nodes]
   group.append({'nodes':nodes,'bits':tree.raw_left_cat_bitsets.astype(int).tolist()})
  trees.append(group)
 return {'numeric_features':list(prep.transformers_[0][2]),'categorical_features':list(prep.transformers_[1][2]),'categories':encoder.categories_,'inner_categories':model._preprocessor.named_transformers_['encoder'].categories_,'tree_input_order':'categorical_then_numeric','baseline':model._baseline_prediction.ravel(),'trees':trees}
def curve(bundle,d):
 remaining=np.ones(len(d));co=np.zeros(len(d));paid=np.zeros(len(d));amount=np.zeros(len(d));out=[]
 for q in range(24):
  x=d.copy();x['age_quarter']=q;p=bundle['calibrator'].predict_proba(calibration_inputs(bundle['hazard'].predict_proba(x)));conditional=np.maximum(bundle['severity'].predict(x),1e-8)*d.amount.to_numpy()*bundle['scale'];co+=remaining*p[:,1];paid+=remaining*p[:,2];amount+=remaining*p[:,1]*conditional;remaining*=p[:,0]
  if (q+1)%4==0:out.append({'months':(q+1)*3,'probability':co.copy(),'paid_probability':paid.copy(),'remaining_probability':remaining.copy(),'expected_amount':amount.copy()})
 return out
def main():
 b=joblib.load(ROOT/'models/hour_long.joblib');a=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl');fit=a[a.disb.lt('2020-07-01')];hazard=native_export(b['hazard']);model={'hazard':hazard,'calibration_coef':b['calibrator'].coef_,'calibration_intercept':b['calibrator'].intercept_,'severity':export_pipeline(b['severity'],'gamma'),'scale':b['scale'],'horizon_months':72,'virtual_freeze':'2020-07-01','target':'cumulative snapshot-recorded gross charge-off amount','approved_for_lending':False,'loan_ranges':{'amount':[float(fit.amount.min()),float(fit.amount.max())],'rate':[float(fit.initial_rate.min()),float(fit.initial_rate.max())],'jobs':[0,int(np.expm1(fit.log_jobs.max()).round())]},'support_percentiles':{c:fit[c].quantile([.005,.995]).tolist() for c in ['log_amount','initial_rate','log_jobs']}}
 save(ROOT/'web/dist/hour_long_model.json',model)
 names=['direct_results','medium_results','hazard_results','granular_results','ensemble_results','long_results','uncertainty_audit','snapshot_revision_audit','horizon_feasibility','horizon_coverage','granular_mapping','validation'];evidence={name:json.loads((OUT/(name+'.json')).read_text()) for name in names}
 for name in ['long_robustness','final_validation']:
  if (OUT/(name+'.json')).exists():evidence[name]=json.loads((OUT/(name+'.json')).read_text())
 evidence['phase3']=json.loads((ROOT/'reports/phase3/new_test_results.json').read_text());evidence['experiment_counts']={'direct':24,'medium':12,'hazard_feature_groups':2,'granular':24,'long_fixed':1,'completed_configurations':63,'twelve_month_attempt':'failed feasibility; not counted as completed','ensemble_weight_comparisons':10};evidence['run_budget']=json.loads((OUT/'run_budget.json').read_text());save(ROOT/'web/dist/hour_results.json',evidence)
 rng=np.random.default_rng(72024);nums=[n for n in hazard['numeric_features'] if n!='age_quarter'];cats=hazard['categorical_features'];fixtures=[]
 for i in range(48):
  f={n:float(rng.uniform(*fit[n].quantile([.05,.95]).tolist())) for n in nums}
  for n,levels in zip(cats,hazard['categories']):f[n]=str(levels[int(rng.integers(len(levels)))])
  if i%8==0:f[nums[i%len(nums)]]=None
  if i%11==0:f[cats[i%len(cats)]]='Never-seen synthetic category'
  amount=float(rng.uniform(10000,2000000));d=pd.DataFrame([f]);d[nums]=d[nums].astype(float);d['amount']=amount;rows=curve(b,d);fixtures.append({'features':f,'amount':amount,'curve':[{k:(float(v[0]) if isinstance(v,np.ndarray) else v) for k,v in row.items()} for row in rows]})
 save(ROOT/'tests/hour_curve_fixtures.json',fixtures);print('Exported 72-month curve, 48 synthetic cases, aggregate evidence.')
if __name__=='__main__':main()
