"""Export the frozen loss model, aggregate context and synthetic parity fixtures."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from train_loss import amounts
from enrich_public import external_tables
from public_features import GROUPS
from features import INDUSTRIES

def clean(obj):
 if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
 if isinstance(obj,(list,tuple,np.ndarray)):return [clean(v) for v in obj]
 if isinstance(obj,(np.floating,float)):return float(obj) if np.isfinite(obj) else None
 if isinstance(obj,np.integer):return int(obj)
 return obj

def save(path,obj):path.write_text(json.dumps(clean(obj),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

def export_pipeline(pipe,kind):
 prep=pipe.named_steps['prep'];num=prep.named_transformers_['num'];cat=prep.named_transformers_['cat'];model=pipe.named_steps['model']
 out={'kind':kind,'numeric_features':list(prep.transformers_[0][2]),'categorical_features':list(prep.transformers_[1][2]),'medians':num.named_steps['impute'].statistics_,'means':num.named_steps['scale'].mean_,'scales':num.named_steps['scale'].scale_,'categories':cat.categories_}
 if hasattr(model,'coef_'):
  out.update(coef=np.asarray(model.coef_).ravel(),intercept=float(np.asarray(model.intercept_).ravel()[0]))
 else:
  trees=[]
  for predictors in model._predictors:
   nodes=[]
   for n in predictors[0].nodes:
    nodes.append({'leaf':bool(n['is_leaf']),'value':float(n['value']),'feature':int(n['feature_idx']),'threshold':float(n['num_threshold']),'left':int(n['left']),'right':int(n['right']),'missing_left':bool(n['missing_go_to_left'])})
   trees.append(nodes)
  out.update(baseline=float(model._baseline_prediction.ravel()[0]),trees=trees)
 return out

def bundle_json(b,fit):
 group=b['group']
 nums,cats=GROUPS[group]
 prob=export_pipeline(b['probability'],b['probability_kind']);prob['calibration_intercept']=float(b['calibrator'].intercept_[0]);prob['calibration_slope']=float(b['calibrator'].coef_[0,0])
 sev={'kind':'constant','value':b['severity']} if isinstance(b['severity'],float) else export_pipeline(b['severity'],b['severity_kind'])
 model={'group':group,'probability':prob,'severity':sev,'amount_scale':b['amount_scale'],'selection_sha256':b['selection_sha256'],'feature_count':len(nums)+len(cats),'numeric_ranges':{c:[fit[c].min(),fit[c].max()] for c in nums},'default_features':{c:float(fit[c].median()) for c in nums}|{c:str(fit[c].mode().iloc[0]) for c in cats},'loan_ranges':{'amount':[float(fit.amount.min()),float(fit.amount.max())],'rate':[float(fit.initial_rate.min()),float(fit.initial_rate.max())],'jobs':[0,int(np.expm1(fit.log_jobs.max()).round())]}}
 model['form_categories']={k:sorted(fit[k].dropna().unique().tolist()) for k in ['state','business_type','naics3','processing_method']};model['industries']=INDUSTRIES
 return model

def main():
 out=ROOT/'web/dist';selection=json.loads((ROOT/'reports/phase3/selection_before_new_test.json').read_text());group=selection['chosen']['group']
 a=pd.read_pickle(ROOT/'data/processed/phase3_development.pkl');fit=a[a.disb.between('2010-01-01','2018-12-31')]
 bundles={g:joblib.load(ROOT/f'models/phase3_{g}.joblib') for g in GROUPS};models={g:bundle_json(b,fit) for g,b in bundles.items()}
 save(out/'loss_model.json',models[group]);save(out/'loss_models.json',models)
 national,regional,cbp=external_tables()
 context={'national':national.set_index('quarter').to_dict('index'),'regional':{q:g.drop(columns='quarter').set_index('state').to_dict('index') for q,g in regional.groupby('quarter')},'cbp':{str(y):{s:g.drop(columns=['state','cbp_year']).set_index('industry').to_dict('index') for s,g in v.groupby('state')} for y,v in cbp.groupby('cbp_year')}}
 save(out/'public_context.json',context)
 report=json.loads((ROOT/'reports/phase3/new_test_results.json').read_text());report['mapping']=json.loads((ROOT/'reports/phase3/new_test_join_audit.json').read_text());report['feature_counts']={g:len(n)+len(c) for g,(n,c) in GROUPS.items()};report['selection_validation']={g:{'amount':r['amount_metrics'],'probability':r['probability_metrics']} for g,r in selection['best_by_group'].items()};save(out/'loss_results.json',report)
 # Entirely synthetic combinations: no source rows or names are exported.
 rng=np.random.default_rng(314159);fixtures=[]
 for g,model in models.items():
  nums,cats=GROUPS[g];prob=model['probability'];bounds={c:fit[c].quantile([.05,.95]).tolist() for c in nums}
  for i in range(60):
   f=model['default_features'].copy()
   for c in nums:
    lo,hi=bounds[c];f[c]=float(rng.uniform(lo,hi)) if pd.notna(lo+hi) else None
   for c,levels in zip(cats,prob['categories']):f[c]=str(levels[int(rng.integers(len(levels)))])
   if i%7==0:f[nums[i%len(nums)]]=None
   if i%11==0:f[cats[i%len(cats)]]='Unseen synthetic category'
   amount=float(rng.uniform(10000,2000000));frame=pd.DataFrame([f]);frame[nums]=frame[nums].astype(float);frame['amount']=amount;p,c,e=amounts(bundles[g],frame)
   fixtures.append({'group':g,'features':f,'amount':amount,'probability':float(p[0]),'conditional_amount':float(c[0]),'expected_amount':float(e[0])})
 save(ROOT/'tests/loss_fixtures.json',fixtures)
 print('Exported frozen groups:',list(models),'selected:',group,'synthetic cases:',len(fixtures))
if __name__=='__main__':main()
