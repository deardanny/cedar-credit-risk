"""Direct amount objectives selected on old chronological folds only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from train_loss import preprocessing,money_metrics
from investigate import split,describe
OUT=ROOT/'reports/hour'

def configs():
 rows=[]
 for group in ['base','loan_extended','macro_vintage']:
  for kind,alphas in [('tweedie_amount',[.01,.1,1]),('tweedie_ratio',[.001,.01,.1])]:
   for alpha in alphas:rows.append({'id':f'{group}/{kind}/{alpha}','group':group,'kind':kind,'alpha':alpha})
  for leaves,iters,minimum in [(7,160,100),(15,220,200)]:rows.append({'id':f'{group}/poisson/{leaves}','group':group,'kind':'poisson','leaves':leaves,'iterations':iters,'minimum':minimum})
 return rows

def fit_direct(fit,cal,config):
 k=config['kind']
 if k.startswith('tweedie'):m=TweedieRegressor(power=1.5,alpha=config['alpha'],link='log',max_iter=400,tol=1e-7)
 else:m=HistGradientBoostingRegressor(loss='poisson',max_iter=config['iterations'],max_leaf_nodes=config['leaves'],min_samples_leaf=config['minimum'],learning_rate=.04,l2_regularization=20,early_stopping=False,random_state=271828)
 model=Pipeline([('prep',preprocessing(config['group'])),('model',m)])
 if k=='tweedie_ratio':model.fit(fit,fit.target_gross_amount/fit.amount,model__sample_weight=np.sqrt(fit.amount/1000))
 else:model.fit(fit,fit.target_gross_amount/1000)
 bundle={'config':config,'model':model,'scale':1.}
 raw=predict_direct(bundle,cal);bundle['scale']=float(cal.target_gross_amount.sum()/raw.sum())
 return bundle

def predict_direct(bundle,d):
 p=np.maximum(bundle['model'].predict(d),1e-12)
 p=p*d.amount.to_numpy() if bundle['config']['kind']=='tweedie_ratio' else p*1000
 return p*bundle['scale']

def save(name,v):(OUT/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
 start=time.time();a=pd.read_pickle(ROOT/'data/processed/phase3_development.pkl');cs=configs();accum={c['id']:[] for c in cs};actual=[];folds=[]
 for year,fit_end,calyear in [('2016','2011-12-31','2012'),('2017','2012-12-31','2013'),('2018','2013-12-31','2014')]:
  fit,cal,val=split(a,'2010-01-01',fit_end,calyear+'-01-01',calyear+'-12-31',year+'-01-01',year+'-12-31');actual.append(val.target_gross_amount.to_numpy());folds.append({'year':year,'fit':describe(fit),'cal':describe(cal),'validation':describe(val)})
  for config in cs:
   b=fit_direct(fit,cal,config);pred=predict_direct(b,val);accum[config['id']].append(pred);print('DIRECT FOLD',year,config['id'],'deviance',round(money_metrics(val.target_gross_amount,pred)['tweedie_deviance'],5),'seconds',round(time.time()-start),flush=True)
 y=np.concatenate(actual);candidates=[{'config':c,'metrics':money_metrics(y,np.concatenate(accum[c['id']]))} for c in cs];ranked=sorted(candidates,key=lambda c:c['metrics']['tweedie_deviance']);top=ranked[:2];mixtures=[]
 for w in [0,.25,.5,.75,1]:
  pred=w*np.concatenate(accum[top[0]['config']['id']])+(1-w)*np.concatenate(accum[top[1]['config']['id']]);mixtures.append({'weight_first':w,'metrics':money_metrics(y,pred)})
 mixture=min(mixtures,key=lambda v:v['metrics']['tweedie_deviance']);selected={'selected_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_count':len(cs),'candidates':candidates,'top_two':top,'mixture':mixture,'mixtures':mixtures,'folds':folds,'selection_uses':'2016/2017/2018 only','2023Q3_already_seen_in_previous_phase':True,'plan_sha256':hashlib.sha256((ROOT/'docs/hour_plan.md').read_bytes()).hexdigest()};save('direct_selection.json',selected)
 fit=a[a.disb.between('2010-01-01','2018-12-31')].copy();cal=a[a.disb.between('2019-01-01','2019-12-31')&~a.borrower_key.isin(set(fit.borrower_key))].copy();bundles=[]
 for i,row in enumerate(top):
  b=fit_direct(fit,cal,row['config']);bundles.append(b);joblib.dump(b,ROOT/f'models/hour_direct_{i}.joblib');print('FROZEN DIRECT',i,row['config']['id'],flush=True)
 # Already-seen cohort: explicitly retrospective, never a fresh holdout claim.
 d=pd.read_pickle(ROOT/'data/processed/phase3_new_test.pkl');known=set(a.loc[a.disb.between('2010-01-01','2019-12-31'),'borrower_key']);d=d[~d.borrower_key.isin(known)].copy();preds=[predict_direct(b,d) for b in bundles];w=mixture['weight_first'];p=w*preds[0]+(1-w)*preds[1]
 results={'retrospective_confirmation':True,'not_a_new_independent_test':True,'new_selection_not_based_on_2023Q3':True,'cohort':describe(d),'components':[{'config':b['config'],'scale':b['scale'],'metrics':money_metrics(d.target_gross_amount,pred)} for b,pred in zip(bundles,preds)],'weight_first':w,'mixture_metrics':money_metrics(d.target_gross_amount,p),'selection_sha256':hashlib.sha256((OUT/'direct_selection.json').read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start,'approved_for_lending':False}
 save('direct_results.json',results);pd.DataFrame({'row_hash':d.row_hash,'borrower_key':d.borrower_key,'actual':d.target_gross_amount,'prediction':p}).to_csv(ROOT/'data/processed/hour_direct_predictions.csv',index=False)
 print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
