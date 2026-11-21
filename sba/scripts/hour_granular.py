"""Recover granular origination fields; compare native categorical and credit-cycle models."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,zipfile,hashlib,time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from public_features import GROUPS,MACRO
from train import metrics,calibrate,predict
from train_loss import money_metrics
from investigate import split,describe
OUT=ROOT/'reports/hour'
CREDIT=['business_delinquency','business_delinquency_change','business_chargeoff','business_chargeoff_change','small_firm_tightening','small_firm_tightening_change']
DETAIL=['business_age_detail','naics4','naics6','project_county']
FEATURES={'base8':GROUPS['base'],'loan12':GROUPS['loan_extended'],'granular16':(GROUPS['loan_extended'][0],GROUPS['loan_extended'][1]+DETAIL),'granular_macro27':(GROUPS['macro_vintage'][0],GROUPS['macro_vintage'][1]+DETAIL),'credit_macro29':(GROUPS['macro_vintage'][0]+CREDIT,GROUPS['macro_vintage'][1]),'granular_credit33':(GROUPS['macro_vintage'][0]+CREDIT,GROUPS['macro_vintage'][1]+DETAIL)}

def credits():
 manifest=json.loads((OUT/'credit_sources.json').read_text());assert len(manifest['quarters'])==56 and not manifest['failures'];rows=[];lineage=[]
 for f in sorted((ROOT/'data/external/hour/credit').glob('*.csv')):
  vintage=pd.Timestamp(f.stem);d=pd.read_csv(f);date=pd.to_datetime(d.iloc[:,0]);stamp=vintage.strftime('%Y%m%d');row={'quarter':str(vintage.date())}
  for series,name in [('DRBLACBS','business_delinquency'),('CORBLACBS','business_chargeoff'),('DRTSCIS','small_firm_tightening')]:
   col=series+'_'+stamp;s=pd.Series(pd.to_numeric(d[col],errors='coerce').to_numpy(),index=date).dropna().sort_index()
   if s.empty:
    row[name]=np.nan;row[name+'_change']=np.nan;lineage.append({'quarter':str(vintage.date()),'series':series,'observation_date':None,'value':np.nan,'year_change':np.nan});continue
   assert s.index.max()<vintage;latest=s.index.max();value=float(s.iloc[-1]);previous=s.get(latest-pd.DateOffset(months=12),np.nan);row[name]=value;row[name+'_change']=value-previous;lineage.append({'quarter':str(vintage.date()),'series':series,'observation_date':str(latest.date()),'value':value,'year_change':value-previous})
  rows.append(row)
 pd.DataFrame(lineage).to_csv(OUT/'credit_lineage.csv',index=False);return pd.DataFrame(rows)

def extra(a,original):
 frames=[];fields=['BusinessAge','NaicsCode','ProjectCounty']
 archive=zipfile.ZipFile(ROOT/'data/raw/Archive.zip') if original else None
 try:
  for name in a.source_file.unique():
   needed=set(a.loc[a.source_file.eq(name),'source_row']);offset=0;source=archive.open(name) if archive else ROOT/'data/external/sba_new'/name
   for raw in pd.read_csv(source,dtype=str,keep_default_na=False,usecols=fields,chunksize=100000):
    raw['source_row']=np.arange(offset+2,offset+len(raw)+2);offset+=len(raw);raw=raw[raw.source_row.isin(needed)].copy();raw['source_file']=name;frames.append(raw)
 finally:
  if archive:archive.close()
 x=a.merge(pd.concat(frames,ignore_index=True),on=['source_file','source_row'],how='left',validate='one_to_one');assert x.BusinessAge.notna().all();x['business_age_detail']=x.BusinessAge.str.strip().replace('','Unknown')
 for n in [4,6]:x[f'naics{n}']=x.NaicsCode.str[:n].where(x.NaicsCode.str.match(r'^\d{'+str(n)+'}'),'Unknown')
 county=x.ProjectCounty.str.strip().str.upper().str.replace(r'\s+',' ',regex=True);x['project_county']=(x.state+'|'+county).where(county.ne(''),'Unknown');return x.drop(columns=fields)

def prepare():
 c=credits();a=extra(pd.read_pickle(ROOT/'data/processed/phase3_development.pkl'),True);d=extra(pd.read_pickle(ROOT/'data/processed/phase3_new_test.pkl'),False)
 for name,frame in [('development',a),('test',d)]:
  frame=frame.merge(c,on='quarter',how='left',validate='many_to_one');frame.to_pickle(ROOT/f'data/processed/hour_granular_{name}.pkl')
 (OUT/'granular_mapping.json').write_text(json.dumps({'development_rows':len(a),'test_rows':len(d),'credit_availability':{k:float(a.merge(c,on='quarter',how='left')[k].notna().mean()) for k in CREDIT},'feature_counts':{k:len(n)+len(cat) for k,(n,cat) in FEATURES.items()},'detail_distinct_values':{k:int(a[k].nunique()) for k in DETAIL},'details_unknown_fraction':{k:float(a[k].eq('Unknown').mean()) for k in DETAIL},'not_a_new_independent_test':True},indent=2)+'\n')

def model_pipeline(group,leaves,regression=False):
 nums,cats=FEATURES[group];prep=ColumnTransformer([('num','passthrough',nums),('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan,max_categories=250,min_frequency=100),cats)],sparse_threshold=0);mask=[False]*len(nums)+[True]*len(cats)
 if regression:m=HistGradientBoostingRegressor(loss='poisson',max_iter=100,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=30,l2_regularization=10,early_stopping=False,categorical_features=mask,random_state=271828)
 else:m=HistGradientBoostingClassifier(max_iter=160 if leaves==7 else 220,max_leaf_nodes=leaves,learning_rate=.05 if leaves==7 else .04,min_samples_leaf=100 if leaves==7 else 150,l2_regularization=20,early_stopping=False,categorical_features=mask,random_state=271828)
 return Pipeline([('prep',prep),('model',m)])

def severity(group,kind,fit):
 positive=fit[fit.y.eq(1)];ratio=positive.loss/positive.amount
 if kind=='constant':return float(ratio.mean())
 model=model_pipeline(group,7,True);model.fit(positive,ratio);return model
def sev_predict(model,d):return np.full(len(d),model) if isinstance(model,float) else np.maximum(model.predict(d),1e-8)
def score(bundle,d):
 p=predict(bundle['probability'],bundle['calibrator'],d);conditional=sev_predict(bundle['severity'],d)*d.amount.to_numpy()*bundle['scale'];return p,conditional,p*conditional
def fit_bundle(group,leaves,sk,fit,cal):
 model=model_pipeline(group,leaves);model.fit(fit,fit.y);c=calibrate(model,cal);s=severity(group,sk,fit);pc=predict(model,c,cal);scale=float(cal.target_gross_amount.sum()/(pc*sev_predict(s,cal)*cal.amount.to_numpy()).sum());return {'group':group,'leaves':leaves,'severity_kind':sk,'probability':model,'calibrator':c,'severity':s,'scale':scale}
def save(name,v):(OUT/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 started=time.time();prepare();a=pd.read_pickle(ROOT/'data/processed/hour_granular_development.pkl');ys=[];losses=[];accum={}
 for year,fit_end,calyear in [('2016','2011-12-31','2012'),('2017','2012-12-31','2013'),('2018','2013-12-31','2014')]:
  fit,cal,val=split(a,'2010-01-01',fit_end,calyear+'-01-01',calyear+'-12-31',year+'-01-01',year+'-12-31');ys.append(val.y.to_numpy());losses.append(val.target_gross_amount.to_numpy())
  for group in FEATURES:
   ss={k:severity(group,k,fit) for k in ['constant','poisson']}
   for leaves in [7,15]:
    model=model_pipeline(group,leaves);model.fit(fit,fit.y);c=calibrate(model,cal);pc=predict(model,c,cal);pv=predict(model,c,val)
    for sk,sev in ss.items():
     scale=float(cal.target_gross_amount.sum()/(pc*sev_predict(sev,cal)*cal.amount.to_numpy()).sum());e=pv*sev_predict(sev,val)*val.amount.to_numpy()*scale;key=f'{group}/{leaves}/{sk}';row=accum.setdefault(key,{'group':group,'leaves':leaves,'severity_kind':sk,'ps':[],'es':[],'folds':[]});row['ps'].append(pv);row['es'].append(e);row['folds'].append({'year':year,'amount':money_metrics(val.target_gross_amount,e),'probability':metrics(val.y,pv)})
    print('GRANULAR FOLD',year,group,leaves,'seconds',round(time.time()-started),flush=True)
 candidates=[]
 for row in accum.values():row['probability']=metrics(np.concatenate(ys),np.concatenate(row.pop('ps')));row['amount']=money_metrics(np.concatenate(losses),np.concatenate(row.pop('es')));candidates.append(row)
 best={g:min([c for c in candidates if c['group']==g],key=lambda r:r['amount']['tweedie_deviance']) for g in FEATURES};chosen=min(candidates,key=lambda r:r['amount']['tweedie_deviance']);selection={'chosen':chosen,'best_by_group':best,'candidates':candidates,'plan_sha256':hashlib.sha256((ROOT/'docs/hour_granular_plan.md').read_bytes()).hexdigest(),'selection_uses':'2016/2017/2018 only','retrospective_exploration':True};save('granular_selection.json',selection)
 fit=a[a.disb.between('2010-01-01','2018-12-31')].copy();cal=a[a.disb.between('2019-01-01','2019-12-31')&~a.borrower_key.isin(set(fit.borrower_key))].copy();bundles={}
 for g,row in best.items():
  b=fit_bundle(g,row['leaves'],row['severity_kind'],fit,cal);bundles[g]=b;joblib.dump(b,ROOT/f'models/hour_granular_{g}.joblib');print('GRANULAR FROZEN',g,flush=True)
 d=pd.read_pickle(ROOT/'data/processed/hour_granular_test.pkl');known=set(a.loc[a.disb.between('2010-01-01','2019-12-31'),'borrower_key']);d=d[~d.borrower_key.isin(known)].copy();results={};pred=d[['row_hash','borrower_key','y','target_gross_amount']].copy()
 for g,b in bundles.items():p,c,e=score(b,d);results[g]={'probability':metrics(d.y,p),'amount':money_metrics(d.target_gross_amount,e),'leaves':b['leaves'],'severity_kind':b['severity_kind']};pred[g+'_probability']=p;pred[g+'_amount']=e
 save('granular_results.json',{'selected_group':chosen['group'],'retrospective_exploration':True,'not_a_new_independent_test':True,'n':len(d),'results':results,'elapsed_seconds':time.time()-started,'approved_for_lending':False});pred.to_csv(ROOT/'data/processed/hour_granular_predictions.csv',index=False);print('GRANULAR COMPLETE',chosen['group'],json.dumps(results[chosen['group']]),flush=True)
if __name__=='__main__':main()
