"""Post-hoc family ensemble, weights selected only from old validation predictions."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from hour_direct import fit_direct,predict_direct
from hour_granular import fit_bundle as native_fit,score as native_score
from train_loss import fit_bundle,amounts,money_metrics
from investigate import split
from hour_audit import deviance
OUT=ROOT/'reports/hour'
def main():
 a=pd.read_pickle(ROOT/'data/processed/hour_granular_development.pkl');ds=json.loads((OUT/'direct_selection.json').read_text());ns=json.loads((OUT/'granular_selection.json').read_text())['best_by_group']['base8'];dw=ds['mixture']['weight_first'];predictions={'direct':[],'original':[],'native':[]};loss=[];years=[]
 for year,end,calyear in [('2016','2011-12-31','2012'),('2017','2012-12-31','2013'),('2018','2013-12-31','2014')]:
  fit,cal,val=split(a,'2010-01-01',end,calyear+'-01-01',calyear+'-12-31',year+'-01-01',year+'-12-31');direct=[predict_direct(fit_direct(fit,cal,row['config']),val) for row in ds['top_two']];predictions['direct'].append(dw*direct[0]+(1-dw)*direct[1]);original=fit_bundle(fit,cal,'base','boosting','poisson');predictions['original'].append(amounts(original,val)[2]);native=native_fit('base8',ns['leaves'],ns['severity_kind'],fit,cal);predictions['native'].append(native_score(native,val)[2]);loss.append(val.target_gross_amount.to_numpy());years.extend([year]*len(val));print('ENSEMBLE VALIDATION',year,flush=True)
 p={k:np.concatenate(v) for k,v in predictions.items()};y=np.concatenate(loss);years=np.asarray(years);candidates=[]
 for family in ['original','native']:
  for weight in [0,.25,.5,.75,1]:
   e=weight*p['direct']+(1-weight)*p[family];candidates.append({'family':family,'direct_weight':weight,'amount':money_metrics(y,e)})
 chosen=min(candidates,key=lambda v:v['amount']['tweedie_deviance']);lopo=[]
 for held in ['2016','2017','2018']:
  take=years!=held;ranked=[]
  for c in candidates:
   e=c['direct_weight']*p['direct']+(1-c['direct_weight'])*p[c['family']];ranked.append((money_metrics(y[take],e[take])['tweedie_deviance'],c,e))
  _,c,e=min(ranked,key=lambda v:v[0]);lopo.append({'excluded_validation_year':held,'chosen_family':c['family'],'chosen_direct_weight':c['direct_weight'],'excluded_year_metrics':money_metrics(y[~take],e[~take])})
 selection={'chosen':chosen,'candidates':candidates,'leave_one_year_out':lopo,'plan_sha256':hashlib.sha256((ROOT/'docs/hour_ensemble_plan.md').read_bytes()).hexdigest(),'design_created_after_2023Q3_seen':True,'weights_fitted_to_2016_2018_only':True};(OUT/'ensemble_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 d=pd.read_pickle(ROOT/'data/processed/hour_granular_test.pkl');known=set(a.loc[a.disb.between('2010-01-01','2019-12-31'),'borrower_key']);d=d[~d.borrower_key.isin(known)].copy();direct=[predict_direct(joblib.load(ROOT/f'models/hour_direct_{i}.joblib'),d) for i in [0,1]];de=dw*direct[0]+(1-dw)*direct[1];bp,bc,base=amounts(joblib.load(ROOT/'models/phase3_base.joblib'),d);native=native_score(joblib.load(ROOT/'models/hour_granular_base8.joblib'),d)[2];w=chosen['direct_weight'];e=w*de+(1-w)*(base if chosen['family']=='original' else native);actual=d.target_gross_amount.to_numpy();bdev=deviance(actual,base);edev=deviance(actual,e);rng=np.random.default_rng(1132026);groups=list(pd.Series(np.arange(len(d))).groupby(d.borrower_key.to_numpy()).apply(list));boot=[]
 for _ in range(1000):
  ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]);boot.append([1-edev[ix].sum()/bdev[ix].sum(),e[ix].sum()/actual[ix].sum()])
 result={'design_created_after_2023Q3_seen':True,'not_a_new_independent_test':True,'selected_family':chosen['family'],'selected_direct_weight':w,'n':len(d),'amount':money_metrics(actual,e),'base_amount':money_metrics(actual,base),'paired_tweedie_improvement_95_ci':np.quantile(np.asarray(boot)[:,0],[.025,.975]).tolist(),'amount_ratio_95_ci':np.quantile(np.asarray(boot)[:,1],[.025,.975]).tolist(),'observed_tweedie_improvement':float(1-edev.sum()/bdev.sum()),'approved_for_lending':False};(OUT/'ensemble_results.json').write_text(json.dumps(result,indent=2)+'\n');pd.DataFrame({'row_hash':d.row_hash,'borrower_key':d.borrower_key,'actual':actual,'prediction':e}).to_csv(ROOT/'data/processed/hour_ensemble_predictions.csv',index=False);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
