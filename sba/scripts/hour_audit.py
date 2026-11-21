"""Paired uncertainty, tail concentration, approval timing and snapshot revision audit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts')]
from train_loss import money_metrics
OUT=ROOT/'reports/hour'
def deviance(y,mu,power=1.5):
 y=np.asarray(y)/1000;mu=np.maximum(np.asarray(mu)/1000,1e-12);p=power
 return 2*(np.power(y,2-p)/((1-p)*(2-p))-y*np.power(mu,1-p)/(1-p)+np.power(mu,2-p)/(2-p))
def main():
 original=pd.read_csv(ROOT/'data/processed/phase3_new_predictions.csv',dtype={'row_hash':str,'borrower_key':str});direct=pd.read_csv(ROOT/'data/processed/hour_direct_predictions.csv',dtype={'row_hash':str,'borrower_key':str});d=original.merge(direct[['row_hash','prediction']],on='row_hash',validate='one_to_one');assert len(d)==8519
 actual=d.target_gross_amount.to_numpy();names=['base_amount','loan_extended_amount','macro_vintage_amount','cbp_exploratory_amount','prediction']
 for name in names:d['dev_'+name]=deviance(actual,d[name]);d['se_'+name]=(actual-d[name])**2
 groups=d.groupby('borrower_key').agg(**{'n':('row_hash','size'),'actual':('target_gross_amount','sum')}|{n:(n,'sum') for n in names}|{'dev_'+n:('dev_'+n,'sum') for n in names});rng=np.random.default_rng(20261119);boot={n:[] for n in names};ratios={n:[] for n in names};values=groups.to_numpy();cols=list(groups.columns)
 for _ in range(1000):
  ids=rng.integers(0,len(groups),len(groups));totals=values[ids].sum(axis=0);bdev=totals[cols.index('dev_base_amount')]
  for name in names:
   boot[name].append(1-totals[cols.index('dev_'+name)]/bdev);ratios[name].append(totals[cols.index(name)]/totals[cols.index('actual')])
 comparisons={n:{'observed_tweedie_improvement_vs_base':float(1-d['dev_'+n].sum()/d.dev_base_amount.sum()),'paired_95_ci':np.quantile(boot[n],[.025,.975]).tolist(),'fraction_bootstrap_improved':float(np.mean(np.asarray(boot[n])>0)),'predicted_actual_ratio_95_ci':np.quantile(ratios[n],[.025,.975]).tolist(),'rmse_dollars':float(np.sqrt(d['se_'+n].mean()))} for n in names}
 ordered=d.sort_values('target_gross_amount',ascending=False);tail=[]
 for n in [1,5,10,20,50,100]:tail.append({'largest_actual_loss_cases':n,'share_of_actual_total':float(ordered.target_gross_amount.head(n).sum()/actual.sum()),'actual_amount':float(ordered.target_gross_amount.head(n).sum()),'base_predicted_amount':float(ordered.base_amount.head(n).sum())})
 timing={}
 for date in ['2023-01-01','2023-07-01']:
  good=pd.to_datetime(d.app).ge(date);timing[date]={'excluded_rows':int((~good).sum()),'included_rows':int(good.sum()),'base':money_metrics(actual[good],d.loc[good,'base_amount']),'macro':money_metrics(actual[good],d.loc[good,'macro_vintage_amount']),'direct':money_metrics(actual[good],d.loc[good,'prediction'])}
 sensitivity={str(p):{n:float(deviance(actual,d[n],p).mean()) for n in names} for p in [1.1,1.5,1.9]}
 out={'bootstrap_replicates':1000,'paired_by_borrower':True,'not_a_new_independent_test':True,'comparisons':comparisons,'actual_loss_concentration':tail,'approval_freeze_sensitivity':timing,'tweedie_power_sensitivity':sensitivity}
 (OUT/'uncertainty_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 # Conservative exact linkage only, with ambiguous keys excluded.
 old=pd.read_pickle(ROOT/'data/processed/phase3_development.pkl');new=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl');keys=['borrower_key','app','disb','amount'];old_unique=~old.duplicated(keys,keep=False);new_unique=~new.duplicated(keys,keep=False);fields=['co','pif','status','loss','initial_rate','log_jobs','industry','state','business_age','business_type','processing_method','naics3']
 joined=old.loc[old_unique,keys+fields].merge(new.loc[new_unique,keys+fields],on=keys,how='inner',validate='one_to_one',suffixes=('_june','_sept'));changes={}
 for field in fields:
  x=joined[field+'_june'];y=joined[field+'_sept'];same=x.eq(y)|(x.isna()&y.isna());changes[field]={'changed_rows':int((~same).sum()),'fraction':float((~same).mean())}
 old_y=joined.co_june.between(joined.disb,joined.disb+pd.DateOffset(months=36));new_y=joined.co_sept.between(joined.disb,joined.disb+pd.DateOffset(months=36));old_amt=joined.loss_june.where(old_y,0);new_amt=joined.loss_sept.where(new_y,0)
 revision={'old_rows':len(old),'new_development_rows':len(new),'old_ambiguous_key_rows':int((~old_unique).sum()),'new_ambiguous_key_rows':int((~new_unique).sum()),'unique_exact_matches':len(joined),'match_fraction_of_old':len(joined)/len(old),'linkage_is_not_a_public_loan_id':True,'field_changes':changes,'36_month_label_changed':int((old_y!=new_y).sum()),'36_month_amount_changed':int((old_amt!=new_amt).sum()),'old_total':float(old_amt.sum()),'new_total':float(new_amt.sum()),'unmatched_rows_not_assumed_different_loans':True};(OUT/'snapshot_revision_audit.json').write_text(json.dumps(revision,indent=2)+'\n');print(json.dumps({'comparisons':comparisons,'concentration':tail,'revision':revision},indent=2),flush=True)
if __name__=='__main__':main()
