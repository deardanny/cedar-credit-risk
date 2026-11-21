"""Paired borrower uncertainty, predeclared research gates, leakage audit."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from paper_methods import VARIANTS,MacroHistory,forecast,sha,save,stamp
OUT=ROOT/'reports/papers'

def per_deviance(y,p):
    y=np.asarray(y,dtype=float)/1000;p=np.maximum(np.asarray(p,dtype=float)/1000,1e-11)
    return 4*(y/np.sqrt(p)+np.sqrt(p)-2*np.sqrt(y))

def uncertainty(frame,replicates=500):
    y=frame.actual.to_numpy();baseline=per_deviance(y,frame.base_amount)
    group=pd.factorize(frame.borrower_key)[0];ng=int(group.max()+1);rng=np.random.default_rng(20261120)
    sums=np.column_stack([np.bincount(group,weights=y),np.bincount(group,weights=baseline)]+
                        [np.bincount(group,weights=frame[v+'_amount']) for v in VARIANTS]+
                        [np.bincount(group,weights=per_deviance(y,frame[v+'_amount'])) for v in VARIANTS])
    ratios={v:[] for v in VARIANTS};gains={v:[] for v in VARIANTS}
    for _ in range(replicates):
        sample=sums[rng.integers(0,ng,ng)].sum(axis=0)
        for j,v in enumerate(VARIANTS):
            if sample[0]>0:ratios[v].append(sample[2+j]/sample[0])
            gains[v].append(1-sample[2+len(VARIANTS)+j]/sample[1])
    return {v:{'ratio_95_ci':np.quantile(ratios[v],[.025,.975]).tolist(),'deviance_improvement':float(1-per_deviance(y,frame[v+'_amount']).mean()/baseline.mean()),'deviance_improvement_95_ci':np.quantile(gains[v],[.025,.975]).tolist(),'replicates':replicates} for v in VARIANTS}

def main():
    results=json.loads((OUT/'results.json').read_text());assert len(results['cohorts'])==4
    f=results['experiment'];assert f['code_sha256']==sha(ROOT/'scripts/paper_methods.py');assert f['plan_sha256']==sha(ROOT/'docs/paper_plan.md')
    for path,digest in f['source_sha256'].items():assert sha(ROOT/path)==digest
    saved=[]
    for year in [2018,2019,2020,2024]:
        r=next(c for c in results['cohorts'] if c['year']==year)
        assert r['training_audit']['models_frozen_utc']<r['scored_utc'];assert sha(ROOT/f'models/paper_{year}.joblib')==r['training_audit']['model_sha256']
        d=pd.read_csv(ROOT/f'data/processed/paper_predictions_{year}.csv',dtype={'row_hash':str,'borrower_key':str})
        assert len(d)==r['n'];assert d.y.sum()==r['events'];assert d.actual.sum()==r['variants']['base'][-1]['amount']['actual_total']
        r['uncertainty']=uncertainty(d)
        if year<2024:saved.append(d)
    pooled=pd.concat(saved,ignore_index=True);pooled_uncertainty=uncertainty(pooled);summary={}
    base_error=np.mean([abs(np.log(c['variants']['base'][-1]['amount']['predicted_actual_ratio'])) for c in results['cohorts'] if c['year']<2024])
    for v in VARIANTS:
        ratios=[c['variants'][v][-1]['amount']['predicted_actual_ratio'] for c in results['cohorts'] if c['year']<2024]
        log_error=float(np.mean(np.abs(np.log(ratios))))
        gates={'all_three_amount_ratios_within_20_percent':bool(all(.8<=r<=1.2 for r in ratios)),
               'lower_mean_absolute_log_ratio':bool(log_error<base_error),
               'positive_pooled_deviance_improvement_95_ci':bool(pooled_uncertainty[v]['deviance_improvement_95_ci'][0]>0)}
        summary[v]={'mean_absolute_log_ratio':log_error,'mean_absolute_relative_error':float(np.mean(np.abs(np.array(ratios)-1))),
                    'pooled_amount':{'actual':float(pooled.actual.sum()),'predicted':float(pooled[v+'_amount'].sum()),'ratio':float(pooled[v+'_amount'].sum()/pooled.actual.sum())},
                    'pooled_uncertainty':pooled_uncertainty[v],'gates':gates,'research_gates_pass':all(gates.values())}
    results['summary']=summary;results['research_candidates_passing']=[v for v in VARIANTS if v!='base' and summary[v]['research_gates_pass']]
    results['approved_for_lending']=False
    # Independently test complete fitted forecasts against poisoned future vintages.
    a=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl');lineage=pd.read_csv(ROOT/'reports/hour/macro_lineage.csv')
    d=a[a.disb.between('2019-07-01','2019-09-30')&a.app.ge('2019-07-01')].head(120)
    history=MacroHistory(lineage);poison=lineage.copy();poison.loc[poison.quarter.gt('2019-07-01'),['value','year_change','year_growth']]=123456.
    altered=MacroHistory(poison);truncated=MacroHistory(lineage[lineage.quarter.le('2019-07-01')]);models=joblib.load(ROOT/'models/paper_2019.joblib');max_error=0.
    for v in ['dynamic_macro','dynamic_macro_iso','dynamic_macro_revert']:
        expected=forecast(models[v],d,history,72)
        for alternative in [altered,truncated]:
            got=forecast(models[v],d,alternative,72)
            for e,g in zip(expected,got):
                for k in ['amount','probability','paid','remaining']:
                    max_error=max(max_error,float(np.max(np.abs(e[k]-g[k]))));np.testing.assert_array_equal(e[k],g[k])
    frozen=[]
    for path in ['reports/phase3/model_freeze_manifest.json','reports/hour/medium_frozen.json']:
        manifest=json.loads((ROOT/path).read_text())
        for file,digest in manifest['files'].items():assert sha(ROOT/file)==digest;frozen.append(file)
    # Baseline replication: not merely a different cohort with a familiar name.
    prior=json.loads((ROOT/'reports/hour/long_robustness.json').read_text())['rows']
    for year in [2018,2019,2020]:
        old=next(c for c in prior if c['test_year']==year);new=next(c for c in results['cohorts'] if c['year']==year)
        assert old['n']==new['n'];np.testing.assert_allclose(old['amount']['predicted_total'],new['variants']['base'][-1]['amount']['predicted_total'],rtol=1e-12)
    validation={'checked_utc':stamp(),'all_checks_passed':True,'future_macro_poison_and_removal_full_forecast_max_error':max_error,'forecast_invariance_cases':120*3*2*6,
                'old_frozen_artifacts_unchanged':len(frozen),'old_six_year_baseline_exactly_reproduced':True,'plan_and_code_hashes_unchanged':True,'model_hashes_match_pre_scoring_freezes':True,
                'unit_tests_log':'reports/papers/unit_tests.log','snapshot_availability_limitations_remain':True,'not_independent_holdout':True}
    save(OUT/'validation.json',validation);results['validation']=validation;save(OUT/'evaluated_results.json',results)
    # Aggregate-only browser payload, no row-level information.
    save(ROOT/'web/dist/paper_results.json',results)
    print(json.dumps({'summary':summary,'candidates':results['research_candidates_passing'],'validation':validation},indent=2),flush=True)

if __name__=='__main__':main()
