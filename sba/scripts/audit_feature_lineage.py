"""Independently rebuild macro vintages and test the actual frozen model's data boundary."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,hashlib,json,sys
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import numpy as np
import pandas as pd
import joblib

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();ROOT=args.root;OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from origination_model import features,predict_bundle,split_training,targets,BASE_NUM,BASE_CAT,MACRO
from origination_experiment import cohort
from paper_methods import MacroHistory

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,d):(OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n')
def same(a,b):
    if pd.api.types.is_numeric_dtype(a):return np.isclose(a,b,atol=1e-10,rtol=1e-10,equal_nan=True)
    return a.eq(b)|(a.isna()&b.isna())

def rebuild():
    entries=[]
    for name in ['reports/phase3/public_source_manifest.json','reports/hour/new_public_sources.json']:entries+=json.load(open(ROOT/name))['sources']
    entries=[e for e in entries if '/alfred/' in e['file']]
    assert len(entries)==320 and len({e['file'] for e in entries})==320
    rebuilt=[];hashes={};cache={};manifest_checks=[]
    for e in entries:
        p=ROOT/e['file'];h=sha(p);assert h==e['sha256'];hashes[e['file']]=h
        q=pd.Timestamp(p.name[:10]);stamp=q.strftime('%Y%m%d');params=parse_qs(urlsplit(e['url']).query)
        assert urlsplit(e['url']).hostname=='alfred.stlouisfed.org'
        ids=params['id'][0].split(',');vintages=params['vintage_date'][0].split(',')
        assert len(ids)==len(vintages) and all(v==str(q.date()) for v in vintages)
        raw=pd.read_csv(p,index_col=0,parse_dates=True)
        assert list(raw.columns)==e['expected_vintage_columns']==[i+'_'+stamp for i in ids]
        for col in raw:
            s=pd.to_numeric(raw[col],errors='coerce').dropna().sort_index();series=col.rsplit('_',1)[0]
            assert s.index.max()<q
            t=s.index[-1];v=float(s.loc[t]);prior=s.get(t-pd.DateOffset(months=12),np.nan)
            rebuilt.append({'quarter':str(q.date()),'series':series,'value':v,'year_change':v-prior,'year_growth':(v/prior-1)*100 if pd.notna(prior) and prior else np.nan,'observation_date':str(t.date()),'age_days':(q-t).days})
            cache[(str(q.date()),series)]=s
        manifest_checks.append({'file':e['file'],'sha256':h,'url':e['url'],'vintage_date':str(q.date()),'columns_checked':len(raw.columns)})
    rebuilt=pd.DataFrame(rebuilt).sort_values(['quarter','series']).reset_index(drop=True)
    lineage=pd.read_csv(ROOT/'reports/hour/macro_lineage.csv').sort_values(['quarter','series']).reset_index(drop=True)
    assert len(rebuilt)==len(lineage)==3712
    comparisons={c:int((~np.asarray(same(rebuilt[c],lineage[c]))).sum()) for c in rebuilt}
    assert not any(comparisons.values()),comparisons
    # Same economic observation can have different values in later vintages.
    revisions=[]
    for series in ['UNRATE','CPIAUCSL','INDPRO','CAUR']:
        old=cache[('2018-07-01',series)];later=cache[('2024-07-01',series)]
        aligned=pd.concat([old.rename('old'),later.rename('later')],axis=1).dropna();changed=~np.isclose(aligned.old,aligned.later,rtol=1e-12,atol=1e-12)
        for date,row in aligned[changed].tail(3).iterrows():revisions.append({'series':series,'observation_date':str(date.date()),'2018-07-01_vintage':row.old,'2024-07-01_vintage':row.later})
    save('macro_file_checks.json',manifest_checks);save('macro_vintage_revision_examples.json',revisions)
    return lineage,{'files_verified':len(entries),'vintages':lineage.quarter.nunique(),'series_per_vintage':lineage.groupby('quarter').size().unique().tolist(),'lineage_rows_rebuilt':len(rebuilt),'mismatches':comparisons,'source_sha256_verified':True,'vintage_url_and_header_verified':True,'all_observation_dates_before_vintage':True,'intraday_availability':'Dates are available; exact decision and publication times within a day are not.'}

def main():
    frozen=[ROOT/'models/origination_v2_2024.joblib',ROOT/'src/origination_model.py',ROOT/'scripts/origination_experiment.py',ROOT/'scripts/paper_methods.py',ROOT/'reports/hour/macro_lineage.csv']
    before={str(p.relative_to(ROOT)):sha(p) for p in frozen}
    lineage,macro=rebuild();history=MacroHistory(lineage)
    dev=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl');hold=pd.read_pickle(ROOT/'data/processed/hour_medium_holdout.pkl');data=pd.concat([dev,hold],ignore_index=True)
    b=joblib.load(frozen[0]);columns={}
    for kind in ['probability','severity']:
        columns[kind]=[c for name,t,cols in b[kind].named_steps['prep'].transformers_ if name!='remainder' for c in cols]
    assert columns['probability']==columns['severity'] and len(columns['probability'])==21
    x=features(data,history);vintage=data.app.dt.to_period('Q').dt.start_time
    macro['loan_rows_checked']=len(data);macro['vintage_after_approval_count']=int(vintage.gt(data.app).sum());assert macro['vintage_after_approval_count']==0
    macro['same_day_approval_and_vintage_rows']=int(vintage.eq(data.app).sum())
    # Reconstruct the joins with an independent merge, not MacroHistory.lookup.
    wide=lineage.pivot(index='quarter',columns='series',values=['value','year_change','year_growth'])
    q=vintage.dt.strftime('%Y-%m-%d').to_numpy();reference=pd.DataFrame(index=data.index)
    for name,series,kind in [('us_unemployment','UNRATE','value'),('us_unemployment_change','UNRATE','year_change'),('fed_funds','FEDFUNDS','value'),('prime_rate','MPRIME','value'),('inflation_yoy','CPIAUCSL','year_growth'),('industrial_growth_yoy','INDPRO','year_growth')]:reference[name]=wide[(kind,series)].reindex(q).to_numpy()
    reference['yield_curve']=(wide[('value','GS10')]-wide[('value','TB3MS')]).reindex(q).to_numpy()
    regional=lineage[lineage.series.str.endswith('UR')].copy();regional['state']=regional.series.str[:-2]
    region=pd.DataFrame({'quarter':q,'state':data.state}).merge(regional[['quarter','state','value','year_change']],on=['quarter','state'],how='left',validate='many_to_one')
    reference['state_unemployment']=region.value.to_numpy();reference['state_unemployment_change']=region.year_change.to_numpy();reference['state_unemployment_gap']=reference.state_unemployment-reference.us_unemployment;reference['rate_over_prime']=data.initial_rate-reference.prime_rate
    macro['independent_feature_join_mismatches']={c:int((~same(x[c],reference[c])).sum()) for c in MACRO};assert not any(macro['independent_feature_join_mismatches'].values())
    # Poison every outcome and stale precomputed macro field; then remove all such columns.
    sample=data.groupby(data.app.dt.year,group_keys=False).head(30).copy();expected=predict_bundle(b,sample,history)
    poisoned=sample.copy()
    for c in ['loss','y','target_gross_amount','event24','gross24','TermInMonths','term_months','age_quarter']+MACRO:poisoned[c]=999999999
    for c in ['co','pif','outcome_end']:poisoned[c]=pd.Timestamp('1900-01-01')
    poisoned['status']='CHGOFF'
    controls={}
    for label,frame in [('poisoned_outcomes_and_precomputed_macro',poisoned),('only_explicit_inputs',sample[BASE_NUM+BASE_CAT+['amount','app']])]:
        actual=predict_bundle(b,frame,history)
        controls[label]={'rows':len(frame),'max_probability_change':float(np.max(np.abs(expected[0]-actual[0]))),'max_amount_change':float(np.max(np.abs(expected[1]-actual[1])))}
        for a,c in zip(expected,actual):np.testing.assert_array_equal(a,c)
    future_checks=[]
    for year in sorted(sample.app.dt.year.unique()):
        frame=sample[sample.app.dt.year.eq(year)].copy();last=frame.app.max();p=lineage.copy();p.loc[pd.to_datetime(p.quarter).gt(last),['value','year_change','year_growth']]=999999999
        expected=predict_bundle(b,frame,history)
        for label,h in [('poison_future',MacroHistory(p)),('remove_future',MacroHistory(lineage[pd.to_datetime(lineage.quarter).le(last)]))]:
            actual=predict_bundle(b,frame,h)
            for a,c in zip(expected,actual):np.testing.assert_array_equal(a,c)
            future_checks.append({'year':int(year),'test':label,'rows':len(frame),'probability_and_amount_unchanged':True})
    split_rows=[]
    for year in [2018,2019,2020,2024]:
        fit,cal=split_training(dev,f'{year}-07-01');test=cohort(hold if year==2024 else dev,dev,year)
        out={'year':year,'fit':len(fit),'calibration':len(cal),'test':len(test),'latest_training_outcome_end':str(max(fit.outcome_end.max(),cal.outcome_end.max())),'fit_cal_borrower_overlap':len(set(fit.borrower_key)&set(cal.borrower_key)),'fit_test_borrower_overlap':len(set(fit.borrower_key)&set(test.borrower_key)),'cal_test_borrower_overlap':len(set(cal.borrower_key)&set(test.borrower_key)),'target_reporting_asof_proven':False}
        assert max(fit.outcome_end.max(),cal.outcome_end.max())<pd.Timestamp(f'{year}-07-01')
        assert out['fit_cal_borrower_overlap']==out['fit_test_borrower_overlap']==out['cal_test_borrower_overlap']==0
        split_rows.append(out)
    # Reproduce all final heldout predictions, not just a handful.
    saved=pd.read_csv(ROOT/'reports/origination_v2/predictions.csv',dtype={'row_hash':str});held=hold.merge(saved[['row_hash','challenger_p','challenger_amount']],on='row_hash',validate='one_to_one');actual=predict_bundle(b,held,history)
    np.testing.assert_allclose(actual[0],held.challenger_p,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(actual[1],held.challenger_amount,rtol=1e-10,atol=1e-8)
    for p in frozen:assert before[str(p.relative_to(ROOT))]==sha(p)
    result={'actual_model_columns':columns,'direct_outcome_columns_used':False,'macro':macro,'negative_controls':controls,'future_macro_controls':future_checks,'time_split_and_borrower_controls':split_rows,'all_final_predictions_reproduced':len(held),'frozen_files_unchanged':before,'limits':['Two 2026 loan snapshots do not establish original approval-time values.','Training targets can have delayed reporting or later revisions even when the event date is before cutoff.','Borrower grouping uses normalized name/address, not immutable legal entity ID.','The evaluation data have been examined in prior development; this is not an untouched prospective test.','Cedar inputs use a selected historical origin date (2024-07-01 by default), not live 2026 economic conditions.','Approval-date final amount/rate/guarantee are conditional loan terms; they must be known proposed terms for preapproval use.']}
    baseline_columns={}
    for year in [2018,2019,2020,2024]:
        baseline=joblib.load(ROOT/f'models/paper_{year}.joblib')['dynamic_macro']
        baseline_columns[str(year)]={kind:[(name,list(cols)) for name,_,cols in baseline[kind].named_steps['prep'].transformers_ if name!='remainder'] for kind in ['hazard','severity']}
    save('cedar_baseline_model_columns.json',baseline_columns)
    save('lineage_and_runtime_audit.json',result)
    print(json.dumps({'macro':macro,'negative_controls':controls,'predictions_reproduced':len(held),'all_frozen_files_unchanged':True},indent=2),flush=True)

if __name__=='__main__':main()
