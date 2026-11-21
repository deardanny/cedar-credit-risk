"""Measure snapshot sensitivity, independent source checks and date/identity limits."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();ROOT=a.root;OUT=a.output
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from features import RAW
from origination_model import features,predict_bundle,BASE_NUM,BASE_CAT,MACRO
from paper_methods import MacroHistory
FIELDS=BASE_NUM+BASE_CAT+['log_amount_per_job','log_unguaranteed_amount']

def save(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n')
def different(x,y):
    if pd.api.types.is_numeric_dtype(x):return ~pd.Series(np.isclose(x,y,equal_nan=True,rtol=1e-12,atol=1e-12),index=x.index)
    return ~(x.eq(y)|(x.isna()&y.isna()))
def prep(d,route):
    d=d.copy();d['linkage']=route
    for name in ['app','disb','amount']:
        if name in d:
            d[name+'_june']=d[name];d[name+'_sept']=d[name]
    if 'BorrName' in d:d['BorrName_sept']=d.BorrName
    return d

def main():
    primary=prep(pd.read_pickle(OUT/'project_matches.pkl'),'date_keys_no_amount')
    dates=prep(pd.read_pickle(OUT/'date_project_matches.pkl'),'amount_key_no_dates')
    moved=prep(pd.read_pickle(OUT/'relocation_matches.pkl'),'exact_name_dates_bank_amount_no_address')
    assert moved.same_bank_and_amount.all()
    keys=['source_file_june','source_row_june','source_file_sept','source_row_sept'];all_pairs=pd.concat([primary[keys],dates[keys],moved[keys]],ignore_index=True).drop_duplicates()
    conflict=all_pairs.duplicated(keys[:2],keep=False)|all_pairs.duplicated(keys[2:],keep=False)
    conflicts=all_pairs[conflict];save('linkage_conflicts.json',conflicts.to_dict('records'))
    unique_pairs=all_pairs[~conflict]
    pairs=pd.concat([primary,dates,moved],ignore_index=True).drop_duplicates(keys).merge(unique_pairs,on=keys,how='inner',validate='one_to_one')
    raw_summary=[]
    for f in RAW:
        changed_raw=different(pairs[f+'_june'],pairs[f+'_sept'])
        raw_summary.append({'raw_field':f,'union_comparable_rows':len(pairs),'changed_rows':int(changed_raw.sum()),'fit_changed':int((changed_raw & pairs.role_sept.eq('fit')).sum()),'assessment_changed':int((changed_raw & pairs.role_sept.eq('assessment')).sum())})
    pd.DataFrame(raw_summary).to_csv(OUT/'all_raw_feature_audit.csv',index=False)
    model=joblib.load(ROOT/'models/origination_v2_2024.joblib');history=MacroHistory(pd.read_csv(ROOT/'reports/hour/macro_lineage.csv'))
    frames={}
    for side in ['june','sept']:
        cols=BASE_NUM+BASE_CAT+['amount','app'];frames[side]=pairs[[c+'_'+side for c in cols]].rename(columns={c+'_'+side:c for c in cols})
    xx={side:features(f,history) for side,f in frames.items()};selected=[c for name,_,cols in model['probability'].named_steps['prep'].transformers_ if name!='remainder' for c in cols]
    assert len(selected)==21
    flags=pd.DataFrame({c:different(xx['june'][c],xx['sept'][c]) for c in selected});changed=flags.any(axis=1)
    summary=[]
    for field in selected:
        summary.append({'feature':field,'union_comparable_rows':len(pairs),'changed_rows':int(flags[field].sum()),'fit_changed':int((flags[field]&pairs.role_sept.eq('fit')).sum()),'calibration_changed':int((flags[field]&pairs.role_sept.eq('calibration')).sum()),'assessment_changed':int((flags[field]&pairs.role_sept.eq('assessment')).sum()),'interpretation':'Union of conservative linkage passes, with audited source fields omitted from at least one pass; nonchanges do not prove immutability.'})
    pd.DataFrame(summary).to_csv(OUT/'all_21_feature_audit.csv',index=False)
    print('all 21 feature changes',pd.DataFrame(summary)[['feature','changed_rows','fit_changed','assessment_changed']].to_json(orient='records'),flush=True)
    before=predict_bundle(model,frames['june'][changed],history);after=predict_bundle(model,frames['sept'][changed],history)
    impact=pairs.loc[changed,['BorrName_sept','role_sept','linkage','app_sept','amount_june','amount_sept','status_june','status_sept']+keys].copy()
    impact['changed_features']=flags[changed].apply(lambda r:','.join(r.index[r]),axis=1)
    impact['p_june_inputs']=before[0];impact['p_sept_inputs']=after[0];impact['probability_change_percentage_points']=(after[0]-before[0])*100
    impact['el_june_inputs']=before[1];impact['el_sept_inputs']=after[1];impact['expected_gross_amount_change']=after[1]-before[1]
    impact.to_csv(OUT/'prediction_snapshot_sensitivity.csv',index=False)
    # Verify canonical feature extraction from the source rows used by the frozen model.
    canonical=pd.concat([pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl'),pd.read_pickle(ROOT/'data/processed/hour_medium_holdout.pkl')],ignore_index=True)
    canon=canonical[['source_file','source_row']+BASE_NUM+BASE_CAT+['amount','app']].rename(columns={c:'canonical_'+c for c in BASE_NUM+BASE_CAT+['amount','app']})
    probe=pairs.merge(canon,left_on=['source_file_sept','source_row_sept'],right_on=['source_file','source_row'],validate='one_to_one')
    mismatch={c:int(different(probe['canonical_'+c],probe[c+'_sept']).sum()) for c in BASE_NUM+BASE_CAT+['amount','app']};assert not any(mismatch.values()),mismatch
    # Raw source spot checks cover every loan with an observed used-feature change.
    wanted=pairs.loc[changed].copy();raw_records={}
    with zipfile.ZipFile(ROOT/'data/raw/Archive.zip') as z:
        for side in ['june','sept']:
            for name,g in wanted.groupby('source_file_'+side):
                needed=set(g['source_row_'+side]);source=z.open(name) if side=='june' else open(ROOT/'data/external/sba_new'/name,'rb')
                with source,io.TextIOWrapper(source,encoding='utf-8-sig') as f:
                    for rownum,row in enumerate(csv.DictReader(f),2):
                        if rownum in needed:raw_records[(side,name,rownum)]=row
                        if rownum>=max(needed):break
    for _,r in wanted.iterrows():
        for side in ['june','sept']:
            raw=raw_records[(side,r['source_file_'+side],r['source_row_'+side])]
            assert raw['BorrName']==r.BorrName_sept or raw['BorrName'].strip().upper()==r.BorrName_sept.strip().upper()
            for f in RAW:
                observed=r[f+'_'+side]
                if f in ['GrossApproval','SBAGuaranteedApproval','InitialInterestRate','JobsSupported']:
                    value=pd.to_numeric(raw[f],errors='coerce');assert (pd.isna(value) and pd.isna(observed)) or np.isclose(value,observed,rtol=1e-12,atol=1e-12)
                else:assert raw[f]==observed
    # Disbursement dates affect target clocks, even though they are not X columns.
    d=dates.merge(unique_pairs,on=keys,how='inner',validate='one_to_one');dc=different(d.disb_june,d.disb_sept);co=d.co_sept
    old_y=co.between(d.disb_june,d.disb_june+pd.DateOffset(months=24));new_y=co.between(d.disb_sept,d.disb_sept+pd.DateOffset(months=24))
    endold=d.disb_june+pd.DateOffset(months=24);endnew=d.disb_sept+pd.DateOffset(months=24)
    date_changes=d.loc[dc,['BorrName_sept','role_sept','amount','app_june','app_sept','disb_june','disb_sept','co_june','co_sept']+keys].copy();date_changes.to_csv(OUT/'disbursement_date_changes.csv',index=False)
    date_summary={'date_independent_project_matches':len(d),'approval_date_changed':int(different(d.app_june,d.app_sept).sum()),'disbursement_date_changed':int(dc.sum()),'changed_by_role':d.loc[dc,'role_sept'].value_counts().to_dict(),'24m_label_flips_from_disbursement_only_holding_sept_chargeoff_fixed':int((old_y!=new_y).sum()),'completed_before_final_training_cutoff_membership_flips':int((endold.lt('2024-07-01')!=endnew.lt('2024-07-01')).sum()),'limits':'Not an original-date guarantee; unchanged labels in this interval do not validate past reporting availability.'}
    save('date_dependency_audit.json',date_summary)
    # Observed feature changes split by final source status, without treating EXEMPT as good.
    transition=[]
    for (a,b),g in pairs.groupby(['status_june','status_sept']):transition.append({'june_status':a,'sept_status':b,'n':len(g),'any_model_feature_changed':int(changed.loc[g.index].sum())})
    result={'union_pair_count':len(pairs),'conflicting_pairs_excluded':len(conflicts),'any_model_feature_changed':int(changed.sum()),'feature_changed_loans_by_role':pairs.loc[changed,'role_sept'].value_counts().to_dict(),'max_absolute_probability_difference_percentage_points':float(impact.probability_change_percentage_points.abs().max()),'max_absolute_expected_gross_amount_difference':float(impact.expected_gross_amount_change.abs().max()),'max_probability_change_case':impact.loc[impact.probability_change_percentage_points.abs().idxmax()].to_dict(),'max_amount_change_case':impact.loc[impact.expected_gross_amount_change.abs().idxmax()].to_dict(),'independent_raw_source_records_verified':len(raw_records),'every_used_feature_change_reverified':True,'canonical_source_extraction_mismatches':mismatch,'status_transition_counts':transition,'interpretation':'Sensitivity of the existing frozen model, NOT an accuracy gain, NOT a retrained or leakage-free model. Neither June nor September proves original contract-time values.'}
    save('snapshot_prediction_impact.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['max_probability_change_case','max_amount_change_case','status_transition_counts','canonical_source_extraction_mismatches']},indent=2),flush=True)

if __name__=='__main__':main()
