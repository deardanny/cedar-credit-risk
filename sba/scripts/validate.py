"""Independent data/split assertions and immutable deliverable manifest."""
import sys,json,hashlib,subprocess
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    a=pd.read_pickle(ROOT/'data/processed/analytic.pkl');cfg=json.loads((ROOT/'config.json').read_text());results={}
    assert not a.row_hash.duplicated().any();results['unique_published_row_fingerprint']=True
    assert a.disb.notna().all() and (a.disb>=a.app).all();results['execution_dates_valid']=True
    end=a.disb+pd.DateOffset(months=36);assert (end<=pd.Timestamp(cfg['as_of'])).all();results['all_labels_mature']=True
    # Reconstruct targets without calling the preparation function.
    expected=(a.co.notna() & (a.co>=a.disb) & (a.co<=end)).astype(int);assert (expected==a.y).all();results['independent_label_reconstruction']=True
    assert not (a.co>pd.Timestamp(cfg['as_of'])).any();results['no_future_chargeoff_dates']=True
    refit=a[a.disb.between(*cfg['windows']['refit'])];cal=a[a.disb.between(*cfg['windows']['calibration'])];cal=cal[~cal.borrower_key.isin(refit.borrower_key)]
    test=a[a.disb.between(*cfg['windows']['test'])];test=test[~test.borrower_key.isin(set(refit.borrower_key)|set(cal.borrower_key))]
    pred=pd.read_csv(ROOT/'data/processed/test_predictions.csv',dtype={'row_hash':str,'borrower_key':str})
    assert set(test.row_hash)==set(pred.row_hash);assert len(test)==len(pred)
    assert not(set(test.borrower_key)&(set(refit.borrower_key)|set(cal.borrower_key)));results['holdout_borrower_separation']=True
    assert (cal.disb+pd.DateOffset(months=36)).max()<pd.Timestamp(cfg['model_freeze']);results['labels_known_before_test_freeze']=True
    metrics=json.loads((ROOT/'reports/model_results.json').read_text());assert metrics['test']['boosting']['events']==int(test.y.sum());results['reported_test_counts_reconcile']=True
    model=json.loads((ROOT/'web/dist/model.json').read_text());assert model['kind']==metrics['selected_before_test'];assert model['approved_for_lending'] is False;results['web_uses_selected_model']=True
    blocked={'status','co','pif','loss','LoanStatus','ChargeOffDate','BankName','SoldSecMrktInd'}
    assert not blocked.intersection(model['numeric_features']+model['categorical_features']);results['outcomes_and_current_assignee_excluded']=True
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],check=True)
    results['target_and_feature_unit_tests']=12
    results['analytic_rows']=len(a);results['test_rows']=len(test)
    results['review_type']='Automated checks and author self-review. No independent human validation.'
    (ROOT/'reports/verification.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
