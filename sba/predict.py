"""Run the trusted frozen research artifact on an explicitly historical example."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, json, hashlib, argparse, math, re
from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from features import transform_raw, RAW, INDUSTRIES, NEW, EXISTING
from origination_model import predict_bundle
from paper_methods import MacroHistory
from public_features import STATE_FIPS

def load_artifacts():
    expected = json.loads((ROOT/'reports/origination_v2/model_freeze.json').read_text())['model_sha256']
    path = ROOT/'models/origination_v2_2024.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Model hash does not match the recorded evaluation')
    # joblib is executable serialization: use only this trusted published artifact.
    return joblib.load(path), MacroHistory(pd.read_csv(ROOT/'reports/hour/macro_lineage.csv'))

def input_frame(data):
    required = set(RAW) | {'ApprovalDate'}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError('Supply exactly the fields shown in examples/input.json')
    for key in ['GrossApproval','SBAGuaranteedApproval','InitialInterestRate','JobsSupported']:
        if type(data[key]) not in (int,float) or not math.isfinite(data[key]):
            raise ValueError(key+' must be a finite JSON number')
    if not 0 < data['GrossApproval'] <= 5_000_000: raise ValueError('Amount outside supported SBA 7(a) scope')
    if not 0 <= data['SBAGuaranteedApproval'] <= data['GrossApproval']: raise ValueError('Invalid guarantee')
    if not 0 < data['InitialInterestRate'] <= 30: raise ValueError('Invalid rate')
    if not 0 <= data['JobsSupported'] <= 100000 or data['JobsSupported'] % 1: raise ValueError('Invalid jobs count')
    for key in ['NaicsCode','ProjectState','BusinessAge','BusinessType','ApprovalDate']:
        if not isinstance(data[key],str): raise ValueError(key+' must be text')
    if not re.fullmatch(r'\d{6}',data['NaicsCode']) or data['NaicsCode'][:2] not in INDUSTRIES: raise ValueError('Invalid NAICS')
    if data['ProjectState'] not in STATE_FIPS: raise ValueError('Unsupported state')
    if data['BusinessAge'] not in NEW|EXISTING: raise ValueError('Use a documented SBA business-age category')
    if data['BusinessType'] not in {'CORPORATION','INDIVIDUAL','PARTNERSHIP'}: raise ValueError('Unsupported legal form')
    if not re.fullmatch(r'2024-\d{2}-\d{2}',data['ApprovalDate']): raise ValueError('This example uses the 2024 Q3 research context')
    date = pd.Timestamp(data['ApprovalDate'])
    if not pd.Timestamp('2024-07-01') <= date <= pd.Timestamp('2024-09-30'): raise ValueError('Only 2024 Q3 historical inputs are supported here')
    frame = transform_raw(pd.DataFrame([data]))
    frame['amount'] = data['GrossApproval']; frame['app'] = date
    return frame

def score(data):
    frame = input_frame(data)
    model, history = load_artifacts()
    p, amount = predict_bundle(model, frame, history)
    return {'context':'2024 Q3 historical research', 'horizon_months_from_disbursement':24,
            'probability_of_recorded_gross_chargeoff':float(p[0]),
            'expected_gross_chargeoff_usd':float(amount[0]),
            'observed_outcome':None, 'approved_for_lending':False,
            'point_in_time_feature_provenance_verified':False,
            'interpretation':'Not bankruptcy probability, realized individual loss, bank net loss, or a current underwriting recommendation.'}

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('input',type=Path)
    args=parser.parse_args()
    print(json.dumps(score(json.loads(args.input.read_text())),indent=2,allow_nan=False))
