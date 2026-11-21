import unittest, sys, json, copy
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from predict import input_frame, score, load_artifacts
from origination_model import targets, split_training, predict_bundle
from paper_methods import MacroHistory

class PublishedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example=json.loads((ROOT/'examples/input.json').read_text())
        cls.model,cls.history=load_artifacts()

    def test_fictional_inputs_reproduce_original_research_pipeline(self):
        fixtures=json.loads((ROOT/'examples/reference_predictions.json').read_text())
        for fixture in fixtures:
            result=score(fixture['input'])
            np.testing.assert_allclose([result['probability_of_recorded_gross_chargeoff'],result['expected_gross_chargeoff_usd']],fixture['original_prediction'],rtol=1e-10)
            self.assertFalse(result['approved_for_lending'])
            self.assertIsNone(result['observed_outcome'])

    def test_outcome_and_stale_macro_columns_cannot_change_inference(self):
        frame=input_frame(self.example)
        original=predict_bundle(self.model,frame,self.history)
        frame['loss']=99999999; frame['co']=pd.Timestamp('2024-07-01')
        frame['y']=1; frame['us_unemployment']=9999
        for a,b in zip(original,predict_bundle(self.model,frame,self.history)):
            np.testing.assert_array_equal(a,b)

    def test_no_future_vintage_dependency(self):
        frame=input_frame(self.example)
        lineage=pd.read_csv(ROOT/'reports/hour/macro_lineage.csv')
        truncated=MacroHistory(lineage[lineage.quarter.le('2024-07-01')])
        for a,b in zip(predict_bundle(self.model,frame,self.history),predict_bundle(self.model,frame,truncated)):
            np.testing.assert_array_equal(a,b)

    def test_rejects_current_dates_unexpected_outcomes_and_invalid_numbers(self):
        invalid=[{'ApprovalDate':'2026-11-21'}, {'GrossApproval':True}, {'GrossApproval':'250000'},
                 {'GrossApproval':float('nan')}, {'JobsSupported':-1}, {'InitialInterestRate':0},
                 {'SBAGuaranteedApproval':300000}, {'ProjectState':'XX'}, {'BusinessType':'UNKNOWN'},
                 {'loss':100}, {'ApprovalDate':'2024-02-30'}]
        for change in invalid:
            with self.subTest(change=change),self.assertRaises(ValueError): input_frame(self.example|change)

    def test_target_includes_exact_horizon_and_excludes_later_event(self):
        frame=pd.DataFrame({'disb':pd.to_datetime(['2020-07-01']*3),'co':pd.to_datetime(['2022-07-01','2022-07-02',None]),'loss':[100,200,0]})
        result=targets(frame)
        self.assertEqual(result.event24.tolist(),[1,0,0]); self.assertEqual(result.gross24.tolist(),[100,0,0])

    def test_training_requires_completed_outcomes_and_separate_borrowers(self):
        frame=pd.DataFrame({'disb':pd.to_datetime(['2021-01-01','2021-01-01','2022-07-01','2023-01-01']),
                            'co':pd.to_datetime([None]*4),'loss':[0]*4,'borrower_key':['00000001','00000000','00000002','00000003']})
        fit,cal=split_training(frame,'2024-07-01')
        self.assertEqual(fit.borrower_key.tolist(),['00000001']); self.assertEqual(cal.borrower_key.tolist(),['00000000'])

    def test_frozen_research_code_and_amount_artifact_match_record(self):
        import hashlib
        freeze=json.loads((ROOT/'reports/origination_v2/freeze.json').read_text())
        for name,digest in freeze['code_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)
        self.assertFalse(self.model['approved_for_lending'])

if __name__=='__main__': unittest.main()
