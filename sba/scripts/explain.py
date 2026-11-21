"""Post-selection diagnostic only: permutation dependence and baseline coefficients."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from train import predict
def main():
 r=json.loads((ROOT/'reports/model_results.json').read_text())
 a=pd.read_pickle(ROOT/'data/processed/analytic.pkl').set_index('row_hash')
 keys=pd.read_csv(ROOT/'data/processed/test_predictions.csv',dtype={'row_hash':str}).row_hash
 test=a.loc[keys].reset_index();bundle=joblib.load(ROOT/f"models/{r['selected_before_test']}.joblib")
 base=predict(bundle['pipeline'],bundle['calibrator'],test);auc=roc_auc_score(test.y,base);ap=average_precision_score(test.y,base)
 rng=np.random.default_rng(271828);rows=[]
 for col in bundle['features']:
  da=[];dp=[]
  for _ in range(3):
   t=test.copy();t[col]=rng.permutation(t[col].to_numpy());p=predict(bundle['pipeline'],bundle['calibrator'],t)
   da.append(auc-roc_auc_score(t.y,p));dp.append(ap-average_precision_score(t.y,p))
  rows.append({'feature':col,'auc_drop_mean':np.mean(da),'auc_drop_shuffle_sd':np.std(da),'ap_drop_mean':np.mean(dp),'repeats':3})
 pd.DataFrame(rows).sort_values('auc_drop_mean',ascending=False).to_csv(ROOT/'reports/global_permutation.csv',index=False)
 b=joblib.load(ROOT/'models/logistic.joblib');pipe=b['pipeline'];coef=pipe['model'].coef_[0]
 pd.DataFrame({'encoded_feature':pipe['prep'].get_feature_names_out(),'unadjusted_logit_coefficient':coef,'calibrated_logit_coefficient':coef*b['calibrator'].coef_[0,0]}).to_csv(ROOT/'reports/baseline_coefficients.csv',index=False)
 print('Global permutation diagnostics and baseline coefficients saved; models unchanged')
if __name__=='__main__':main()
