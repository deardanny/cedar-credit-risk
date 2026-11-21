"""Export the historical 2020 macro models and synthetic parity fixtures."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from paper_methods import MacroHistory,forecast
from hour_export import native_export
from export_loss import export_pipeline,clean

def write(path,obj):path.write_text(json.dumps(clean(obj),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def main():
    bundles=joblib.load(ROOT/'models/paper_2020.joblib');history=MacroHistory(pd.read_csv(ROOT/'reports/hour/macro_lineage.csv'))
    names=['base','dynamic_macro','dynamic_macro_iso','dynamic_macro_revert']
    fitted={}
    for name in names:
        b=bundles[name];c=b['calibration']
        calibration={'kind':'joint','coef':c['model'].coef_,'intercept':c['model'].intercept_} if c['kind']=='joint' else {'kind':'isotonic','co_x':c['co'].X_thresholds_,'co_y':c['co'].y_thresholds_,'paid_x':c['paid_conditional'].X_thresholds_,'paid_y':c['paid_conditional'].y_thresholds_}
        fitted[name]={'hazard':('base' if name=='base' else 'dynamic'),'calibration':calibration,'scale':b['scale'],'macro_mode':b['macro_mode'],'path_mode':b['path_mode']}
    current=history.table.loc[pd.Timestamp('2020-07-01')];means=history.means.loc[pd.Timestamp('2020-07-01')]
    previous=json.loads((ROOT/'web/dist/hour_long_model.json').read_text())
    model={'hazards':{'base':native_export(bundles['base']['hazard']),'dynamic':native_export(bundles['dynamic_macro']['hazard'])},'variants':fitted,
           'severity':export_pipeline(bundles['base']['severity'],'gamma'),'macro_columns':history.columns,'macro_current':current.to_dict(orient='index'),'macro_mean':means.to_dict(orient='index'),
           'loan_ranges':previous['loan_ranges'],'support_percentiles':previous['support_percentiles'],'approval_vintage':'2020-07-01','virtual_freeze':'2020-07-01','approved_for_lending':False,'target':'gross charge-off amount, not net loss'}
    write(ROOT/'web/dist/paper_models.json',model)
    rng=np.random.default_rng(5432026);fixtures=[];base=model['hazards']['base']
    for i in range(48):
        amount=float(rng.uniform(10000,2000000));f={'amount':amount,'log_amount':float(np.log(amount)),'guarantee_share':float(rng.uniform(.5,.85)),'initial_rate':float(rng.uniform(3,12)),'log_jobs':float(np.log1p(rng.integers(0,100)))}
        for key,levels in zip(base['categorical_features'],base['categories']):f[key]=str(levels[int(rng.integers(len(levels)))])
        if i%11==0:f['industry']='Never seen category'
        if i%13==0:f['initial_rate']=None
        d=pd.DataFrame([{**f,'app':pd.Timestamp('2020-07-01')}]);d['initial_rate']=d.initial_rate.astype(float)
        rows={}
        for name in names:
            rows[name]=[{k:(float(v[0]) if isinstance(v,np.ndarray) else v) for k,v in row.items()} for row in forecast(bundles[name],d,history,72)]
        fixtures.append({'features':f,'curves':rows})
    write(ROOT/'tests/paper_curve_fixtures.json',fixtures)
    print('Exported four research variants and 48 x 4 x 6 synthetic curve checks.')
if __name__=='__main__':main()
