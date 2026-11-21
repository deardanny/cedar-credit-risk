"""Chronological development with label maturity embargo and borrower separation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/sba_mpl')
import sys,json,time,platform,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn,scipy,joblib
from scipy.special import expit,logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss,roc_curve
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from features import NUMERIC,CATEGORICAL

def pipeline(kind,params,drop=()):
    num=[v for v in NUMERIC if v not in drop];cat=[v for v in CATEGORICAL if v not in drop]
    prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)],sparse_threshold=0)
    model=LogisticRegression(max_iter=1000,solver='lbfgs',**params) if kind=='logistic' else HistGradientBoostingClassifier(early_stopping=False,random_state=271828,**params)
    return Pipeline([('prep',prep),('model',model)])

def metrics(y,p):
    y=np.asarray(y);p=np.clip(np.asarray(p),1e-8,1-1e-8);n=len(y)
    result={'n':n,'events':int(y.sum()),'observed_rate':float(y.mean()),'predicted_rate':float(p.mean()),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'roc_auc':None,'pr_auc':None,'ks':None,'top_decile_lift':None,'calibration_intercept':None,'calibration_slope':None}
    if len(np.unique(y))<2:return result
    result['roc_auc']=float(roc_auc_score(y,p));result['pr_auc']=float(average_precision_score(y,p));fpr,tpr,_=roc_curve(y,p);result['ks']=float(max(tpr-fpr))
    if np.ptp(p)<1e-12:return result
    top=np.argsort(-p,kind='stable')[:max(1,int(np.ceil(n/10)))];result['top_decile_lift']=float(y[top].mean()/y.mean())
    cal=LogisticRegression(C=1e6,max_iter=1000).fit(logit(p).reshape(-1,1),y)
    result['calibration_intercept']=float(cal.intercept_[0]);result['calibration_slope']=float(cal.coef_[0,0])
    return result

def calibrate(model,frame):
    raw=model.predict_proba(frame)[:,1]
    cal=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(raw,1e-8,1-1e-8)).reshape(-1,1),frame.y)
    return cal

def predict(model,cal,frame):
    return cal.predict_proba(logit(np.clip(model.predict_proba(frame)[:,1],1e-8,1-1e-8)).reshape(-1,1))[:,1]

def main():
    cfg=json.loads((ROOT/'config.json').read_text());a=pd.read_pickle(ROOT/'data/processed/analytic.pkl');windows=cfg['windows'];audit={}
    def window(name):
        lo,hi=windows[name];return a.loc[a.disb.between(lo,hi)].copy()
    dev=window('development');tune=window('tuning');n=len(tune);tune=tune[~tune.borrower_key.isin(set(dev.borrower_key))];audit['tuning_repeat_borrower_removed']=n-len(tune)
    refit=window('refit');cal=window('calibration');n=len(cal);cal=cal[~cal.borrower_key.isin(set(refit.borrower_key))];audit['calibration_repeat_borrower_removed']=n-len(cal)
    test=window('test');n=len(test);test=test[~test.borrower_key.isin(set(refit.borrower_key)|set(cal.borrower_key))];audit['test_repeat_borrower_removed']=n-len(test)
    # Conservative unknown-label interval ends before the scored cohort starts.
    assert (dev.disb+pd.DateOffset(months=36)).max()<tune.disb.min()
    assert (refit.disb+pd.DateOffset(months=36)).max()<cal.disb.max()+pd.DateOffset(months=36)
    assert (cal.disb+pd.DateOffset(months=36)).max()<pd.Timestamp(cfg['model_freeze'])
    assert test.disb.min()>=pd.Timestamp(cfg['model_freeze'])
    assert not(set(test.borrower_key)&(set(refit.borrower_key)|set(cal.borrower_key)))
    for name,d in [('development',dev),('tuning',tune),('refit',refit),('calibration',cal),('test',test)]:
        audit[name]={'n':len(d),'events':int(d.y.sum()),'rate':float(d.y.mean()),'start':str(d.disb.min().date()),'end':str(d.disb.max().date()),'label_complete_by':str((d.disb+pd.DateOffset(months=36)).max().date())}
        print(name,audit[name],flush=True)
    (ROOT/'reports/split_audit.json').write_text(json.dumps(audit,indent=2))
    experiments=[];choices={}
    configs=[('logistic',{'C':0.1}),('logistic',{'C':1.0}),('boosting',{'max_iter':120,'learning_rate':0.06,'max_leaf_nodes':7,'l2_regularization':10,'min_samples_leaf':100}),('boosting',{'max_iter':160,'learning_rate':0.05,'max_leaf_nodes':15,'l2_regularization':20,'min_samples_leaf':100})]
    for kind,params in configs:
        start=time.time();m=pipeline(kind,params);m.fit(dev,dev.y);score=metrics(tune.y,m.predict_proba(tune)[:,1]);row={'kind':kind,'parameters':params,'metrics':score,'seconds':round(time.time()-start,2)};experiments.append(row)
        if kind not in choices or score['pr_auc']>choices[kind]['metrics']['pr_auc']:choices[kind]=row
        print('TUNING',json.dumps(row),flush=True)
    (ROOT/'reports/tuning.json').write_text(json.dumps(experiments,indent=2))
    # Choose family on tuning data only; final test is used once for acceptance, never for selection.
    base=choices['logistic']['metrics'];chall=choices['boosting']['metrics']
    selected='boosting' if chall['pr_auc']>=base['pr_auc']*1.10 and chall['roc_auc']>=base['roc_auc'] else 'logistic'
    bundles={};preds={};results={};cal_results={}
    for kind,row in choices.items():
        m=pipeline(kind,row['parameters']);m.fit(refit,refit.y);c=calibrate(m,cal);preds[kind]=predict(m,c,test);results[kind]=metrics(test.y,preds[kind]);cal_results[kind]=metrics(cal.y,predict(m,c,cal))
        bundle={'pipeline':m,'calibrator':c,'kind':kind,'features':NUMERIC+CATEGORICAL,'asof':cfg['as_of'],'model_freeze':cfg['model_freeze'],'horizon_months':36}
        bundles[kind]=bundle;joblib.dump(bundle,ROOT/f'models/{kind}.joblib')
        print('TEST',kind,json.dumps(results[kind]),flush=True)
    constant=float(cal.y.mean());results['constant_calibration_rate']=metrics(test.y,np.full(len(test),constant))
    p=preds[selected];r=results[selected]
    # Predeclared research acceptance criteria, not bank policy thresholds.
    gates={'roc_auc_at_least_0_65':r['roc_auc']>=.65,'pr_auc_at_least_1_5_times_prevalence':r['pr_auc']>=r['observed_rate']*1.5,'top_decile_lift_at_least_2':r['top_decile_lift']>=2,'brier_better_than_frozen_constant':r['brier']<results['constant_calibration_rate']['brier'],'mean_prediction_within_25_percent':abs(r['predicted_rate']/r['observed_rate']-1)<=.25}
    # Borrower-cluster bootstrap preserves dependence among repeated borrower rows.
    rng=np.random.default_rng(cfg['seed']);boots=[];y=test.y.to_numpy()
    groups=list(pd.Series(np.arange(len(test))).groupby(test.borrower_key.to_numpy()).apply(list))
    for _ in range(200):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);boots.append([roc_auc_score(y[ix],p[ix]),average_precision_score(y[ix],p[ix]),brier_score_loss(y[ix],p[ix])])
    ci={name:[float(v) for v in np.quantile(np.array(boots)[:,i],[.025,.975])] for i,name in enumerate(['roc_auc','pr_auc','brier'])}
    summary={'selected_before_test':selected,'selection_rule':'Challenger requires >=10% relative tuning AP gain and no ROC-AUC decline; otherwise logistic. Highest tuning AP chooses hyperparameters.','test':results,'calibration_fit_diagnostic_not_holdout':cal_results,'research_gates':gates,'research_gate_pass':all(gates.values()),'bootstrap_95_ci':ci,'bootstrap_unit':'normalized borrower key','bootstrap_replicates':200,'test_used_for_tuning':False,'constant_rate':constant}
    (ROOT/'reports/model_results.json').write_text(json.dumps(summary,indent=2))
    scores=test[['row_hash','borrower_key','disb','y','industry','state','business_age','log_amount']].copy()
    for k,v in preds.items():scores[k]=v
    scores.to_csv(ROOT/'data/processed/test_predictions.csv',index=False)
    calp=predict(bundles[selected]['pipeline'],bundles[selected]['calibrator'],cal)
    bins=np.unique(np.quantile(calp,np.linspace(0,1,11)));bins[0]=-np.inf;bins[-1]=np.inf
    deciles=pd.DataFrame({'y':y,'p':p,'bin':pd.cut(p,bins,labels=False,include_lowest=True)})
    deciles.groupby('bin').agg(n=('y','size'),events=('y','sum'),observed=('y','mean'),predicted=('p','mean')).to_csv(ROOT/'reports/risk_bands.csv')
    segments=[]
    for column,values in [('year',test.disb.dt.year.astype(str)),('industry',test.industry),('state',test.state),('business_age',test.business_age),('amount_band',pd.cut(np.exp(test.log_amount),[0,150000,500000,1000000,np.inf],labels=['<=150k','150k-500k','500k-1m','>1m']))]:
        for value in values.unique():
            ix=np.asarray(values==value)
            if ix.sum():segments.append({'dimension':column,'segment':str(value),**metrics(y[ix],p[ix]),'low_event_count':int(y[ix].sum())<30})
    pd.DataFrame(segments).to_csv(ROOT/'reports/segments.csv',index=False)
    # Sensitivity models have fixed baseline settings and cannot replace the selected champion.
    sensitivity={}
    for name,drop in [('without_rate',['initial_rate']),('without_age',['business_age']),('without_geography',['state'])]:
        m=pipeline('logistic',choices['logistic']['parameters'],drop);m.fit(refit,refit.y);c=calibrate(m,cal);sensitivity[name]=metrics(test.y,predict(m,c,test))
    (ROOT/'reports/sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
    # Export the preselected champion; verify exported inference against sklearn fixtures.
    b=bundles[selected];prep=b['pipeline']['prep'];lm=b['pipeline']['model'];num=prep.named_transformers_['num'];encoder=prep.named_transformers_['cat']
    reference={c:float(refit[c].median()) for c in NUMERIC};reference.update({c:str(refit[c].mode().iloc[0]) for c in CATEGORICAL})
    export={'kind':selected,'role':'research_champion','numeric_features':NUMERIC,'categorical_features':CATEGORICAL,'medians':num['impute'].statistics_.tolist(),'means':num['scale'].mean_.tolist(),'scales':num['scale'].scale_.tolist(),'categories':[v.tolist() for v in encoder.categories_],'calibration_intercept':float(b['calibrator'].intercept_[0]),'calibration_slope':float(b['calibrator'].coef_[0,0]),'feature_names':prep.get_feature_names_out().tolist(),'as_of':cfg['as_of'],'model_freeze':cfg['model_freeze'],'horizon_months':36,'test_metrics':results[selected],'research_gate_pass':all(gates.values()),'approved_for_lending':False,'training_ranges':{c:[float(refit[c].min()),float(refit[c].max())] for c in NUMERIC},'reference_input':reference,'reference_quantiles':np.quantile(calp,np.linspace(0,1,101)).tolist()}
    if selected=='logistic':export.update({'coef':lm.coef_[0].tolist(),'intercept':float(lm.intercept_[0])})
    else:
        trees=[]
        for iteration in lm._predictors:
            tree=[]
            for node in iteration[0].nodes:
                assert not node['is_categorical']
                tree.append({'value':float(node['value']),'feature':int(node['feature_idx']),'threshold':float(node['num_threshold']),'left':int(node['left']),'right':int(node['right']),'leaf':bool(node['is_leaf']),'missing_left':bool(node['missing_go_to_left'])})
            trees.append(tree)
        export.update({'baseline':float(lm._baseline_prediction[0,0]),'trees':trees,'export_contract':'sklearn 1.7.2 numeric TreePredictor; leaf values include learning-rate shrinkage'})
    (ROOT/'web/dist/model.json').write_text(json.dumps(export,indent=2))
    # Reference fixtures verify browser inference and exact numerical parity.
    fixture=test.iloc[np.linspace(0,len(test)-1,40,dtype=int)]
    clean=fixture[NUMERIC+CATEGORICAL].replace({np.nan:None}).to_dict('records')
    fixtures={'cases':clean,'probabilities':predict(b['pipeline'],b['calibrator'],fixture).tolist()}
    (ROOT/'tests/scoring_fixtures.json').write_text(json.dumps(fixtures,indent=2,allow_nan=False))
    stability=[]
    for feature in NUMERIC:
        edges=np.unique(np.quantile(refit[feature].dropna(),np.linspace(0,1,11)));edges[0]=-np.inf;edges[-1]=np.inf
        old=np.histogram(refit[feature].dropna(),edges)[0];new=np.histogram(test[feature].dropna(),edges)[0]
        q=np.maximum(old/old.sum(),1e-6);s=np.maximum(new/new.sum(),1e-6)
        stability.append({'feature':feature,'psi':float(np.sum((s-q)*np.log(s/q))),'train_missing':float(refit[feature].isna().mean()),'test_missing':float(test[feature].isna().mean()),'test_outside_train_range':int((test[feature].lt(refit[feature].min())|test[feature].gt(refit[feature].max())).sum())})
    for feature in CATEGORICAL:
        cats=sorted(set(refit[feature])|set(test[feature]));q=np.maximum(refit[feature].value_counts(normalize=True).reindex(cats,fill_value=0).to_numpy(),1e-6);s=np.maximum(test[feature].value_counts(normalize=True).reindex(cats,fill_value=0).to_numpy(),1e-6)
        stability.append({'feature':feature,'psi':float(np.sum((s-q)*np.log(s/q))),'test_unseen_rows':int((~test[feature].isin(set(refit[feature]))).sum())})
    (ROOT/'reports/stability.json').write_text(json.dumps(stability,indent=2))
    supported=np.ones(len(test),dtype=bool)
    for feature in NUMERIC:supported &= (test[feature].notna() & test[feature].ge(refit[feature].min()) & test[feature].le(refit[feature].max())).to_numpy()
    for feature in CATEGORICAL:supported &= test[feature].isin(set(refit[feature])).to_numpy()
    (ROOT/'reports/demo_coverage.json').write_text(json.dumps({'supported_n':int(supported.sum()),'unsupported_n':int((~supported).sum()),'coverage':float(supported.mean()),'supported_subset_metrics':metrics(y[supported],p[supported]),'note':'Historical test metrics include all analytic test rows; browser input validation abstains outside training ranges or on missing inputs.'},indent=2))
    version={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'scipy':scipy.__version__,'joblib':joblib.__version__}
    (ROOT/'reports/environment.json').write_text(json.dumps(version,indent=2))
    print('FINISHED',selected,gates,flush=True)

if __name__=='__main__':main()
