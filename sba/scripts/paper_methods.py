"""Paper-inspired discrete hazards, as-of macro paths and held-out calibration.

No true default, recovery or balance labels are inferred from gross charge-offs.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys, json, hashlib, time, gc, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from public_features import GROUPS, MACRO, STATE_FIPS
from hour_hazard import calibration_inputs, severity_pipeline
from hour_metrics import money_metrics
from train import metrics
OUT=ROOT/'reports/papers'
BASE_NUM, BASE_CAT=GROUPS['base']
VARIANTS=['base','age_logit_joint','age_logit_iso','static_macro','dynamic_macro','dynamic_macro_iso','dynamic_macro_revert']


def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


class MacroHistory:
    """Quarter-start vintages, with no access to vintages after prediction origin."""
    def __init__(self, lineage):
        self.lineage=lineage.copy()
        q=pd.to_datetime(lineage.quarter)
        assert (pd.to_datetime(lineage.observation_date)<q).all()
        assert not lineage.duplicated(['quarter','series']).any()
        rows=[]
        for quarter,g in lineage.groupby('quarter',sort=True):
            g=g.set_index('series')
            national={'us_unemployment':g.loc['UNRATE','value'],
                      'us_unemployment_change':g.loc['UNRATE','year_change'],
                      'fed_funds':g.loc['FEDFUNDS','value'],
                      'prime_rate':g.loc['MPRIME','value'],
                      'inflation_yoy':g.loc['CPIAUCSL','year_growth'],
                      'industrial_growth_yoy':g.loc['INDPRO','year_growth'],
                      'yield_curve':g.loc['GS10','value']-g.loc['TB3MS','value']}
            for state in [*STATE_FIPS,'Unknown']:
                series=state+'UR'
                v=g.loc[series,'value'] if series in g.index else np.nan
                change=g.loc[series,'year_change'] if series in g.index else np.nan
                rows.append({'vintage':pd.Timestamp(quarter),'state':state,**national,
                             'state_unemployment':v,'state_unemployment_change':change,
                             'state_unemployment_gap':v-national['us_unemployment']})
        self.columns=[c for c in MACRO if c!='rate_over_prime']
        table=pd.DataFrame(rows).sort_values(['state','vintage'])
        self.table=table.set_index(['vintage','state'])[self.columns].sort_index()
        # One-sided rolling reference: only the current and preceding 19 vintages.
        mean=table.copy()
        mean[self.columns]=table.groupby('state',sort=False)[self.columns].transform(lambda v:v.rolling(20,min_periods=1).mean())
        self.means=mean.set_index(['vintage','state'])[self.columns].sort_index()

    def lookup(self, dates, states, mean=False):
        dates=pd.DatetimeIndex(dates)
        vintages=dates.to_period('Q').start_time
        assert (vintages<=dates).all()
        states=pd.Series(np.asarray(states)).where(lambda s:s.isin(STATE_FIPS),'Unknown')
        keys=pd.MultiIndex.from_arrays([vintages,states.to_numpy()],names=['vintage','state'])
        table=self.means if mean else self.table
        ix=table.index.get_indexer(keys)
        if (ix<0).any(): raise ValueError('Missing historical vintage: '+str(keys[ix<0][:3]))
        return table.iloc[ix].to_numpy(dtype=float)

    def attach(self, frame, dates):
        frame=frame.copy()
        frame[self.columns]=self.lookup(dates,frame.state)
        frame['rate_over_prime']=frame.initial_rate-frame.prime_rate
        return frame

    def path(self, frame, quarter, mode='persistence'):
        # Macro origin is APPROVAL date, not disbursement or later observed future.
        current=self.lookup(frame.app,frame.state)
        if mode=='revert':
            long_run=self.lookup(frame.app,frame.state,mean=True)
            current=long_run+(current-long_run)*2**(-quarter/8)
        elif mode!='persistence': raise ValueError(mode)
        x=frame.copy();x[self.columns]=current
        x['rate_over_prime']=x.initial_rate-x.prime_rate
        return x


def risk_intervals(a,cutoff,quarters=24):
    cut=pd.Timestamp(cutoff);d=a[a.disb.lt(cut)].copy()
    d['is_cal']=d.borrower_key.map(lambda k:int(k[:8],16)%5==0)
    cols=BASE_NUM+BASE_CAT+MACRO+['amount','loss','borrower_key','is_cal']
    frames=[]
    for q in range(quarters):
        start=d.disb+pd.DateOffset(months=3*q)
        end=d.disb+pd.DateOffset(months=3*(q+1))
        alive=(d.co.isna()|d.co.gt(start))&(d.pif.isna()|d.pif.gt(start))
        if q==0: alive=(d.co.isna()|d.co.ge(start))&(d.pif.isna()|d.pif.ge(start))
        good=end.lt(cut)&alive
        x=d.loc[good,cols].copy()
        x['age_quarter']=q;x['interval_start']=start[good];x['interval_end']=end[good]
        co=d.co.ge(start)&d.co.le(end);paid=d.pif.ge(start)&d.pif.le(end)
        x['event']=np.where(co[good],1,np.where(paid[good],2,0)).astype('int8')
        x['gross']=x.loss.where(x.event.eq(1),0)
        frames.append(x)
    risk=pd.concat(frames,ignore_index=True)
    assert risk.interval_end.lt(cut).all()
    assert risk.loc[risk.event.eq(1),'gross'].notna().all()
    return risk


def pipeline(kind, macro=False):
    nums=BASE_NUM+(MACRO if macro else [])
    if kind=='logit':
        prep=ColumnTransformer([
            ('num',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),nums),
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=True),BASE_CAT),
            ('age',OneHotEncoder(categories=[np.arange(24)],handle_unknown='error',sparse_output=True),['age_quarter'])],sparse_threshold=1)
        model=LogisticRegression(C=.1,max_iter=600,tol=1e-5,solver='lbfgs')
    else:
        nums=nums+['age_quarter']
        prep=ColumnTransformer([('num','passthrough',nums),('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan),BASE_CAT)],sparse_threshold=0)
        model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,learning_rate=.06,min_samples_leaf=200,l2_regularization=20,early_stopping=False,categorical_features=[False]*len(nums)+[True]*len(BASE_CAT),random_state=1618033)
    return Pipeline([('prep',prep),('model',model)])


def fit_calibration(raw, events, kind):
    if kind=='joint':
        return {'kind':kind,'model':LogisticRegression(C=.1,max_iter=500).fit(calibration_inputs(raw),events)}
    # Chain rule provides valid competing probabilities even for extreme iso outputs.
    co=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(raw[:,1],np.asarray(events)==1)
    eligible=np.asarray(events)!=1
    cp=np.divide(raw[:,2],1-raw[:,1],out=np.zeros(len(raw)),where=(1-raw[:,1])>1e-12)
    paid=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(cp[eligible],np.asarray(events)[eligible]==2)
    return {'kind':kind,'co':co,'paid_conditional':paid}


def calibrated(cal,raw):
    if cal['kind']=='joint': return cal['model'].predict_proba(calibration_inputs(raw))
    co=np.clip(cal['co'].predict(raw[:,1]),0,1)
    cp=np.divide(raw[:,2],1-raw[:,1],out=np.zeros(len(raw)),where=(1-raw[:,1])>1e-12)
    paid=np.clip(cal['paid_conditional'].predict(cp),0,1)
    result=np.column_stack([(1-co)*(1-paid),co,(1-co)*paid])
    assert np.allclose(result.sum(axis=1),1)
    return result


def make_bundle(model,cal,severity,cal_data,raw,cutoff,kind,macro_mode='none',path_mode='persistence'):
    adjust=fit_calibration(raw,cal_data.event,kind)
    p=calibrated(adjust,raw)
    conditional=np.maximum(severity.predict(cal_data),1e-8)*cal_data.amount.to_numpy()
    denom=float((p[:,1]*conditional).sum())
    if denom<=0: raise ValueError('No predicted calibration gross amount')
    scale=float(cal_data.gross.sum()/denom)
    return {'hazard':model,'calibration':adjust,'severity':severity,'scale':scale,'virtual_freeze':cutoff,'macro_mode':macro_mode,'path_mode':path_mode,'horizon_months':72}


def forecast(bundle,d,history,months=72):
    remain=np.ones(len(d));co=np.zeros(len(d));paid=np.zeros(len(d));amount=np.zeros(len(d));curve=[]
    for q in range(months//3):
        x=history.path(d,q,bundle['path_mode']) if bundle['macro_mode']=='dynamic' else d.copy()
        x['age_quarter']=q
        raw=bundle['hazard'].predict_proba(x);p=calibrated(bundle['calibration'],raw)
        severity=np.maximum(bundle['severity'].predict(x),1e-8)*d.amount.to_numpy()*bundle['scale']
        old_co=co.copy();old_amount=amount.copy()
        co+=remain*p[:,1];paid+=remain*p[:,2];amount+=remain*p[:,1]*severity;remain*=p[:,0]
        assert np.allclose(co+paid+remain,1,atol=1e-10)
        assert (co>=old_co-1e-12).all() and (amount>=old_amount-1e-8).all()
        assert np.isfinite(amount).all() and (amount>=0).all()
        if (q+1)%4==0:
            curve.append({'months':(q+1)*3,'probability':co.copy(),'paid':paid.copy(),'remaining':remain.copy(),'amount':amount.copy()})
    return curve


def evaluate(d,curve):
    rows=[]
    for pred in curve:
        y=(d.co.ge(d.disb)&d.co.le(d.disb+pd.DateOffset(months=pred['months']))).astype(int)
        actual=d.loss.where(y.eq(1),0)
        rows.append({'months':pred['months'],'probability':metrics(y,pred['probability']),'amount':money_metrics(actual,pred['amount']),
                     'paid_probability_mean':float(pred['paid'].mean()),'remaining_probability_mean':float(pred['remaining'].mean()),
                     'mass_max_error':float(np.max(np.abs(pred['probability']+pred['paid']+pred['remaining']-1)))})
    return rows


def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    freeze={'created_utc':stamp(),'plan_sha256':sha(ROOT/'docs/paper_plan.md'),'code_sha256':sha(__file__),
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/processed/hour_short_development.pkl',ROOT/'data/processed/hour_medium_holdout.pkl',ROOT/'reports/hour/macro_lineage.csv']},
            'variants':VARIANTS,'not_independent_holdout':True}
    save(OUT/'experiment_freeze.json',freeze)
    a=pd.read_pickle(ROOT/'data/processed/hour_short_development.pkl')
    history=MacroHistory(pd.read_csv(ROOT/'reports/hour/macro_lineage.csv'))
    all_results=[]
    for year in [2018,2019,2020,2024]:
        cut=f'{year}-07-01';months=24 if year==2024 else 72
        risk=risk_intervals(a,cut)
        fit=risk[~risk.is_cal].copy();cal=risk[risk.is_cal&risk.interval_end.ge(pd.Timestamp(cut)-pd.DateOffset(months=12))].copy()
        assert not set(fit.borrower_key)&set(cal.borrower_key)
        audit={'cutoff':cut,'fit_intervals':len(fit),'cal_intervals':len(cal),'fit_borrowers':int(fit.borrower_key.nunique()),'cal_borrowers':int(cal.borrower_key.nunique()),
               'fit_events':{str(k):int(v) for k,v in fit.event.value_counts().items()},'cal_events':{str(k):int(v) for k,v in cal.event.value_counts().items()},
               'last_interval':str(risk.interval_end.max()),'fit_cal_borrower_overlap':0}
        del risk;gc.collect()
        print('DATA',year,audit,flush=True)
        old_name='hour_long.joblib' if year==2020 else f'hour_long_{year}.joblib'
        bundles={}
        if year!=2024:
            old=joblib.load(ROOT/'models'/old_name);severity=old['severity']
            bundles['base']={'hazard':old['hazard'],'calibration':{'kind':'joint','model':old['calibrator']},'severity':severity,'scale':old['scale'],'virtual_freeze':cut,'macro_mode':'none','path_mode':'persistence'}
        else:
            positive=fit[fit.event.eq(1)];severity=severity_pipeline();severity.fit(positive,np.maximum(positive.gross/positive.amount,1e-8))
            base=pipeline('hgb');base.fit(fit,fit.event)
            bundles['base']=make_bundle(base,None,severity,cal,base.predict_proba(cal),cut,'joint')
        # Explicit sampling probabilities: all events + 15% of no-event rows.
        rng=np.random.default_rng(20261120)
        keep=fit.event.ne(0).to_numpy()|(rng.random(len(fit))<.15)
        sample=fit.loc[keep];weights=np.where(sample.event.eq(0),1/.15,1.)
        logit=pipeline('logit')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            logit.fit(sample,sample.event,model__sample_weight=weights)
        audit['logit']={'sample_intervals':len(sample),'no_event_probability':.15,'inverse_probability_weighting':True,'iterations':logit.named_steps['model'].n_iter_.tolist(),'warnings':[str(w.message) for w in caught]}
        if any(issubclass(w.category,ConvergenceWarning) for w in caught): raise RuntimeError('Logistic did not converge; amend numeric solver before evaluation')
        raw=logit.predict_proba(cal)
        for kind,suffix in [('joint','joint'),('isotonic','iso')]:
            bundles['age_logit_'+suffix]=make_bundle(logit,None,severity,cal,raw,cut,kind)
        del sample;gc.collect();print('FIT age logit',year,round(time.time()-start),'s',flush=True)
        static=pipeline('hgb',True);static.fit(fit,fit.event)
        bundles['static_macro']=make_bundle(static,None,severity,cal,static.predict_proba(cal),cut,'joint',macro_mode='static')
        print('FIT static macro',year,round(time.time()-start),'s',flush=True)
        dynamic_fit=history.attach(fit,fit.interval_start);dynamic_cal=history.attach(cal,cal.interval_start)
        # Only previously published quarter-start information enters the risk set.
        assert dynamic_fit.interval_start.dt.to_period('Q').dt.start_time.le(dynamic_fit.interval_start).all()
        audit['dynamic_macro_missing']={c:int(dynamic_fit[c].isna().sum()) for c in MACRO}
        del fit;gc.collect()
        dynamic=pipeline('hgb',True);dynamic.fit(dynamic_fit,dynamic_fit.event);raw=dynamic.predict_proba(dynamic_cal)
        for kind,suffix in [('joint',''),('isotonic','_iso')]:
            bundles['dynamic_macro'+suffix]=make_bundle(dynamic,None,severity,dynamic_cal,raw,cut,kind,macro_mode='dynamic')
        bundles['dynamic_macro_revert']={**bundles['dynamic_macro'],'path_mode':'revert'}
        del dynamic_fit,dynamic_cal,cal;gc.collect();print('FIT dynamic macro',year,round(time.time()-start),'s',flush=True)
        # Save fitted objects and hashes before producing any test predictions.
        model_path=ROOT/f'models/paper_{year}.joblib';joblib.dump(bundles,model_path)
        audit['models_frozen_utc']=stamp();audit['model_sha256']=sha(model_path)
        save(OUT/f'audit_{year}.json',audit)
        source=pd.read_pickle(ROOT/'data/processed/hour_medium_holdout.pkl') if year==2024 else a
        known=set(a.loc[a.disb.lt(cut),'borrower_key'])
        d=source[source.disb.between(cut,f'{year}-09-30')&source.app.ge(cut)&source.app.le(f'{year}-09-30')&~source.borrower_key.isin(known)].copy()
        assert (d.disb+pd.DateOffset(months=months)).max()<=pd.Timestamp('2026-09-30')
        assert not set(d.borrower_key)&known
        y=d.co.ge(d.disb)&d.co.le(d.disb+pd.DateOffset(months=months));actual=d.loss.where(y,0)
        predictions=pd.DataFrame({'row_hash':d.row_hash,'borrower_key':d.borrower_key,'actual':actual,'y':y.astype(int)})
        rows={}
        for name in VARIANTS:
            curve=forecast(bundles[name],d,history,months)
            rows[name]=evaluate(d,curve)
            predictions[name+'_amount']=curve[-1]['amount'];predictions[name+'_probability']=curve[-1]['probability']
            print('EVAL',year,name,'ratio',round(rows[name][-1]['amount']['predicted_actual_ratio'],4),'deviance',round(rows[name][-1]['amount']['tweedie_deviance'],4),flush=True)
        result={'year':year,'cutoff':cut,'months':months,'n':len(d),'events':int(y.sum()),'scored_utc':stamp(),'training_audit':audit,'variants':rows,'retrospective':True,'approved_for_lending':False}
        save(OUT/f'results_{year}.json',result);predictions.to_csv(ROOT/f'data/processed/paper_predictions_{year}.csv',index=False)
        all_results.append(result);save(OUT/'results.json',{'experiment':freeze,'cohorts':all_results,'elapsed_seconds':time.time()-start})
        del bundles,logit,static,dynamic;gc.collect()
    print('COMPLETE',round(time.time()-start),'seconds',flush=True)

if __name__=='__main__': main()
