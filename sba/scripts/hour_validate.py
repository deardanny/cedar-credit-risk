"""Validate reserved-horizon timing, frozen artifacts and source lineage."""
import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from train_loss import amounts
OUT=ROOT/'reports/hour'
def main():
 report=json.loads((OUT/'medium_results.json').read_text());frozen=json.loads((OUT/'medium_frozen.json').read_text());selection=json.loads((OUT/'medium_selection.json').read_text());checks={}
 assert frozen['reserved_targets_not_opened'];assert not (ROOT/'data/processed/hour_short_holdout.pkl').exists();checks['twelve_month_target_not_silently_reused']=True
 assert selection['horizon_months']==24 and report['horizon_months']==24
 for name,sha in frozen['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
 checks['medium_model_code_and_selection_unchanged']=True
 testfile=ROOT/'data/processed/hour_medium_holdout.pkl';assert pd.Timestamp(frozen['frozen_at_utc'])<pd.Timestamp(datetime.fromtimestamp(testfile.stat().st_mtime,timezone.utc));checks['selection_frozen_before_reserved_target_extraction']=True
 a=pd.read_pickle(ROOT/'data/processed/hour_medium_development.pkl');d=pd.read_pickle(testfile);fit=a[a.disb.between('2010-01-01','2021-06-30')];cal=a[a.disb.between('2021-07-01','2022-06-30')&~a.borrower_key.isin(set(fit.borrower_key))];known=set(fit.borrower_key)|set(cal.borrower_key);d=d[~d.borrower_key.isin(known)].copy()
 assert len(d)==report['cohort']['n'];assert not set(d.borrower_key)&known;assert not set(fit.borrower_key)&set(cal.borrower_key);assert (cal.disb+pd.DateOffset(months=24)).max()<pd.Timestamp('2024-07-01');assert d.app.ge('2024-07-01').all();assert (d.disb+pd.DateOffset(months=24)).max()<=pd.Timestamp('2026-09-30');checks['approval_time_and_borrower_separation']=True
 for frame in [a,d]:
  reconstructed=(frame.co.ge(frame.disb)&frame.co.le(frame.disb+pd.DateOffset(months=24))).astype(int);assert np.array_equal(frame.y,reconstructed);assert np.array_equal(frame.target_gross_amount,frame.loss.where(reconstructed.eq(1),0));assert (pd.to_datetime(frame.quarter)<=frame.app).all()
 checks['twenty_four_month_targets_independently_recomputed']=True
 for group in ['base','loan_extended','macro_vintage']:
  b=joblib.load(ROOT/f'models/hour_medium_{group}.joblib');assert b['horizon_months']==24;p,c,e=amounts(b,d);assert np.allclose(p*c,e);assert np.isclose(e.sum(),report['results'][group]['amount']['predicted_total'])
 checks['all_medium_predictions_recomputed']=True
 sources=json.loads((OUT/'new_public_sources.json').read_text());assert not sources['failures']
 for source in sources['sources']:assert hashlib.sha256((ROOT/source['file']).read_bytes()).hexdigest()==source['sha256']
 lineage=pd.read_csv(OUT/'macro_lineage.csv');assert (pd.to_datetime(lineage.observation_date)<pd.to_datetime(lineage.quarter)).all();checks['additional_vintage_files_verified']=len(sources['sources'])
 phase3=json.loads((ROOT/'reports/phase3/model_freeze_manifest.json').read_text())
 for name,sha in phase3['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
 checks['previous_phase3_models_and_learning_code_preserved']=True
 hazard=json.loads((OUT/'hazard_results.json').read_text());assert hazard['competing_probability_mass_max_error']<1e-10;checks['competing_event_mass_conserved']=True
 (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
