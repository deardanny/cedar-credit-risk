"""Predeclared retrospective challenger experiment; never overwrites old models."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import sys, json, hashlib, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_tweedie_deviance
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from origination_model import CONFIGS, split_training, targets, fit_bundle, predict_bundle
from paper_methods import MacroHistory, forecast
from train import metrics
from hour_metrics import money_metrics
OUT = ROOT / 'reports/origination_v2'


def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name, value): (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def cohort(source, development, year):
    cutoff = f'{year}-07-01'
    known = set(development.loc[development.disb.lt(cutoff), 'borrower_key'])
    d = source[source.disb.between(cutoff, f'{year}-09-30') & source.app.between(cutoff, f'{year}-09-30') & ~source.borrower_key.isin(known)].copy()
    d = targets(d)
    assert d.outcome_end.max() <= pd.Timestamp('2026-09-30')
    assert not set(d.borrower_key) & known
    return d


def evaluate(d, p, amount):
    result = {'probability': metrics(d.event24, p), 'amount': money_metrics(d.gross24, amount)}
    result['expected_events'] = float(np.sum(p))
    result['observed_events'] = int(d.event24.sum())
    result['predicted_amount_per_expected_event'] = float(amount.sum() / p.sum())
    result['observed_amount_per_event'] = float(d.gross24.sum() / d.event24.sum()) if d.event24.sum() else None
    return result


def paired_bootstrap(d, bp, be, p, e):
    y, actual = d.event24.to_numpy(), d.gross24.to_numpy()
    keys = d.borrower_key.to_numpy()
    groups = list(pd.Series(np.arange(len(d))).groupby(keys).apply(list))
    rng, rows = np.random.default_rng(20261121), []
    for _ in range(500):
        ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if actual[ix].sum() <= 0: continue
        before = mean_tweedie_deviance(actual[ix] / 1000, np.maximum(be[ix], 1e-8) / 1000, power=1.5)
        after = mean_tweedie_deviance(actual[ix] / 1000, np.maximum(e[ix], 1e-8) / 1000, power=1.5)
        rows.append([1 - after / before, e[ix].sum() / actual[ix].sum(), np.mean((bp[ix]-y[ix])**2) - np.mean((p[ix]-y[ix])**2)])
    return {'replicates': len(rows), 'unit': 'borrower cluster', 'conditional_on_fitted_model': True,
            **{name: np.quantile(np.array(rows)[:, j], [.025, .975]).tolist() for j, name in enumerate(['deviance_improvement_95_ci', 'amount_ratio_95_ci', 'brier_improvement_95_ci'])}}


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').exists(): raise SystemExit('Completed experiment exists. Do not overwrite an evaluated selection.')
    start = time.time()
    sources = ['data/processed/hour_short_development.pkl', 'data/processed/hour_medium_holdout.pkl', 'reports/hour/macro_lineage.csv']
    freeze = {'created_utc': stamp(), 'plan_sha256': sha(ROOT / 'docs/origination_v2_plan.md'),
        'code_sha256': {p: sha(ROOT / p) for p in ['src/origination_model.py', 'scripts/origination_experiment.py']},
        'source_sha256': {p: sha(ROOT / p) for p in sources}, 'configs': CONFIGS,
        'selection_years': [2018, 2019, 2020], 'assessment_year': 2024,
        'all_historical_outcomes_previously_examined': True, 'prospective_independent_test': False}
    old_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'models').glob('paper_*.joblib')}
    freeze['preserved_original_models'] = old_hashes
    save('freeze.json', freeze)
    development = pd.read_pickle(ROOT / sources[0])
    history = MacroHistory(pd.read_csv(ROOT / sources[2]))
    rows, audits = [], []
    for year in [2018, 2019, 2020]:
        cut = f'{year}-07-01'
        fit, cal = split_training(development, cut)
        d = cohort(development, development, year)
        baseline = joblib.load(ROOT / f'models/paper_{year}.joblib')['dynamic_macro']
        b = forecast(baseline, d, history, 24)[-1]
        baseline_result = evaluate(d, b['probability'], b['amount'])
        audit = {'year': year, 'fit_loans': len(fit), 'calibration_loans': len(cal), 'fit_events': int(fit.event24.sum()),
            'calibration_events': int(cal.event24.sum()), 'last_label_end': str(max(fit.outcome_end.max(), cal.outcome_end.max())),
            'fit_cal_borrower_overlap': len(set(fit.borrower_key) & set(cal.borrower_key)), 'test_loans': len(d)}
        audits.append(audit); print('SPLIT', json.dumps(audit), flush=True)
        for name in CONFIGS:
            model = fit_bundle(fit, cal, history, cut, name)
            p, amount = predict_bundle(model, d, history)
            result = evaluate(d, p, amount)
            rows.append({'year': year, 'config': name, 'baseline': baseline_result, **result})
            save('selection_progress.json', {'rows': rows, 'audits': audits})
            print('VALIDATION', year, name, 'ratio', round(result['amount']['predicted_actual_ratio'], 3), 'deviance', round(result['amount']['tweedie_deviance'], 4), 'auc', round(result['probability']['roc_auc'], 4), 'seconds', round(time.time()-start), flush=True)
    scores = {name: float(np.mean([np.log(r['amount']['tweedie_deviance'] / r['baseline']['amount']['tweedie_deviance']) for r in rows if r['config'] == name])) for name in CONFIGS}
    selected = min(scores, key=scores.get)
    selection = {'selected': selected, 'selection_scores': scores, 'rule': 'Minimum mean log Tweedie-deviance ratio vs matched old model across 2018/2019/2020, all at 24 months', 'selected_utc': stamp(), 'rows': rows, 'audits': audits}
    save('selection.json', selection)
    print('SELECTED BEFORE 2024 SCORING', selected, scores, flush=True)
    fit, cal = split_training(development, '2024-07-01')
    model = fit_bundle(fit, cal, history, '2024-07-01', selected)
    path = ROOT / 'models/origination_v2_2024.joblib'
    joblib.dump(model, path)
    audit = {'frozen_utc': stamp(), 'model_sha256': sha(path), 'selection_sha256': sha(OUT / 'selection.json'),
        'fit_loans': len(fit), 'calibration_loans': len(cal), 'fit_events': int(fit.event24.sum()),
        'calibration_events': int(cal.event24.sum()), 'calibration_first_disbursement': str(cal.disb.min()),
        'calibration_last_disbursement': str(cal.disb.max()), 'last_label_end': str(max(fit.outcome_end.max(), cal.outcome_end.max())),
        'borrower_overlap': len(set(fit.borrower_key) & set(cal.borrower_key)), 'amount_scale': model['amount_scale']}
    save('model_freeze.json', audit)
    d = cohort(pd.read_pickle(ROOT / sources[1]), development, 2024)
    p, e = predict_bundle(model, d, history)
    b = forecast(joblib.load(ROOT / 'models/paper_2024.joblib')['dynamic_macro'], d, history, 24)[-1]
    bp, be = b['probability'], b['amount']
    old, new = evaluate(d, bp, be), evaluate(d, p, e)
    reference = pd.read_csv(ROOT / 'data/processed/paper_predictions_2024.csv', dtype={'row_hash': str})
    comparison = pd.DataFrame({'row_hash': d.row_hash.astype(str), 'b': be}).merge(reference, on='row_hash', validate='one_to_one')
    assert len(comparison) == len(d)
    np.testing.assert_allclose(comparison.b, comparison.dynamic_macro_amount, rtol=1e-10)
    uncertainty = paired_bootstrap(d, bp, be, p, e)
    segments = []
    for dimension in ['industry', 'business_age', 'state']:
        for value, g in d.groupby(dimension):
            if len(g) < 100: continue
            ix = d.index.get_indexer(g.index)
            segments.append({'dimension': dimension, 'value': str(value), 'loans': len(g), 'events': int(g.event24.sum()),
                'actual': float(g.gross24.sum()), 'baseline': float(be[ix].sum()), 'challenger': float(e[ix].sum()),
                'low_event_count': int(g.event24.sum()) < 30})
    calibration = []
    for ix in np.array_split(np.argsort(p), 10):
        calibration.append({'loans': len(ix), 'events': int(d.event24.iloc[ix].sum()), 'predicted': float(p[ix].mean()), 'observed': float(d.event24.iloc[ix].mean())})
    gates = {'amount_within_25_percent': .75 <= new['amount']['predicted_actual_ratio'] <= 1.25,
        'probability_within_25_percent': .75 <= new['probability']['predicted_rate'] / new['probability']['observed_rate'] <= 1.25,
        'auc_at_least_baseline': new['probability']['roc_auc'] >= old['probability']['roc_auc'],
        'paired_deviance_improvement_95_ci_positive': uncertainty['deviance_improvement_95_ci'][0] > 0}
    result = {'created_utc': stamp(), 'freeze': freeze, 'selection': selection, 'model': audit,
        'assessment': {'year': 2024, 'months': 24, 'loans': len(d), 'events': int(d.event24.sum()),
            'baseline': old, 'challenger': new, 'uncertainty': uncertainty, 'research_gates': gates,
            'all_research_gates_pass': all(gates.values()), 'calibration_deciles': calibration, 'segments': segments},
        'approved_for_lending': False, 'retrospective': True, 'elapsed_seconds': time.time()-start}
    pd.DataFrame({'row_hash': d.row_hash, 'borrower_key': d.borrower_key, 'actual': d.gross24, 'event': d.event24, 'baseline_p': bp, 'baseline_amount': be, 'challenger_p': p, 'challenger_amount': e}).to_csv(OUT / 'predictions.csv', index=False)
    for relative, digest in old_hashes.items(): assert sha(ROOT / relative) == digest
    save('results.json', result)
    print('COMPLETE', json.dumps({'selected': selected, 'baseline': old, 'challenger': new, 'uncertainty': uncertainty, 'gates': gates}, indent=2), flush=True)


if __name__ == '__main__': main()
