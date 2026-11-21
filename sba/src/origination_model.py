"""Two-year origination forecasts with matched-horizon borrower calibration."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

BASE_NUM = ['log_amount', 'guarantee_share', 'initial_rate', 'log_jobs']
BASE_CAT = ['industry', 'state', 'business_age', 'business_type']
EXTRA = ['log_amount_per_job', 'log_unguaranteed_amount']
MACRO = ['us_unemployment', 'us_unemployment_change', 'fed_funds', 'prime_rate', 'inflation_yoy', 'industrial_growth_yoy', 'yield_curve', 'state_unemployment', 'state_unemployment_change', 'state_unemployment_gap', 'rate_over_prime']
CONFIGS = {
    'direct_base': {'macro': False, 'recent': False, 'amount_method': 'hurdle'},
    'direct_macro': {'macro': True, 'recent': False, 'amount_method': 'hurdle'},
    'recent_macro': {'macro': True, 'recent': True, 'amount_method': 'hurdle'},
    'recent_poisson': {'macro': True, 'recent': True, 'amount_method': 'poisson'},
}


def targets(data, months=24):
    frame = data.copy()
    frame['outcome_end'] = frame.disb + pd.DateOffset(months=months)
    frame['event24'] = (frame.co.ge(frame.disb) & frame.co.le(frame.outcome_end)).astype(int)
    frame['gross24'] = frame.loss.where(frame.event24.eq(1), 0.)
    return frame


def split_training(data, cutoff):
    cut = pd.Timestamp(cutoff)
    observed = targets(data)
    observed = observed[observed.outcome_end.lt(cut)].copy()
    cal_group = observed.borrower_key.map(lambda key: int(str(key)[:8], 16) % 5 == 0)
    fit = observed[~cal_group].copy()
    # The most recent two years of *fully completed origination cohorts*.
    cal = observed[cal_group & observed.outcome_end.ge(cut - pd.DateOffset(years=2))].copy()
    assert not set(fit.borrower_key) & set(cal.borrower_key)
    assert fit.outcome_end.max() < cut and cal.outcome_end.max() < cut
    return fit, cal


def features(frame, history):
    # Rebuild public inputs from the approval-date vintage, not existing columns
    # which could have been attached at another date or contain outcome fields.
    x = frame[BASE_NUM + BASE_CAT + ['amount', 'app']].copy()
    x['log_amount_per_job'] = x.log_amount - x.log_jobs
    x['log_unguaranteed_amount'] = np.log1p(x.amount * (1 - x.guarantee_share))
    return history.attach(x, x.app)


def pipeline(kind, macro):
    nums = BASE_NUM + (EXTRA + MACRO if macro else [])
    prep = ColumnTransformer([('num', 'passthrough', nums), ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan), BASE_CAT)], sparse_threshold=0)
    params = dict(max_iter=160, max_leaf_nodes=15, learning_rate=.06, min_samples_leaf=150,
                  l2_regularization=15, max_bins=128, early_stopping=False,
                  categorical_features=[False] * len(nums) + [True] * len(BASE_CAT), random_state=20261120)
    if kind == 'probability':
        model = HistGradientBoostingClassifier(**params)
    else:
        params.update(max_iter=120, max_leaf_nodes=7, min_samples_leaf=50)
        model = HistGradientBoostingRegressor(loss='gamma' if kind == 'severity' else 'poisson', **params)
    return Pipeline([('prep', prep), ('model', model)])


def logodds(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_bundle(fit, cal, history, cutoff, config):
    cfg = CONFIGS[config]
    x, cx = features(fit, history), features(cal, history)
    weights = np.ones(len(fit))
    if cfg['recent']:
        age = (pd.Timestamp(cutoff) - fit.outcome_end).dt.days.to_numpy() / 365.25
        weights = np.power(.5, age / 4.)
    probability = pipeline('probability', cfg['macro'])
    probability.fit(x, fit.event24, model__sample_weight=weights)
    raw = probability.predict_proba(cx)[:, 1]
    calibrator = LogisticRegression(C=10, max_iter=1000).fit(logodds(raw), cal.event24)
    cp = calibrator.predict_proba(logodds(raw))[:, 1]
    if cfg['amount_method'] == 'hurdle':
        positive = fit.event24.eq(1).to_numpy()
        severity = pipeline('severity', cfg['macro'])
        severity.fit(x.loc[positive], np.maximum(fit.loc[positive, 'gross24'] / fit.loc[positive, 'amount'], 1e-8), model__sample_weight=weights[positive])
        unscaled = cp * severity.predict(cx) * cal.amount.to_numpy()
    else:
        severity = pipeline('poisson', cfg['macro'])
        exposure_weights = weights * fit.amount.to_numpy() / float(fit.amount.median())
        severity.fit(x, fit.gross24 / fit.amount, model__sample_weight=exposure_weights)
        unscaled = severity.predict(cx) * cal.amount.to_numpy()
    scale = float(cal.gross24.sum() / unscaled.sum())
    numeric_support = {c: [float(fit[c].quantile(.01)), float(fit[c].quantile(.99))] for c in BASE_NUM}
    reference = {c: float(fit[c].median()) for c in BASE_NUM}
    reference.update({c: str(fit[c].mode().iloc[0]) for c in BASE_CAT})
    return {'config': config, 'specification': cfg, 'probability': probability, 'calibrator': calibrator,
            'severity': severity, 'amount_scale': scale, 'cutoff': cutoff, 'months': 24,
            'numeric_support': numeric_support, 'reference': reference,
            'approved_for_lending': False, 'retrospective': True}


def predict_bundle(bundle, frame, history):
    x = features(frame, history)
    raw = bundle['probability'].predict_proba(x)[:, 1]
    probability = bundle['calibrator'].predict_proba(logodds(raw))[:, 1]
    ratio = bundle['severity'].predict(x)
    expected = ratio * frame.amount.to_numpy() * bundle['amount_scale']
    if bundle['specification']['amount_method'] == 'hurdle':
        expected *= probability
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    assert np.isfinite(expected).all() and (expected >= 0).all()
    return probability, expected
