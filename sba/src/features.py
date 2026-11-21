"""Shared feature contract. No outcome, assignee, resale or servicing fields."""
import numpy as np
import pandas as pd

NUMERIC = ['log_amount', 'guarantee_share', 'initial_rate', 'log_jobs']
CATEGORICAL = ['industry', 'state', 'business_age', 'business_type']
RAW = ['GrossApproval', 'SBAGuaranteedApproval', 'InitialInterestRate', 'JobsSupported', 'NaicsCode', 'ProjectState', 'BusinessAge', 'BusinessType']
INDUSTRIES = {'11':'Agriculture', '21':'Mining', '22':'Utilities', '23':'Construction', '31':'Manufacturing', '32':'Manufacturing', '33':'Manufacturing', '42':'Wholesale', '44':'Retail', '45':'Retail', '48':'Transport', '49':'Transport', '51':'Information', '52':'Finance', '53':'Real estate', '54':'Professional services', '55':'Management', '56':'Administrative services', '61':'Education', '62':'Health care', '71':'Arts and recreation', '72':'Accommodation and food', '81':'Other services', '92':'Public administration'}
NEW = {'New, Less than 1 Year old', 'Startup, Loan Funds will Open Business', 'New Business or 2 years or less'}
EXISTING = {'Less than 4 years old but at least 3', 'Less than 3 years old but at least 2', 'Less than 5 years old but at least 4', 'Existing, 5 or more years', 'Existing or more than 2 years old'}

def transform_raw(raw):
    """Deterministic business transformations; fitted imputation stays in pipeline."""
    x = pd.DataFrame(index=raw.index)
    amount = pd.to_numeric(raw.GrossApproval, errors='coerce')
    guarantee = pd.to_numeric(raw.SBAGuaranteedApproval, errors='coerce')
    rate = pd.to_numeric(raw.InitialInterestRate, errors='coerce')
    jobs = pd.to_numeric(raw.JobsSupported, errors='coerce')
    x['log_amount'] = np.log(amount.where(amount > 0))
    x['guarantee_share'] = (guarantee / amount).where((guarantee >= 0) & (guarantee <= amount))
    x['initial_rate'] = rate.where((rate > 0) & (rate <= 30))
    x['log_jobs'] = np.log1p(jobs.where(jobs >= 0))
    x['industry'] = raw.NaicsCode.str[:2].map(INDUSTRIES).fillna('Unknown')
    x['state'] = raw.ProjectState.str.strip().str.upper().replace('', 'Unknown')
    x['business_age'] = raw.BusinessAge.map(lambda v: 'New' if v in NEW else 'Existing' if v in EXISTING else 'Unknown')
    x['business_type'] = raw.BusinessType.str.strip().str.upper().replace('', 'UNKNOWN')
    return x

def assign_target(df, asof, horizon_months=36):
    """Return nullable label; incomplete follow-up and invalid dates stay unknown."""
    end = df.disb + pd.DateOffset(months=horizon_months)
    valid = df.disb.notna() & df.disb.le(asof) & df.disb.ge(df.app) & end.le(asof)
    valid &= df.status.isin(['PIF', 'EXEMPT', 'CHGOFF'])
    valid &= (df.co.isna() | (df.co.ge(df.disb) & df.co.le(asof)))
    valid &= (df.pif.isna() | (df.pif.ge(df.disb) & df.pif.le(asof)))
    valid &= ~(df.co.notna() & df.pif.notna())
    valid &= ~(df.status.eq('CHGOFF') & df.co.isna())
    valid &= ~(df.status.eq('PIF') & df.pif.isna())
    valid &= ~(df.co.notna() & df.status.ne('CHGOFF'))
    valid &= ~(df.pif.notna() & df.status.ne('PIF'))
    y = pd.Series(pd.NA, index=df.index, dtype='Int64')
    y.loc[valid] = (df.co.ge(df.disb) & df.co.le(end) & df.status.eq('CHGOFF')).loc[valid].astype(int)
    return y
