"""Collect official public context, preserve provenance, and audit exact SBA joins.

This collection does not retrain or replace any frozen model. Current revised
economic/bank histories are not certified approval-time information.
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/external/underwriting_public_v1'
OUT = ROOT / 'reports/underwriting_public_v1'
TABLES = ROOT / 'data/processed/underwriting_public_v1'
sys.path.insert(0, str(ROOT / 'src'))
from public_features import STATE_FIPS

FDIC_FIELDS = ['CERT', 'REPDTE', 'NAME', 'ASSET', 'DEP', 'LNLSNET', 'EQ', 'NETINC', 'ROA', 'RBC1AAJ', 'NCLNLSR']
BDS_VALUES = ['firms', 'estabs', 'emp', 'estabs_entry_rate', 'estabs_exit_rate', 'job_creation_rate', 'job_destruction_rate', 'net_job_creation_rate', 'firmdeath_firms']
Q_VALUES = ['annual_avg_estabs', 'annual_avg_emplvl', 'avg_annual_pay', 'annual_avg_wkly_wage', 'oty_annual_avg_estabs_pct_chg', 'oty_annual_avg_emplvl_pct_chg', 'oty_avg_annual_pay_pct_chg']


def now(): return datetime.now(timezone.utc).isoformat()
def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def fetch(task):
    name, url, kind = task
    p = RAW / name
    p.parent.mkdir(parents=True, exist_ok=True)
    side = p.with_suffix(p.suffix + '.source.json')
    if p.exists() and side.exists():
        m = json.loads(side.read_text())
        if digest(p) == m['sha256']:
            return m
    last = None
    for attempt in range(3):
        try:
            time.sleep(2)  # Respect official endpoint request limits.
            req = urllib.request.Request(url, headers={'User-Agent': 'CedarResearch/1.0 public-data-analysis', 'Accept': '*/*'})
            with urllib.request.urlopen(req, timeout=60) as response:
                headers = dict(response.headers)
                final_url = response.url
                tmp = p.with_suffix(p.suffix + '.partial')
                with tmp.open('wb') as f:
                    while True:
                        b = response.read(1024 * 1024)
                        if not b: break
                        f.write(b)
            if kind == 'zip':
                with zipfile.ZipFile(tmp) as z:
                    assert z.testzip() is None
            elif kind == 'fdic':
                value = json.loads(tmp.read_text())
                assert len(value['data']) == value['meta']['total'], 'Truncated FDIC response'
            elif kind == 'csv':
                assert '<html' not in tmp.read_bytes()[:300].decode(errors='ignore').lower()
            tmp.replace(p)
            m = {'file': str(p.relative_to(ROOT)), 'url': url, 'final_url': final_url, 'retrieved_at_utc': now(), 'bytes': p.stat().st_size,
                 'sha256': digest(p), 'content_type': headers.get('Content-Type'), 'last_modified_header': headers.get('Last-Modified'),
                 'kind': kind, 'status': 'downloaded', 'historical_vintage_verified': False}
            if kind == 'fdic': m['rows'] = len(value['data']); m['index'] = value['meta'].get('index')
            side.write_text(json.dumps(m, indent=2) + '\n')
            return m
        except Exception as e:
            last = str(e)
            if getattr(e, 'code', None) in [403, 404]: break
            if getattr(e, 'code', None) == 429:
                time.sleep(min(30, max(10, int(e.headers.get('Retry-After', '10')))))
            else:
                time.sleep(attempt + 1)
    return {'file': str(p.relative_to(ROOT)), 'url': url, 'status': 'unavailable', 'error': last, 'attempted_at_utc': now()}


def tasks():
    result = [
        ('bds/state_sector_2023.csv', 'https://www2.census.gov/programs-surveys/bds/tables/time-series/2023/bds2023_st_sec.csv', 'csv'),
        ('qcew/area_titles.csv', 'https://data.bls.gov/cew/doc/titles/area/area_titles.csv', 'csv'),
        ('fdic/definitions.yaml', 'https://api.fdic.gov/banks/docs/risview_properties.yaml', 'text'),
    ]
    for year in range(2007, 2026):
        result.append((f'qcew/{year}_annual.zip', f'https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip', 'zip'))
    for quarter in pd.period_range('2008Q4', '2026Q2', freq='Q'):
        date = quarter.end_time.strftime('%Y%m%d')
        q = {'filters': 'REPDTE:' + date, 'fields': ','.join(FDIC_FIELDS), 'limit': 10000, 'format': 'json', 'sort_by': 'CERT', 'sort_order': 'ASC'}
        result.append((f'fdic/{date}.json', 'https://api.fdic.gov/banks/financials?' + urllib.parse.urlencode(q), 'fdic'))
    return result


def fetch_all():
    OUT.mkdir(parents=True, exist_ok=True)
    # Reuse bytes fetched from the same official endpoints during source discovery.
    probe = Path('/private/tmp/cedar-data-collection')
    existing = {'bds/state_sector_2023.csv': 'bds.bin', 'qcew/2022_annual.zip': 'qcew.bin'}
    for name, url, kind in tasks():
        if name in existing and (probe / existing[name]).exists() and not (RAW / name).exists():
            p = RAW / name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(probe / existing[name], p)
            m = {'file': str(p.relative_to(ROOT)), 'url': url, 'retrieved_at_utc': now(), 'retrieval_note': 'Downloaded in this collection session; reused discovery cache',
                 'sha256': digest(p), 'bytes': p.stat().st_size, 'kind': kind, 'status': 'downloaded', 'historical_vintage_verified': False}
            p.with_suffix(p.suffix + '.source.json').write_text(json.dumps(m, indent=2) + '\n')
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(fetch, task): task[0] for task in tasks()}
        for future in concurrent.futures.as_completed(futures):
            m = future.result(); records.append(m)
            save('manifest.json', {'updated_at_utc': now(), 'sources': sorted(records, key=lambda x:x['file']), 'expected_sources': len(futures)})
            print('COLLECT', len(records), '/', len(futures), futures[future], m['status'], flush=True)
    save('access_limits.json', {'sec_company_ticker_directory': {'url': 'https://www.sec.gov/files/company_tickers.json', 'status': 'HTTP 403 during official download', 'borrower_financials_joined': 0},
        'private_data': ['borrower financial statements', 'monthly repayment and delinquency records', 'recovery/guarantee cash flows', 'rejected applications'],
        'statement': 'No private data was obtained. Do not substitute lender or industry financials for borrower financials.'})


def source_loans():
    wanted = ['source_file', 'source_row', 'row_hash', 'borrower_key', 'app', 'amount', 'state', 'industry']
    parts = []
    for label, name in [('development', 'hour_short_development.pkl'), ('evaluation_2024', 'hour_medium_holdout.pkl')]:
        a = pd.read_pickle(ROOT / 'data/processed' / name)[wanted].copy()
        a['cohort'] = label; parts.append(a)
    a = pd.concat(parts, ignore_index=True)
    assert not a.duplicated(['source_file', 'source_row']).any()
    raw_fields = ['BorrName', 'BorrStreet', 'BorrZip', 'ApprovalDate', 'GrossApproval', 'ProjectState', 'ProjectCounty', 'NaicsCode', 'BusinessAge', 'BankFDICNumber', 'BankNCUANumber', 'LocationID', 'TermInMonths', 'FixedorVariableInterestInd', 'CollateralInd', 'FranchiseCode', 'FranchiseName']
    fragments = []
    for name, group in a.groupby('source_file'):
        needed = set(group.source_row)
        offset = 0
        for raw in pd.read_csv(ROOT / 'data/external/sba_new' / name, usecols=raw_fields, dtype=str, keep_default_na=False, chunksize=100000):
            raw['source_row'] = np.arange(offset + 2, offset + 2 + len(raw)); offset += len(raw)
            raw = raw[raw.source_row.isin(needed)].copy()
            if raw.empty: continue
            key = raw.BorrName.str.upper().str.replace(r'[^A-Z0-9]', '', regex=True) + '|' + raw.BorrStreet.str.upper().str.replace(r'[^A-Z0-9]', '', regex=True) + '|' + raw.BorrZip.str[:5]
            raw['verified_borrower_key'] = key.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
            raw['source_file'] = name
            # Do not duplicate names or street addresses in the feature matrix.
            fragments.append(raw.drop(columns=['BorrName', 'BorrStreet', 'BorrZip']))
    a = a.merge(pd.concat(fragments, ignore_index=True), on=['source_file', 'source_row'], how='left', validate='one_to_one')
    assert a.borrower_key.eq(a.verified_borrower_key).all()
    assert a.app.eq(pd.to_datetime(a.ApprovalDate)).all()
    assert np.allclose(a.amount, pd.to_numeric(a.GrossApproval))
    a['fdic_cert'] = pd.to_numeric(a.BankFDICNumber, errors='coerce').astype('Int64')
    a['state_fips'] = a.state.map(STATE_FIPS)
    sector = a.NaicsCode.str[:2]
    a['sector'] = sector.replace({'31':'31-33','32':'31-33','33':'31-33','44':'44-45','45':'44-45','48':'48-49','49':'48-49'})
    a['term_months'] = pd.to_numeric(a.TermInMonths, errors='coerce').where(lambda x: x.between(1, 600))
    a['rate_type'] = a.FixedorVariableInterestInd.replace('', np.nan)
    a['collateral_reported'] = a.CollateralInd.replace('', np.nan)
    a['business_age_detail'] = a.BusinessAge.replace('', np.nan)
    a['naics6'] = a.NaicsCode.where(a.NaicsCode.str.fullmatch(r'\d{6}'))
    return a


def county_key(text):
    text = str(text).upper().strip()
    text = re.sub(r'\s+(COUNTY|PARISH|BOROUGH|CENSUS AREA|MUNICIPALITY)$', '', text)
    # Keep CITY distinct from COUNTY; do not fuzzy-match ambiguous geography.
    return re.sub(r'[^A-Z0-9]', '', text)


def geography(a):
    areas = pd.read_csv(RAW / 'qcew/area_titles.csv', dtype=str, keep_default_na=False)
    areas = areas[areas.area_fips.str.fullmatch(r'\d{5}') & ~areas.area_fips.str.endswith('000')].copy()
    areas['state'] = areas.area_fips.str[:2].map({v:k for k,v in STATE_FIPS.items()})
    areas = areas[areas.state.notna()].copy()
    areas['county_key'] = areas.area_title.str.rsplit(',', n=1).str[0].map(county_key)
    # The official title file repeats identical rows across code-list sections.
    # Identical IDs are duplicates, not competing geography candidates.
    areas = areas.drop_duplicates(['state', 'county_key', 'area_fips'])
    duplicates = areas.duplicated(['state', 'county_key'], keep=False)
    save('county_ambiguities.json', areas.loc[duplicates].to_dict('records'))
    areas = areas[~duplicates & areas.state.notna()]
    a['county_key'] = a.ProjectCounty.map(county_key)
    a = a.merge(areas[['state', 'county_key', 'area_fips', 'area_title']], on=['state', 'county_key'], how='left', validate='many_to_one')
    a.rename(columns={'area_fips':'county_fips', 'area_title':'county_title'}, inplace=True)
    return a


def normalized_public():
    b = pd.read_csv(RAW / 'bds/state_sector_2023.csv', dtype=str, keep_default_na=False)
    b['state_fips'] = b.st.str.zfill(2)
    b['year'] = pd.to_numeric(b.year).astype(int)
    for col in BDS_VALUES: b[col] = pd.to_numeric(b[col], errors='coerce')
    b = b[['year', 'state_fips', 'sector'] + BDS_VALUES]
    assert not b.duplicated(['year', 'state_fips', 'sector']).any()
    b.to_csv(TABLES / 'bds_state_sector.csv.gz', index=False)
    bank = []
    for p in sorted((RAW / 'fdic').glob('????????.json')):
        d = json.loads(p.read_text())
        assert len(d['data']) == d['meta']['total']
        bank.extend(row['data'] for row in d['data'])
    f = pd.DataFrame(bank)
    f['fdic_cert'] = pd.to_numeric(f.CERT).astype('Int64')
    f['report_date'] = pd.to_datetime(f.REPDTE, format='%Y%m%d')
    for col in FDIC_FIELDS[3:]: f[col] = pd.to_numeric(f[col], errors='coerce')
    assert not f.duplicated(['fdic_cert', 'report_date']).any()
    # Buffer is an explicit exploratory assumption, not a verified filing date.
    f['assumed_available_date'] = f.report_date + pd.Timedelta(days=90)
    f['bank_equity_assets'] = f.EQ / f.ASSET.where(f.ASSET.gt(0))
    f['bank_loans_deposits'] = f.LNLSNET / f.DEP.where(f.DEP.gt(0))
    f['bank_roa_percent'] = f.ROA
    f['bank_log_assets_thousands_usd'] = np.log1p(f.ASSET.where(f.ASSET.ge(0)))
    f.to_csv(TABLES / 'fdic_quarterly.csv.gz', index=False)
    q_parts = []
    q_raw_counts = {}
    sectors = set(b.sector.unique()) | {'10'}
    for p in sorted((RAW / 'qcew').glob('*_annual.zip')):
        q_raw_counts[p.name] = 0
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as source:
                needed = ['area_fips','own_code','industry_code','size_code','year','disclosure_code','oty_disclosure_code'] + Q_VALUES
                for c in pd.read_csv(source, usecols=needed, dtype=str, keep_default_na=False, chunksize=150000):
                    q_raw_counts[p.name] += len(c)
                    keep = c.area_fips.str.fullmatch(r'\d{5}') & c.own_code.eq('5') & c.industry_code.isin(sectors) & c.size_code.eq('0')
                    c = c.loc[keep, ['area_fips', 'industry_code', 'year', 'disclosure_code', 'oty_disclosure_code'] + Q_VALUES].copy()
                    for col in Q_VALUES:
                        c[col] = pd.to_numeric(c[col], errors='coerce')
                        # Suppression values are not true zeros.
                        flag = c.oty_disclosure_code if col.startswith('oty_') else c.disclosure_code
                        c.loc[flag.str.strip().eq('N'), col] = np.nan
                    q_parts.append(c)
        print('NORMALIZE QCEW', p.name, flush=True)
    q = pd.concat(q_parts, ignore_index=True)
    q.rename(columns={'area_fips':'county_fips', 'industry_code':'sector'}, inplace=True)
    q['year'] = pd.to_numeric(q.year).astype(int)
    assert not q.duplicated(['year','county_fips','sector']).any()
    q.to_csv(TABLES / 'qcew_county_state_sector.csv.gz', index=False)
    save('qcew_raw_counts.json', q_raw_counts)
    return b, f, q


def join_data(a, b, f, q):
    n = len(a)
    # Two-year lag excludes contemporaneous/future reference periods, but cannot
    # remove revisions in a currently downloaded series. These are candidate joins.
    a['context_year'] = a.app.dt.year - 2
    bb = b.rename(columns={'year':'context_year', **{c:'bds_'+c for c in BDS_VALUES}})
    a = a.merge(bb, on=['context_year', 'state_fips', 'sector'], how='left', validate='many_to_one')
    a['bds_reference_end'] = pd.to_datetime(a.context_year.astype(str) + '-12-31')
    qq = q.rename(columns={'year':'context_year', **{c:'qcew_'+c for c in Q_VALUES}})
    a = a.merge(qq.drop(columns=['disclosure_code','oty_disclosure_code']), on=['context_year', 'county_fips', 'sector'], how='left', validate='many_to_one')
    a['collection_row'] = np.arange(n)
    nonmissing = a[a.fdic_cert.notna()].copy()
    fc = ['fdic_cert', 'report_date', 'assumed_available_date', 'bank_equity_assets', 'bank_loans_deposits', 'bank_roa_percent', 'bank_log_assets_thousands_usd']
    # pandas asof requires the date key globally sorted even with a by key.
    joined = pd.merge_asof(nonmissing.sort_values('app'), f[fc].sort_values('assumed_available_date'), left_on='app', right_on='assumed_available_date', by='fdic_cert', direction='backward', tolerance=pd.Timedelta(days=185))
    matched = joined.report_date.notna()
    assert (joined.loc[matched, 'assumed_available_date'] <= joined.loc[matched, 'app']).all()
    a = a.merge(joined[['collection_row'] + fc[1:]], on='collection_row', how='left', validate='one_to_one').sort_values('collection_row')
    assert len(a) == n and not a.row_hash.duplicated().any()
    assert (a.bds_reference_end < a.app).all()
    a['historical_vintage_verified'] = False
    a['approved_for_training'] = False
    # Missing company financials stay explicitly unavailable, never filled with bank/sector values.
    for c in ['borrower_revenue','borrower_operating_cash_flow','borrower_existing_debt','borrower_monthly_debt_service','loan_balance_at_default','net_recovery_cash_flow']:
        a[c] = np.nan
    a.to_csv(TABLES / 'loan_context_candidates.csv.gz', index=False)
    return a


def prepare():
    TABLES.mkdir(parents=True, exist_ok=True)
    b, f, q = normalized_public()
    a = join_data(geography(source_loans()), b, f, q)
    groups = []
    for label, g in a.groupby('cohort'):
        groups.append({'cohort':label, 'loans':len(g), 'county_exact_mapping':int(g.county_fips.notna().sum()), 'fdic_identifier':int(g.fdic_cert.notna().sum()),
          'bank_context_matched':int(g.report_date.notna().sum()), 'bds_sector_context_matched':int(g.bds_estabs_exit_rate.notna().sum()),
          'qcew_sector_context_matched':int(g.qcew_annual_avg_emplvl.notna().sum()), 'term_months_present':int(g.term_months.notna().sum()),
          'collateral_present':int(g.collateral_reported.notna().sum()), 'borrower_financials_matched':0})
    by_year=[]
    for year,g in a.groupby(a.app.dt.year):
        by_year.append({'approval_year':int(year),'loans':len(g),
            'bds_matched':int(g.bds_estabs_exit_rate.notna().sum()),
            'bank_matched':int(g.bank_equity_assets.notna().sum()),
            'qcew_matched':int(g.qcew_annual_avg_emplvl.notna().sum())})
    save('coverage.json', {'created_utc':now(), 'loans':len(a), 'by_cohort':groups, 'by_approval_year':by_year,
        'source_table_rows':{'bds_state_sector':len(b),'fdic_quarterly':len(f),'qcew_county_state_sector':len(q)},
        'source_years':{'bds':[int(b.year.min()),int(b.year.max())], 'fdic':[str(f.report_date.min().date()),str(f.report_date.max().date())], 'qcew':sorted(map(int,q.year.unique()))},
        'field_coverage':{c:float(a[c].notna().mean()) for c in ['term_months','rate_type','collateral_reported','business_age_detail','naics6','county_fips','bds_estabs_exit_rate','bank_equity_assets','qcew_annual_avg_emplvl']},
        'zero_missing_fabrication':True, 'exact_source_row_identity_verified':True, 'joins_are_many_to_one':True,
        'historical_vintage_verified':False, 'approved_for_training':False, 'approved_for_lending':False,
        'time_policy':{'bds_qcew':'reference year = approval year - 2; revision history not certified', 'fdic':'latest report with report_date + 90 days <= approval date, maximum additional age 185 days; actual filing/publication timestamps not certified'},
        'limitations':['BDS establishment exits are not loan defaults', 'FDIC financials describe the lender, not the borrower', 'QCEW payroll describes local workers, not applicant revenue', 'QCEW suppressed cells and unmatched historical geographies stay missing', 'SBA snapshot fields lack a complete approval-time revision history', 'SEC official company directory request returned HTTP 403; no borrower financials joined']})
    a.loc[a.county_fips.isna(), ['state','ProjectCounty']].value_counts(dropna=False).rename('loan_count').reset_index().to_csv(OUT/'unmatched_counties.csv',index=False)
    a.loc[a.report_date.isna(), ['BankFDICNumber','BankNCUANumber']].value_counts(dropna=False).rename('loan_count').reset_index().to_csv(OUT/'unmatched_lender_ids.csv',index=False)
    save('artifacts.json', [{'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(TABLES.glob('*.gz'))])
    print(json.dumps(json.loads((OUT/'coverage.json').read_text()),ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fetch','prepare']);args=parser.parse_args()
    fetch_all() if args.stage=='fetch' else prepare()
