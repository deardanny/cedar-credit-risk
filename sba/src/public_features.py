"""Feature groups and official state-code mappings for public enrichment."""
from features import NUMERIC,CATEGORICAL
STATE_FIPS=dict(zip('AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY'.split(),'01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56'.split()))
MACRO_IDS=['UNRATE','FEDFUNDS','CPIAUCSL','INDPRO','GS10','TB3MS','MPRIME']
MACRO=['us_unemployment','us_unemployment_change','fed_funds','prime_rate','inflation_yoy','industrial_growth_yoy','yield_curve','state_unemployment','state_unemployment_change','state_unemployment_gap','rate_over_prime']
EXTRA_NUM=['log_amount_per_job','log_unguaranteed_amount']
EXTRA_CAT=['processing_method','naics3']
CBP=['cbp_log_establishments','cbp_employees_per_establishment','cbp_log_pay_per_employee','cbp_establishment_growth','cbp_employment_growth']
GROUPS={
 'base':(NUMERIC,CATEGORICAL),
 'loan_extended':(NUMERIC+EXTRA_NUM,CATEGORICAL+EXTRA_CAT),
 'macro_vintage':(NUMERIC+EXTRA_NUM+MACRO,CATEGORICAL+EXTRA_CAT),
 'cbp_exploratory':(NUMERIC+EXTRA_NUM+MACRO+CBP,CATEGORICAL+EXTRA_CAT)}
