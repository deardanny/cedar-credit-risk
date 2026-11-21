"""Join the supplied dictionary to whole-archive inventory and modelling decisions."""
import csv,json
from pathlib import Path
import openpyxl
ROOT=Path(__file__).resolve().parents[1]
FEATURES={
 'GrossApproval':'log(amount); positive amount required',
 'SBAGuaranteedApproval':'guaranteed amount / total amount; 0 to 1 required',
 'InitialInterestRate':'approval-time rate; outside (0,30] treated as missing; training median imputation',
 'JobsSupported':'log1p(jobs); negative treated as missing; training median imputation; self-reported application estimate',
 'NaicsCode':'two-digit industry; merge 31-33, 44-45 and 48-49; unknown category retained',
 'ProjectState':'uppercase project state; empty becomes Unknown',
 'BusinessAge':'harmonize New/Existing/Unknown; source definitions change around 2018; sensitivity tested',
 'BusinessType':'uppercase legal business type; empty becomes UNKNOWN'}
CONTROL={
 'AsOfDate':'fixed snapshot validation', 'Program':'7(a) source-file cohort selection',
 'ApprovalDate':'date validity; never an input feature', 'ApprovalFY':'source inventory and FY2010+ archive selection',
 'FirstDisbursementDate':'36-month outcome clock and temporal split',
 'RevolverStatus':'term-loan cohort; actual values N/Y differ from dictionary 0/1',
 'LoanStatus':'outcome consistency and cancelled/undisbursed exclusions; never an input',
 'ChargeOffDate':'36-month outcome and date validity; never an input',
 'PaidInFullDate':'date/status consistency; never an input',
 'GrossChargeOffAmount':'outcome-quality audit only; not EAD, LGD or an input',
 'BorrName':'normalized SHA256 borrower linking only; no raw name in analytic file',
 'BorrStreet':'normalized SHA256 borrower linking only; no raw street in analytic file',
 'BorrZip':'first five characters for borrower linking only'}
EXCLUDE={
 'LocationID':'lender identifier, not a unique loan ID',
 'SoldSecMrktInd':'post-origination sale indicator; target leakage risk',
 'TermInMonths':'original approval-time value not established by dictionary',
 'CollateralInd':'reported collateral timing not established',
 'NaicsDescription':'redundant with NAICS code',
 'FixedorVariableInterestInd':'not used in compact baseline feature set',
 'ProcessingMethod':'not used in compact baseline feature set',
 'BorrCity':'fine geography excluded; project state retained',
 'BorrState':'borrower state excluded; project state retained'}
def main():
 c=json.loads((ROOT/'config.json').read_text());p=Path(c['dictionary']);p=p if p.is_absolute() else ROOT/p
 inv=json.loads((ROOT/'reports/source_inventory.json').read_text())['csv_inventory'];rows=[]
 for s in openpyxl.load_workbook(p,read_only=True,data_only=True):
  program='7a' if s.title.startswith('7') else '504';sources=[v for v in inv if Path(v['file']).name.startswith('FOIA_'+program+'_')]
  n=sum(v['rows'] for v in sources)
  for row in list(s.values)[1:]:
   field,definition=row[:2]
   if not field:continue
   missing=sum(v['missing'].get(field,0) for v in sources)
   if program=='504':role='inventory_only';decision='504 reviewed separately; excluded from 7(a) model population'
   elif field in FEATURES:role='predictor';decision=FEATURES[field]
   elif field in CONTROL:role='cohort_target_or_audit';decision=CONTROL[field]
   else:
    role='excluded';decision=EXCLUDE.get(field,'current lender/assignee information excluded' if field.startswith('Bank') else 'not used in compact baseline feature set')
   dtype='ISO date string' if field.endswith('Date') else 'numeric string' if field in {'GrossApproval','SBAGuaranteedApproval','InitialInterestRate','JobsSupported','GrossChargeOffAmount','TermInMonths','ThirdPartyDollars','ApprovalFY'} else 'categorical or free text string'
   rows.append({'program':program,'field':field,'source_files':';'.join(v['file'] for v in sources),'supplied_definition':definition,'stored_type':dtype,'archive_rows':n,'blank_rows':missing,'blank_fraction':missing/n if n else '', 'role':role,'decision':decision})
 out=ROOT/'reports/field_catalogue.csv'
 with out.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(f'Field catalogue: {len(rows)} program-specific field definitions')
if __name__=='__main__':main()
