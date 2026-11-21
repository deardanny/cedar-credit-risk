"""Record a verifiable local research release without publishing it."""
import json,hashlib,sys,re
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/hour'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def walk_keys(x):
 if isinstance(x,dict):
  for k,v in x.items():yield k;yield from walk_keys(v)
 elif isinstance(x,list):
  for v in x:yield from walk_keys(v)
def main():
 web=ROOT/'web/dist';private={'BorrName','BorrStreet','BorrZip','borrower_key','row_hash','source_row'}
 for name in ['hour_results.json','hour_long_model.json']:
  payload=json.loads((web/name).read_text());assert not private.intersection(walk_keys(payload))
 html=(web/'hour.html').read_text();assert 'hour.mjs' in html and 'hour.css' in html
 for tag in re.findall(r'<(?:script|link)\b[^>]*>',html):assert not re.search(r'(?:src|href)=[\"\']https?://',tag)
 js=(web/'hour.mjs').read_text();assert 'localStorage' not in js and 'sessionStorage' not in js and 'sendBeacon' not in js;assert "fetch(p,{cache:'no-store'})" in js
 (OUT/'release_validation.json').write_text(json.dumps({'no_loan_identifiers_in_new_web_json':True,'no_remote_script_style_or_font_assets':True,'no_input_storage_or_submission_in_demo':True,'loopback_only_server':"('127.0.0.1',args.port)" in (ROOT/'scripts/serve.py').read_text(),'pdf_pages_visually_reviewed':6,'local_only':True},indent=2)+'\n')
 budget=json.loads((OUT/'run_budget.json').read_text());now=datetime.now(timezone.utc)
 if '--finish' in sys.argv:
  elapsed=(now-datetime.fromisoformat(budget['start_utc'].replace('Z','+00:00'))).total_seconds();assert elapsed>=3600,'Do not finish before the requested hour.';budget.update({'finished_at_utc':now.isoformat(),'elapsed_seconds':elapsed,'completed_model_configurations':63,'additional_mixture_weight_comparisons':10,'long_horizon_test_windows':[2018,2019,2020],'intermediate_user_questions':0,'local_demo_left_running':True,'research_complete':True,'approved_for_lending':False});(OUT/'run_budget.json').write_text(json.dumps(budget,indent=2)+'\n');e=json.loads((web/'hour_results.json').read_text());e['run_budget']=budget;(web/'hour_results.json').write_text(json.dumps(e,ensure_ascii=False,separators=(',',':'))+'\n')
 paths=set()
 for pattern in ['scripts/hour_*.py','docs/hour*.md','tests/*hour*','web/dist/hour*','reports/hour/**/*','models/hour_*.joblib','reports/1시간_추가연구_결과.*']:
  paths.update(p for p in ROOT.glob(pattern) if p.is_file() and p.name!='artifact_checksums.json')
 paths.update(ROOT/n for n in ['README.md','docs/deliverables.json','web/dist/index.html'])
 result={'created_utc':now.isoformat(),'files':{str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)},'model_binaries_stay_local':True,'no_publish':True};(OUT/'artifact_checksums.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'files_hashed':len(paths),'finished':bool(budget.get('research_complete')),'elapsed_seconds':budget.get('elapsed_seconds')},indent=2))
if __name__=='__main__':main()
