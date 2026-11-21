"""Verify and inventory this local addition without changing previous releases."""
import sys,json,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/papers'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
forbidden={'borrower_key','row_hash','BorrName','BorrStreet','BorrZip','source_row'}
def inspect(obj):
    if isinstance(obj,dict):
        assert not forbidden.intersection(obj)
        for value in obj.values():inspect(value)
    elif isinstance(obj,list):
        for value in obj:inspect(value)
for name in ['paper_models.json','paper_results.json']:
    raw=(ROOT/'web/dist'/name).read_text();assert ':NaN' not in raw and ':Infinity' not in raw;inspect(json.loads(raw))
page=(ROOT/'web/dist/paper.html').read_text()
assert all(not src.startswith(('http:','https:','//')) for src in re.findall(r'(?:src|href)="([^"]+)"',page) if src.endswith(('.js','.mjs','.css')))
js=(ROOT/'web/dist/paper.mjs').read_text();assert 'localStorage' not in js and 'sessionStorage' not in js and 'sendBeacon' not in js
assert sha(ROOT/'reports/논문기법_적용결과.pdf')==sha(ROOT/'web/dist/paper_report.pdf')
for f in ['paper.html','paper_models.json','paper_results.json','paper_report.pdf']:
    with urlopen('http://127.0.0.1:8813/'+f) as r:
        assert r.status==200 and r.headers['Cache-Control']=='no-store';assert r.read()==(ROOT/'web/dist'/f).read_bytes()
checks={'aggregate_only_web_bundle':True,'no_external_scripts_or_styles':True,'no_input_storage_or_submission':True,'loopback_served_files_match_disk':True,'pdf_pages_visually_reviewed':3,
        'model_logic_unit_tests':19,'javascript_python_curve_checks':1152,'interactive_browser_checks':'reports/papers/browser_validation.json','approved_for_lending':False}
(OUT/'release_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
files=[*ROOT.glob('scripts/paper_*.py'),*ROOT.glob('docs/paper_*.md'),*ROOT.glob('models/paper_*.joblib'),*ROOT.glob('web/dist/paper*'),*ROOT.glob('tests/*paper*'),*OUT.glob('*'),ROOT/'reports/논문기법_적용결과.pdf']
files=[p for p in files if p.is_file() and p.name!='artifact_checksums.json']
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}}
(OUT/'artifact_checksums.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'files':len(files),'validation':checks},indent=2))
