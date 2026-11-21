"""Seal a first-run model selection before fetching the reserved SBA snapshot."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'reports/phase3/model_freeze_manifest.json'
 if out.exists():raise SystemExit('A freeze manifest already exists. Keep it unchanged; use validate_loss.py for replay checks.')
 if (ROOT/'data/external/sba_new').exists():raise SystemExit('Reserved SBA data already present. This run cannot be declared blind to that snapshot.')
 paths=[ROOT/'docs/phase3_plan.md',ROOT/'scripts/train_loss.py',ROOT/'scripts/enrich_public.py',ROOT/'src/public_features.py',ROOT/'reports/phase3/selection_before_new_test.json',*sorted(ROOT.glob('models/phase3_*.joblib'))]
 if len(list(ROOT.glob('models/phase3_*.joblib')))!=4:raise SystemExit('Fit all four frozen models first.')
 manifest={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'before_new_test_download':True,'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
 out.write_text(json.dumps(manifest,indent=2)+'\n');print(out)
if __name__=='__main__':main()
