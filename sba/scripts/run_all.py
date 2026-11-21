import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for script in ['prepare.py','catalogue.py','train.py','validate.py','explain.py','report.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/script)],check=True,cwd=ROOT)
print('Complete. Browser inference parity can be checked with: node tests/test_scoring.mjs')
