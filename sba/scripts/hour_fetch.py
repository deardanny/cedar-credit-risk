"""Extend official ALFRED vintages without changing the frozen phase-3 cache."""
import sys,json,urllib.parse,concurrent.futures
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from public_features import STATE_FIPS,MACRO_IDS
from fetch_public import fetch
def main():
 ids=MACRO_IDS+[s+'UR' for s in STATE_FIPS];tasks=[]
 for quarter in pd.period_range('2023Q4','2025Q3',freq='Q'):
  date=quarter.start_time.strftime('%Y-%m-%d')
  for i in range(0,len(ids),12):
   block=ids[i:i+12];params={'id':','.join(block),'vintage_date':','.join([date]*len(block))};url='https://alfred.stlouisfed.org/graph/alfredgraph.csv?'+urllib.parse.urlencode(params)
   tasks.append((f'hour/alfred/{date}_{i//12}.csv',url,[s+'_'+date.replace('-','') for s in block]))
 records=[];failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  fs={pool.submit(fetch,t):t[0] for t in tasks}
  for f in concurrent.futures.as_completed(fs):
   try:records.append(f.result())
   except Exception as e:failures.append({'file':fs[f],'error':str(e)})
   print('Vintages',len(records),'/',len(tasks),'failures',len(failures),flush=True)
 p=ROOT/'reports/hour';p.mkdir(exist_ok=True);(p/'new_public_sources.json').write_text(json.dumps({'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'sources':records,'failures':failures},indent=2)+'\n')
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
