"""Cache official public extracts. No borrower identifiers leave this computer."""
import sys,json,time,hashlib,csv,io,urllib.request,urllib.parse,concurrent.futures,argparse
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from public_features import STATE_FIPS,MACRO_IDS
def fetch(item):
 name,url,expected=item;p=ROOT/'data/external'/name;p.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(4):
  try:
   if p.exists():b=p.read_bytes()
   else:
    with urllib.request.urlopen(url,timeout=30) as r:b=r.read()
   if expected:
    header=next(csv.reader(io.StringIO(b.decode())))
    if header[1:]!=expected:raise ValueError(f'Unexpected vintage columns {header[:3]}')
   if not p.exists():p.write_bytes(b)
   return {'file':str(p.relative_to(ROOT)),'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'expected_vintage_columns':expected}
  except Exception as exc:
   print('Retry',name,attempt+1,str(exc)[:120],flush=True)
   if attempt==3:raise
   time.sleep(2**attempt)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--holdout',action='store_true');args=parser.parse_args()
 tasks=[]
 if args.holdout:
  assert (ROOT/'reports/phase3/selection_before_new_test.json').exists(),'Select models before downloading new outcomes'
  for span in ['FY2010_FY2019','FY2020_Present']:
   name=f'FOIA_7a_{span}_asof_260930.csv';tasks.append(('sba_new/'+name,'https://data.sba.gov/sites/default/files/uploaded_resources/'+name,None))
 else:
  ids=MACRO_IDS+[s+'UR' for s in STATE_FIPS]
  for quarter in pd.period_range('2009Q4','2023Q3',freq='Q'):
   vintage=quarter.start_time.strftime('%Y-%m-%d')
   for i in range(0,len(ids),12):
    block=ids[i:i+12];params={'id':','.join(block),'vintage_date':','.join([vintage]*len(block))}
    url='https://alfred.stlouisfed.org/graph/alfredgraph.csv?'+urllib.parse.urlencode(params)
    tasks.append((f'alfred/{vintage}_{i//12}.csv',url,[v+'_'+vintage.replace('-','') for v in block]))
  for year in range(2005,2021):
   name=f'cbp{year%100:02d}st.zip';tasks.append(('cbp/'+name,f'https://www2.census.gov/programs-surveys/cbp/datasets/{year}/{name}',None))
 results=[];failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
  fs={executor.submit(fetch,t):t[0] for t in tasks}
  for future in concurrent.futures.as_completed(fs):
   try:results.append(future.result())
   except Exception as e:failures.append({'file':fs[future],'error':str(e)})
   if (len(results)+len(failures))%15==0:print(f'Public files: {len(results)} ready, {len(failures)} failed, {len(tasks)} total',flush=True)
 out=ROOT/'reports/phase3';out.mkdir(parents=True,exist_ok=True)
 name='new_test_source_manifest.json' if args.holdout else 'public_source_manifest.json'
 (out/name).write_text(json.dumps({'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'sources':sorted(results,key=lambda v:v['file']),'failures':failures},indent=2)+'\n')
 print('Downloaded/cached',len(results),'files. Failures',failures,flush=True)
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
