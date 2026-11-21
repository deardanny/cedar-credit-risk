"""Official credit-cycle vintage downloads, including ZIP responses and missing archives."""
import sys,json,urllib.parse,urllib.request,urllib.error,concurrent.futures,hashlib,zipfile,io,time
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
IDS=['DRBLACBS','CORBLACBS','DRTSCIS']

def request(date,ids,label):
 url='https://alfred.stlouisfed.org/graph/alfredgraph.csv?'+urllib.parse.urlencode({'id':','.join(ids),'vintage_date':','.join([date]*len(ids))})
 path=ROOT/f'data/external/hour/credit_raw/{date}_{label}.bin';path.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(4):
  try:
   if path.exists():b=path.read_bytes()
   else:
    with urllib.request.urlopen(url,timeout=30) as r:b=r.read()
    path.write_bytes(b)
   stamp=date.replace('-','');frames=[]
   if zipfile.is_zipfile(io.BytesIO(b)):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
     for name in z.namelist():
      if name.endswith('.csv'):frames.append(pd.read_csv(z.open(name)))
   else:frames=[pd.read_csv(io.BytesIO(b))]
   combined=None
   for frame in frames:
    assert frame.columns[0]=='observation_date'
    assert all(c.endswith('_'+stamp) for c in frame.columns[1:])
    combined=frame if combined is None else combined.merge(frame,on='observation_date',how='outer',validate='one_to_one')
   return combined,{'file':str(path.relative_to(ROOT)),'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'http_status':200}
  except urllib.error.HTTPError as e:
   if e.code==404:return None,{'file':None,'url':url,'http_status':404,'reason':'requested historical archive unavailable'}
   if attempt==3:raise
  except Exception:
   if attempt==3:raise
  time.sleep(2**attempt)

def quarter(date):
 data,meta=request(date,IDS,'combined');sources=[meta]
 if data is None:
  for series in IDS:
   frame,m=request(date,[series],series);sources.append(m)
   if frame is not None:data=frame if data is None else data.merge(frame,on='observation_date',how='outer',validate='one_to_one')
 stamp=date.replace('-','');expected=[s+'_'+stamp for s in IDS]
 if data is None:data=pd.DataFrame(columns=['observation_date']+expected)
 for c in expected:
  if c not in data:data[c]=float('nan')
 data=data[['observation_date']+expected].sort_values('observation_date');p=ROOT/f'data/external/hour/credit/{date}.csv';p.parent.mkdir(parents=True,exist_ok=True);data.to_csv(p,index=False)
 return {'quarter':date,'canonical_file':str(p.relative_to(ROOT)),'canonical_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'unavailable_series':[s for s in IDS if data[s+'_'+stamp].notna().sum()==0],'raw_sources':sources}

def main():
 records=[];errors=[];dates=[q.start_time.strftime('%Y-%m-%d') for q in pd.period_range('2009Q4','2023Q3',freq='Q')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  fs={pool.submit(quarter,d):d for d in dates}
  for f in concurrent.futures.as_completed(fs):
   try:records.append(f.result())
   except Exception as e:errors.append({'quarter':fs[f],'error':str(e)})
   print('Credit vintages',len(records),'/',len(dates),'failures',len(errors),flush=True)
 (ROOT/'reports/hour/credit_sources.json').write_text(json.dumps({'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'quarters':records,'failures':errors},indent=2)+'\n')
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
