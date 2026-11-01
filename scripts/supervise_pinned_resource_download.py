"""Wait for an existing download, retry incomplete pinned ranges, and require full hashes."""
import argparse
from datetime import datetime,timezone
import json,os
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gsdeploy.profiling import resources
from gsdeploy.profiling.runner import save

def legacy_alive(pid,study):
 proc=Path('/proc')/str(pid)
 try:
  if proc.joinpath('stat').read_text().split(') ',1)[1].split()[0]=='Z':return False
  command=proc.joinpath('cmdline').read_bytes().decode().replace('\0',' ')
  return 'gsdeploy.profiling.resources' in command and str(study) in command
 except FileNotFoundError:return False

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ['study','metadata','assets','output','state']:parser.add_argument('--'+name,required=True,type=Path)
 parser.add_argument('--legacy-pid',type=int,required=True);args=parser.parse_args()
 state={'status':'WAITING_FOR_LEGACY_DOWNLOAD','pid':os.getpid(),'legacy_pid':args.legacy_pid,'attempt':0,'gpu_started':False,'pool_admitted':False}
 save(args.state,state)
 original=resources.download_file
 try:
  while legacy_alive(args.legacy_pid,args.study):time.sleep(60)
  inventory_path=args.output/'resource-inventory.json'
  if inventory_path.exists():
   inventory=json.loads(inventory_path.read_text())
   if inventory['status']=='COMPLETE' and all(f['status']=='VERIFIED' for m in inventory['models'].values() for f in m['files']):
    state.update(status='COMPLETE_ALL_FILES_VERIFIED',files_verified=True);return
  for attempt in range(1,9):
   workers=8 if attempt<=3 else 4
   resources.download_file=lambda url,path,entry:original(url,path,entry,workers=workers)
   state.update(status='DOWNLOADING',attempt=attempt,range_workers=workers,updated_utc=datetime.now(timezone.utc).isoformat());save(args.state,state)
   result=resources.prepare_resources(args.study,args.metadata,args.assets,args.output)
   failures=[{'name':f['name'],'error':f.get('error')} for m in result['models'].values() for f in m['files'] if f['status']!='VERIFIED']
   with args.state.with_suffix('.attempts.jsonl').open('a') as handle:handle.write(json.dumps({'attempt':attempt,'status':result['status'],'failures':failures,'utc':datetime.now(timezone.utc).isoformat()})+'\n')
   if result['status']=='COMPLETE':state.update(status='COMPLETE_ALL_FILES_VERIFIED',files_verified=True);break
   if any('hash' in str(f['error']).lower() for f in failures):state.update(status='INTEGRITY_FAILURE_REQUIRES_REVIEW',failures=failures);break
   if attempt==8:state.update(status='RETRIES_EXHAUSTED_REQUIRES_REVIEW',failures=failures);break
   state.update(status='RETRY_BACKOFF',failures=failures);save(args.state,state);time.sleep(min(60,15*attempt))
 except BaseException as error:
  state.update(status='SUPERVISOR_FAILED',error=type(error).__name__+': '+str(error));raise
 finally:
  state['updated_utc']=datetime.now(timezone.utc).isoformat();save(args.state,state)

if __name__=='__main__':main()
