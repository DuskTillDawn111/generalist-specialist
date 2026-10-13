"""Pinned six-checkpoint transport; no inference or measurement."""
import concurrent.futures,hashlib,requests
from module14.common import *
from module13b.download import fetch,ranges

def tv(entry):
 d=Path(entry['directory']);d.mkdir(parents=True,exist_ok=True);p=d/entry['filename'];sess=requests.Session();sess.trust_env=False
 if not p.exists():
  r=sess.head(entry['weight_url'],timeout=60);r.raise_for_status();size=int(r.headers['Content-Length'])
  print('TORCHVISION_TRANSPORT',entry['model_id'],size,flush=True)
  # Official CDN supports ranges, with complete payload hash verification.
  part=ranges(entry['weight_url'],p,size);sha=hashlib.sha256(part.read_bytes()).hexdigest();assert sha.startswith(entry['expected_sha256_prefix']);part.replace(p)
 sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha.startswith(entry['expected_sha256_prefix'])
 return {'model_id':entry['model_id'],'revision':entry['revision'],'resolved_artifact_sha256':sha,'local_directory':str(d),'success':True,'files':[ref(p)],'observable_download_bytes':p.stat().st_size,'url':entry['weight_url']}
def hf(entry):
 repo=entry['model_id'];d=Path(entry['directory']);d.mkdir(parents=True,exist_ok=True);api=read(OUT/'sources'/(repo.replace('/','--')+'.api.json'));assert api['sha']==entry['revision']
 items=[s for s in api['siblings'] if s['rfilename'] in entry['weight_files'] or ('/' not in s['rfilename'] and s['rfilename'].endswith(('.json','.txt','.yml','.yaml','.model','.jinja')) and 'index.' not in s['rfilename'])]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:files=list(ex.map(lambda s:fetch(repo,entry['revision'],s,d),items))
 return {'model_id':repo,'revision':entry['revision'],'resolved_revision':api['sha'],'local_directory':str(d),'success':all(f.get('status')!='DOWNLOAD_FAILED' for f in files),'files':files,'observable_download_bytes':sum(f.get('bytes',0) for f in files)}
def main():
 manifest={'at_utc':now(),'models':[],'completed':False};save('download_manifest.json',manifest)
 for entry in read(OUT/'measurement_protocol_freeze.json')['entries']:
  print('DOWNLOAD_START',entry['model_id'],flush=True)
  try:r=tv(entry) if entry['backend']=='torchvision' else hf(entry)
  except Exception as e:r={'model_id':entry['model_id'],'revision':entry['revision'],'success':False,'error':repr(e)}
  manifest['models'].append(r);save('download_manifest.json',manifest);print('DOWNLOAD_FINISH',entry['model_id'],r['success'],r.get('error'),flush=True)
 manifest.update(completed=True,all_success=all(m['success'] for m in manifest['models']));save('download_manifest.json',manifest)
if __name__=='__main__':main()
