"""Exact 13A snapshots, sequential checkpoint order, no model/runtime imports."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import requests
import os
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/module13_real_specialists'
MODELS=ROOT/'models/module13b'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d): p.write_text(json.dumps(d,indent=2)+'\n')

def ranges(url,p,size):
    """Transport concurrency only. Every returned byte range and final SHA checked."""
    piece=16*1024*1024
    target=Path(str(p)+'.ranged.partial')
    if not target.exists() or target.stat().st_size!=size:
        with target.open('wb') as f:f.truncate(size)
    # Resume fully written extents from a transport-only interrupted attempt.
    # Final full-file SHA is mandatory, including all resumed bytes.
    completed=set()
    fd=os.open(target,os.O_RDONLY)
    try:
        for start in range(0,size,piece):
            try:
                if os.lseek(fd,start,os.SEEK_HOLE)>=min(size,start+piece):completed.add(start)
            except OSError:pass
    finally:os.close(fd)
    print('RESUME_FULL_EXTENTS',p.name,len(completed),flush=True)
    def get(start):
        if start in completed:return
        end=min(size-1,start+piece-1)
        for attempt in range(4):
            session=requests.Session();session.trust_env=False
            try:
                with session.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,90)) as r:
                    r.raise_for_status()
                    assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{size}', 'range mismatch'
                    written=0
                    with target.open('r+b') as f:
                        f.seek(start)
                        for chunk in r.iter_content(1048576):f.write(chunk);written+=len(chunk)
                    assert written==end-start+1
                    return
            except Exception:
                if attempt==3:raise
            finally:session.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as ex:
        for index,_ in enumerate(ex.map(get,range(0,size,piece))):
            if index%16==0:print('RANGE_PROGRESS',p.name,min(size,(index+1)*piece),size,flush=True)
    return target

def fetch(repo,revision,item,directory):
    name=item['rfilename'];p=directory/name;p.parent.mkdir(parents=True,exist_ok=True)
    expected=(item.get('lfs') or {}).get('sha256')
    if p.exists():
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.stat().st_size==item['size'] and (not expected or sha==expected):
            return {'file':name,'path':str(p),'bytes':p.stat().st_size,'sha256':sha,'reused':True,'lfs_sha_verified':bool(expected)}
    errors=[]
    for host in ['hf-mirror.com','huggingface.co']:
        url=f'https://{host}/{repo}/resolve/{revision}/{name}'
        try:
            session=requests.Session();session.trust_env=False
            part=Path(str(p)+'.partial');hasher=hashlib.sha256()
            if item['size']>32*1024*1024:
                part=ranges(url,p,item['size'])
                with part.open('rb') as f:
                    for chunk in iter(lambda:f.read(4*1024*1024),b''):hasher.update(chunk)
            else:
                with session.get(url,stream=True,timeout=(20,90)) as r:
                    r.raise_for_status()
                    with part.open('wb') as f:
                        for chunk in r.iter_content(4*1024*1024):
                            f.write(chunk);hasher.update(chunk)
            assert part.stat().st_size==item['size'], 'size mismatch'
            if expected and hasher.hexdigest()!=expected:
                part.unlink()
                raise AssertionError('LFS sha mismatch; invalid partial removed before full transport retry')
            if item.get('blobId') and not expected:
                data=part.read_bytes()
                assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['blobId'],'git blob mismatch'
            part.replace(p)
            stale=Path(str(p)+'.partial')
            if stale.exists():stale.unlink()
            return {'file':name,'path':str(p),'bytes':p.stat().st_size,'sha256':hasher.hexdigest(),
                    'source_url':url,'lfs_sha_verified':bool(expected),'earlier_transport_errors':errors}
        except Exception as e:
            errors.append({'host':host,'type':type(e).__name__,'reason':str(e).split('?')[0][:300]})
    return {'file':name,'path':str(p),'status':'DOWNLOAD_FAILED','errors':errors}

def main():
    shortlist=json.loads((OUT/'measurement_shortlist.json').read_text())
    seen=set();order=[]
    for e in shortlist['entries']:
        if e['model_id'] not in seen:seen.add(e['model_id']);order.append(e)
    record={'at_utc':now(),'scope':'Module13B pinned download; one copy per unique checkpoint',
            'order':[e['model_id'] for e in order],'models':[],'completed':False}
    write(OUT/'download_manifest.json',record)
    for entry in order:
        repo=entry['model_id'];revision=entry['revision'];directory=MODELS/repo.replace('/','--')
        api=json.loads((OUT/'sources'/(repo.replace('/','--')+'.api.json')).read_text())
        assert api['sha']==revision
        directory.mkdir(parents=True,exist_ok=True)
        items=[x for x in api['siblings'] if x['rfilename'] in entry['weight_files'] or
               ('/' not in x['rfilename'] and x['rfilename'].endswith(('.json','.txt','.model','.yaml','.jinja')) and not x['rfilename'].startswith('eval_') and 'index.' not in x['rfilename'])]
        print('START',repo,revision,sum(x['size'] for x in items),flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            files=list(ex.map(lambda i:fetch(repo,revision,i,directory),items))
        ok=all(f.get('status')!='DOWNLOAD_FAILED' for f in files)
        item={'model_id':repo,'frozen_revision':revision,'resolved_revision':api['sha'],
              'task_paths':[e['task'] for e in shortlist['entries'] if e['model_id']==repo],
              'local_directory':str(directory),'files':files,'success':ok,
              'download_bytes_observable':sum(f.get('bytes',0) for f in files),'at_utc':now()}
        record['models'].append(item);write(OUT/'download_manifest.json',record)
        print('FINISH',repo,'SUCCESS' if ok else 'FAILED',flush=True)
    record['completed']=True;record['all_success']=all(x['success'] for x in record['models']);write(OUT/'download_manifest.json',record)

if __name__=='__main__':main()
