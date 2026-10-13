from .common import *
import traceback,fcntl

def verify(f):
    for r in f['native_sources']+f['sources']+f['inputs']:
        assert ref(r['path'])['sha256']==r['sha256'],r['path']
def main():
    from .isolation import install
    install('supervisor')
    lock=(OUT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from .gpu_native import run
    from .gpu_assess import assess
    import sys
    f=read(OUT/(sys.argv[1] if len(sys.argv)>1 else 'frozen_gpu_manifest.json'));observations=[];failure=None
    assert not list((OUT/'raw_runs').glob('*.json'))
    try:
        for slot in f['slots']:
            verify(f);save('campaign_status.json',dict(status='RUNNING',run_id=slot['run_id'],completed=len(observations),expected=len(f['slots']),pid=os.getpid()))
            print('GPU_START',slot['run_id'],flush=True)
            raw=run(slot,f);o=assess(raw);save('gpu_observations/'+slot['run_id']+'.json',o);observations.append(o)
            save('gpu_observations.json',observations)
            print('GPU_FINISH',slot['run_id'],'technical',o['technical_valid'],'stable',o['stable'],'U',o['U'],flush=True)
            if not o['technical_valid']:raise RuntimeError(('TECHNICAL_INVALID_NO_RETRY',slot['run_id'],o['checks'],raw['errors']))
    except BaseException as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc());save('gpu_campaign_failure.json',failure);print('GPU_STOP',repr(exc),flush=True)
    finally:save('campaign_status.json',dict(status='STOPPED_ON_FAILURE' if failure else 'COMPLETED',completed=len(observations),expected=len(f['slots']),failure=failure))
    if failure:raise SystemExit(1)
if __name__=='__main__':main()
