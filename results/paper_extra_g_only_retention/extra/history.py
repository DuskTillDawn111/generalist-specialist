"""Full SHA256 verification, including historical results, code and weights."""
from .common import *
import sys,stat
def inventory():
    files={}
    excluded={'envs','datasets','pip_cache','conda_pkgs','hf_cache','.module13b_tmp','.vscode'}
    for folder in ROOT.iterdir():
        if folder.name in excluded:continue
        if folder.is_file():paths=[folder]
        elif folder.is_dir():paths=folder.rglob('*')
        else:continue
        for p in paths:
            if OUT==p or OUT in p.parents or not (p.is_file() or p.is_symlink()):continue
            if p.is_symlink():files[str(p)]=dict(symlink=os.readlink(p))
            else:
                st=p.stat();files[str(p)]={**ref(p), 'mtime_ns':st.st_mtime_ns}
    return files
def main():
    mode=sys.argv[1];start=time.perf_counter()
    if mode=='before':
        assert not (OUT/'history_sha256_before.json').exists()
        data=inventory();save('history_sha256_before.json',data)
        print('HASH_BEFORE',len(data),'seconds',time.perf_counter()-start,flush=True)
    else:
        before=read(OUT/'history_sha256_before.json');changed=[]
        for i,(p,v) in enumerate(before.items()):
            pp=Path(p)
            if 'symlink' in v:ok=pp.is_symlink() and os.readlink(pp)==v['symlink']
            else:ok=pp.is_file() and ref(pp)['sha256']==v['sha256'] and pp.stat().st_mtime_ns==v['mtime_ns']
            if not ok:changed.append(p)
        save('history_integrity_after.json',dict(passed=not changed,files=len(before),changed=changed,seconds=time.perf_counter()-start,full_SHA256=True,model_weights_included=True))
        print('HASH_AFTER',len(before),'changed',len(changed),flush=True)
        if changed:raise SystemExit(1)
if __name__=='__main__':main()
