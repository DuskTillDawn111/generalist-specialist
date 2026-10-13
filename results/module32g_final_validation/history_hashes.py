"""Content hashes of every historical Module result plus frozen scientific inputs."""
import sys,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32g.common import *
def capture(label):
 f=read(ROOT/'results/module32d_router_repair/frozen_validation_manifest.json');paths=set()
 for parent in (ROOT/'results').glob('module*'):
  if parent==OUT:continue
  if parent.is_file():paths.add(parent)
  else:paths.update(p for p in parent.rglob('*') if p.is_file())
 for row in f['sources']+f['frozen_inputs']+f['checkpoints']+f['request_input_files']:paths.add(Path(row['path']))
 unique=sorted(paths);print('HASH_START',label,len(unique),sum(p.stat().st_size for p in unique),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(ref,unique))
 rows=[dict(r,mtime_ns=Path(r['path']).stat().st_mtime_ns,is_symlink=Path(r['path']).is_symlink(),symlink_target=os.readlink(r['path']) if Path(r['path']).is_symlink() else None) for r in rows]
 save('history_hashes/'+label+'.json',dict(label=label,at_unix_ns=time.time_ns(),files=rows,scientific_history_baseline='Current state includes explicitly documented irreversible32D progress reconstruction; no claim of pre32D byte-exact restoration',prior_incident=ref(OUT/'prior_incident_record.json')))
 print('HASH_DONE',label,len(rows),flush=True);return rows
def compare(a,b):
 A=read(OUT/'history_hashes'/f'{a}.json');B=read(OUT/'history_hashes'/f'{b}.json');aa={r['path']:r for r in A['files']};bb={r['path']:r for r in B['files']}
 changed=[p for p in aa.keys()|bb.keys() if p not in aa or p not in bb or any(aa[p][k]!=bb[p][k] for k in ['sha256','bytes','mtime_ns','is_symlink','symlink_target'])]
 result=dict(passed=not changed,before=ref(OUT/'history_hashes'/f'{a}.json'),after=ref(OUT/'history_hashes'/f'{b}.json'),files=len(aa),changed=changed,old32D_unrestorable_difference_not_erased=True)
 save('history_hashes/'+a+'_vs_'+b+'.json',result);print('HASH_COMPARE',result['passed'],len(changed),flush=True);return result
if __name__=='__main__':
 from m32g.isolation import install
 install()
 if sys.argv[1]=='compare':compare(sys.argv[2],sys.argv[3])
 else:capture(sys.argv[1])
