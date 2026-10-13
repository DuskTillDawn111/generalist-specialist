"""Supervisor inheritance + worker restriction + native non-output denial."""
import sys,ctypes,subprocess,os,json
from m32f.isolation import install,ROOT,VIOLATIONS
install('supervisor')
from m32f.common import save
from pathlib import Path
py='/root/autodl-tmp/generalist-specialist/envs/module14b_str/bin/python'
for i in range(2):
 result=subprocess.run([py,str(ROOT/'minimal_cuda.py'),'repaired'],text=True,capture_output=True,timeout=90)
 parsed=json.loads(result.stdout.strip().splitlines()[-1]);assert result.returncode==0 and parsed['passed'] and not parsed['unexpected_python_writes']
 save(f'supervisor_worker_cuda/{i}.json',dict(stdout=result.stdout,stderr=result.stderr,return_code=result.returncode,result=parsed))
code='''import ctypes,os,json
from m32f.isolation import install,ROOT
install('repaired')
libc=ctypes.CDLL(None,use_errno=True);rows=[]
for p in ['/proc/self/oom_score_adj','/root/autodl-tmp/generalist-specialist/results/module32e_infrastructure_repair/module32e_gate.json']:
 fd=libc.open(os.fsencode(p),os.O_WRONLY)
 rows.append(dict(path=p,fd=fd,errno=ctypes.get_errno()))
 assert fd<0,rows
path=f'/proc/self/task/{os.getpid()}/comm';fd=libc.open(os.fsencode(path),os.O_WRONLY);assert fd>=0;os.close(fd)
print(json.dumps(dict(passed=True,blocked=rows,own_comm_writable=True)))
'''
result=subprocess.run([py,'-c',code],text=True,capture_output=True,env=dict(os.environ,PYTHONPATH=str(ROOT)),timeout=30)
assert result.returncode==0,(result.stdout,result.stderr)
save('proc_scope_protection.json',dict(passed=True,worker_native_test=json.loads(result.stdout.strip().splitlines()[-1]),
 inherited_supervisor_to_worker_CUDA_repetitions=2,policy_unchanged=True,unexpected_supervisor_python_writes=VIOLATIONS))
assert not VIOLATIONS
print('PROC_SCOPE_PASS',flush=True)
