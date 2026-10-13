"""Actual history targets, no successful mutation; native descendants included."""
import sys,os,subprocess,ctypes,hashlib
from pathlib import Path
from m32e.isolation import install,VIOLATIONS
install()
from m32e.common import *
from m32e.monitor import IncrementalMonitor
from types import SimpleNamespace
targets=[ROOT/'results/module30_replica_scaling/live_progress.json',ROOT/'results/module32d_router_repair/module32d_gate.json']
# Locate Module30's authoritative namespace instead of guessing its spelling.
import module30.telemetry as legacy
targets[0]=legacy.OUT/'live_progress.json'
before=[ref(p) for p in targets];rows=[]
def denied(name,fn):
 try:fn();rows.append(dict(test=name,blocked=False))
 except (PermissionError,OSError,AssertionError,subprocess.CalledProcessError) as e:rows.append(dict(test=name,blocked=True,error=repr(e)))
for p in targets:
 denied('python_append:'+str(p),lambda:p.open('a'))
 denied('python_truncate:'+str(p),lambda:os.truncate(p,p.stat().st_size))
 denied('python_metadata:'+str(p),lambda:os.utime(p,None))
 denied('python_delete:'+str(p),lambda:os.unlink(p))
 denied('python_rename:'+str(p),lambda:os.rename(p,OUT/'work/disallowed_move'))
 denied('native_child_write:'+str(p),lambda:subprocess.run(['bash','-c','exec 3>> "$1"','probe',str(p)],check=True,capture_output=True))
 # A read-only FD cannot be a loophole for ftruncate; native pathname truncate
 # is denied by seccomp even though ABI1 Landlock alone cannot mediate it.
 libc=ctypes.CDLL(None,use_errno=True)
 def native_truncate():
  rc=libc.truncate(os.fsencode(p),ctypes.c_longlong(p.stat().st_size))
  if rc:raise OSError(ctypes.get_errno(),'native truncate blocked')
 denied('native_path_truncate:'+str(p),native_truncate)
link=OUT/'work/history_symlink'
if not link.is_symlink():link.symlink_to(targets[0])
denied('symlink_escape',lambda:link.open('a'))
denied('legacy_output_constructor',lambda:IncrementalMonitor(SimpleNamespace(),dict(run_id='FORBIDDEN'),1,output_dir=legacy.OUT))
denied('legacy_save_callback',lambda:legacy.save('live_progress.json',{'should_never_write':True}))
positive=OUT/'work/positive_output';positive.write_text('isolated\n');assert positive.read_text()=='isolated\n'
ownfd=os.open(OUT/'work',os.O_RDONLY|os.O_DIRECTORY)
try:
 positive2=OUT/'work/dirfd_positive';positive2.write_text('valid relative directory FD\n')
 os.unlink('dirfd_positive',dir_fd=ownfd);assert not positive2.exists()
finally:os.close(ownfd)
histfd=os.open(targets[0].parent,os.O_RDONLY|os.O_DIRECTORY)
try:denied('dirfd_history_escape',lambda:os.unlink(targets[0].name,dir_fd=histfd))
finally:os.close(histfd)
after=[ref(p) for p in targets]
save('history_write_protection_tests.json',dict(passed=all(r['blocked'] for r in rows) and before==after,
 tests=rows,before=before,after=after,expected_python_denials=VIOLATIONS,positive_output=True,
 native_descendant_inherits_confinement=True,scope='All ordinary filesystem mutations denied outside Module32E; device/IPC exception only. Full historical SHA manifests additionally detect changes by unrelated processes.'))
assert all(r['blocked'] for r in rows) and before==after,rows
print('HISTORY_PROTECTION_PASS',len(rows),flush=True)
