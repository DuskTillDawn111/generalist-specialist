"""Process/descendant write confinement; reads unrestricted, output path explicit.

Linux5.15 Landlock ABI1 protects write-open/create/delete/link/rename. ABI1 lacks
truncate/metadata mediation, so a seccomp filter denies those native syscalls;
Python audit hooks additionally resolve mutation targets including symlink/fd
escapes. No chmod/remount of scientific inputs. Restrictions precede imports,
threads and child launch; do not grant any historical directory write access.
"""
import ctypes,errno,os,sys,json,time
from pathlib import Path

ROOT=Path('/root/autodl-tmp/generalist-specialist/results/module32g_final_validation').resolve()
VIOLATIONS=[]
class Ruleset(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
class PathRule(ctypes.Structure):
 _pack_=1
 _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
class Filter(ctypes.Structure):_fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]
class Program(ctypes.Structure):_fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(Filter))]
INSTALLED=False

def writable(path):
 if isinstance(path,int):
  try:path=os.readlink('/proc/self/fd/'+str(path))
  except OSError:return False
  if path.startswith(('pipe:','socket:')):return True
 p=Path(os.fsdecode(path)).resolve()
 # Native CUDA sets sibling thread names through procfs comm. This is process
 # control, not a scratch/output directory. Python gets only this process's
 # exact task/comm files, even in the supervisor's inheritable procfs envelope.
 parts=p.parts
 proc_comm=len(parts)==6 and parts[:3]==('/', 'proc',str(os.getpid())) and parts[3]=='task' and parts[4].isdigit() and parts[5]=='comm'
 return p==ROOT or ROOT in p.parents or p==Path('/dev') or Path('/dev') in p.parents or proc_comm

def audit(event,args):
 targets=[]
 def target(path,dir_fd=None):
  if isinstance(path,int):return path
  if dir_fd is not None and dir_fd>=0 and not os.path.isabs(path):
   return os.path.join(os.readlink('/proc/self/fd/'+str(dir_fd)),os.fsdecode(path))
  return path
 if event=='open':
  path,mode,flags=args
  if (flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) or mode and any(c in mode for c in 'wax+'):targets=[path]
 elif event in ('os.remove','os.rmdir'):targets=[target(args[0],args[1])]
 elif event in ('os.mkdir','os.chmod'):targets=[target(args[0],args[2])]
 elif event in ('os.chown','os.utime'):targets=[target(args[0],args[3])]
 elif event in ('os.truncate','os.setxattr','os.removexattr'):targets=[args[0]]
 elif event in ('os.rename','os.link'):targets=[target(args[0],args[2]),target(args[1],args[3])]
 elif event=='os.symlink':targets=[target(args[1],args[2])]
 for target in targets:
  if not writable(target):
   VIOLATIONS.append(dict(event=event,target=os.fsdecode(target) if not isinstance(target,int) else target,perf_ns=time.perf_counter_ns()))
   raise PermissionError(errno.EACCES,'Module32G write confinement: scientific input is read-only',str(target))

def install(layer="repaired"):
 assert layer in ("audit","landlock","seccomp","full","repaired","supervisor")
 global INSTALLED
 if INSTALLED:return
 ROOT.mkdir(parents=True,exist_ok=True)
 scratch=ROOT/'work';scratch.mkdir(exist_ok=True)
 for d in ['tmp','cache','cache/cuda','cache/torch','cache/hf','cache/matplotlib','control_guards']:(scratch/d).mkdir(parents=True,exist_ok=True)
 os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(scratch/'tmp'),TMP=str(scratch/'tmp'),TEMP=str(scratch/'tmp'),CUDA_CACHE_PATH=str(scratch/'cache/cuda'),TORCH_HOME=str(scratch/'cache/torch'),HF_HOME=str(scratch/'cache/hf'),MPLCONFIGDIR=str(scratch/'cache/matplotlib'),XDG_CACHE_HOME=str(scratch/'cache'),NUMBA_CACHE_DIR=str(scratch/'cache/numba'),TRITON_CACHE_DIR=str(scratch/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(scratch/'cache/torchinductor'))
 libc=ctypes.CDLL(None,use_errno=True)
 abi=libc.syscall(444,0,0,1)
 if abi<1:raise OSError(ctypes.get_errno(),'Landlock unavailable; no tests/GPU authorized')
 if libc.prctl(38,1,0,0,0):raise OSError(ctypes.get_errno(),'NO_NEW_PRIVS failed')
 mask=sum(1<<i for i in (1,4,5,6,7,8,9,10,11,12));attr=Ruleset(mask)
 if layer in ("landlock","full","repaired","supervisor"):
  rulefd=libc.syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0)
  if rulefd<0:raise OSError(ctypes.get_errno(),'Landlock create failed')
  paths=[(ROOT,mask),(Path('/dev'),1<<1),(Path('/dev/shm'),mask)]
  if layer=='repaired':paths.append((Path('/proc/self/task'),1<<1))
  # An inherited parent domain cannot later be broadened for a new worker PID.
  # Supervisor envelopes procfs write-existing for descendants; each Python
  # worker installs a second narrower domain for its own /proc/self/task before
  # importing the unchanged backend. Intersection keeps history fully read-only.
  if layer=='supervisor':paths.append((Path('/proc'),1<<1))
  for path,bits in paths:
   fd=os.open(path,os.O_PATH|os.O_CLOEXEC);r=PathRule(bits,fd)
   try:
    if libc.syscall(445,rulefd,1,ctypes.byref(r),0):raise OSError(ctypes.get_errno(),'Landlock path rule failed')
   finally:os.close(fd)
  if libc.syscall(446,rulefd,0):raise OSError(ctypes.get_errno(),'Landlock restrict failed')
  os.close(rulefd)
 # Deny ABI1 holes even from non-Python libraries/descendants. ftruncate may
 # operate only on write FDs, which Landlock prevents opening outside the root.
 denied=[76,90,91,92,93,94,132,155,165,166,188,189,190,197,198,199,235,260,261,268,280,452]
 if layer in ("seccomp","full","repaired","supervisor"):
  code=[Filter(0x20,0,0,4),Filter(0x15,1,0,0xc000003e),Filter(0x06,0,0,0x00050001),Filter(0x20,0,0,0),Filter(0x35,0,1,0x40000000),Filter(0x06,0,0,0x00050001)]
  for nr in denied:code.extend([Filter(0x15,0,1,nr),Filter(0x06,0,0,0x00050001)])
  code.append(Filter(0x06,0,0,0x7fff0000));arr=(Filter*len(code))(*code);prog=Program(len(code),arr)
  if libc.syscall(317,1,0,ctypes.byref(prog)):raise OSError(ctypes.get_errno(),'Seccomp filter failed')
 sys.addaudithook(audit);INSTALLED=True
 marker=scratch/'control_guards'/f'{os.getpid()}.json'
 marker.write_text(json.dumps(dict(pid=os.getpid(),landlock_ABI=abi,write_root=str(ROOT),reads_unrestricted=True,diagnostic_layer=layer,seccomp_legacy_mutations_denied=denied,python_audit=True,dev_IPC_exception='/dev write-existing, /dev/shm IPC',procfs_rule=str(Path('/proc/self/task').resolve()) if layer=='repaired' else '/proc write-existing supervisor envelope; worker must narrow' if layer=='supervisor' else None,no_history_permission_changes=True,perf_ns=time.perf_counter_ns()),indent=2)+'\n')

