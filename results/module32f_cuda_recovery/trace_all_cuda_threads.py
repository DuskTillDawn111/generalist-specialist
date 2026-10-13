"""Trace own child+threads with PTRACE_GET_SYSCALL_INFO, CUDA marker bounds."""
import ctypes,os,signal,sys,errno
from m32f.common import *
class Info(ctypes.Structure):
 _fields_=[('op',ctypes.c_ubyte),('pad',ctypes.c_ubyte*3),('arch',ctypes.c_uint32),('ip',ctypes.c_uint64),('sp',ctypes.c_uint64),('data',ctypes.c_uint64*9)]
libc=ctypes.CDLL(None,use_errno=True);libc.ptrace.restype=ctypes.c_long
layer,py=sys.argv[1:3]
os.environ.update(read(ROOT/'results/module32e_infrastructure_repair/frozen_validation_manifest.json')['plan']['environment_overrides'])
pid=os.fork()
if pid==0:
 if libc.ptrace(0,0,0,0)<0:os._exit(90)
 os.kill(os.getpid(),signal.SIGSTOP)
 fd=os.open(OUT/f'trace_all_{layer}_child.log',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600);os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
 os.execv(py,[py,str(OUT/'minimal_cuda.py'),layer])
os.waitpid(pid,0);options=1|8|0x100000
assert libc.ptrace(0x4200,pid,0,options)==0
active={pid};calls={};rows=[];comm=[];cuda=False;exits={};count=0
pathidx={2:(0,),21:(0,),76:(0,),90:(0,),92:(0,),94:(0,),132:(0,),188:(0,),189:(0,),197:(0,),198:(0,),257:(1,),260:(1,),261:(1,),268:(1,),280:(1,),437:(1,)}
def string(tid,address):
 try:
  fd=os.open(f'/proc/{tid}/mem',os.O_RDONLY)
  try:return os.pread(fd,2048,address).split(b'\0',1)[0].decode(errors='replace')
  finally:os.close(fd)
 except OSError as e:return repr(e)
def resume(tid,sig=0):
 if libc.ptrace(24,tid,0,sig)<0 and ctypes.get_errno()!=errno.ESRCH:raise OSError(ctypes.get_errno(),'ptrace resume')
resume(pid)
while active:
 tid,status=os.waitpid(-1,0x40000000)
 if os.WIFEXITED(status) or os.WIFSIGNALED(status):
  exits[tid]=os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status);active.discard(tid);continue
 active.add(tid);sig=os.WSTOPSIG(status);event=status>>16
 if event==3:
  child=ctypes.c_ulonglong();assert libc.ptrace(0x4201,tid,0,ctypes.byref(child))==0;active.add(child.value)
 elif sig==(signal.SIGTRAP|0x80):
  info=Info();assert libc.ptrace(0x420e,tid,ctypes.sizeof(info),ctypes.byref(info))>0
  if info.op==1:
   nr=info.data[0];args=list(info.data[1:7]);count+=1
   if nr==1 and args[0] in (1,2) and 'MIN_CUDA_INIT_BEGIN' in string(tid,args[1]):cuda=True
   call=dict(tid=tid,syscall=nr,args=args,cuda_phase=cuda,paths=[string(tid,args[j]) for j in pathidx.get(nr,())])
   if nr in (1,16,91,93,77,138,221,292,32,33):
    try:call['fd_target']=os.readlink(f'/proc/{tid}/fd/{args[0]}')
    except OSError:pass
   calls[tid]=call
  elif info.op==2:
   ret=ctypes.c_longlong(info.data[0]).value;call=calls.get(tid)
   if call and ret<0 and (cuda or ret in (-errno.EACCES,-errno.EPERM)):
    rows.append(dict(call,return_value=ret,perf_ns=time.perf_counter_ns()))
   if call and cuda and (any('/task/' in p and p.endswith('/comm') for p in call['paths']) or '/task/' in call.get('fd_target','') and call.get('fd_target','').endswith('/comm')):
    comm.append(dict(call,return_value=ret,perf_ns=time.perf_counter_ns()))
 if sig not in (signal.SIGTRAP,signal.SIGTRAP|0x80,signal.SIGSTOP):resume(tid,sig)
 else:resume(tid)
save(f'syscall_trace_all_{layer}.json',dict(pid=pid,exit_codes=exits,failed_syscalls=rows,proc_comm_syscalls=comm,total_syscall_entries=count,
 traced_all_clone_threads=True,scope='Own fresh minimal CUDA diagnostic; ptrace perturbs timing; not routing/service benchmark'))
print('TRACE_ALL_COMPLETE',layer,exits.get(pid),len(rows),rows[-30:],flush=True)
