"""Small x86_64 ptrace tracer for own fresh CUDA diagnostic child.

Capture failing filesystem syscalls and their paths, not arbitrary memory.
Tracing perturbs timing; it is used only for initialization root cause.
"""
import os,sys,ctypes,signal,errno,time,json
from pathlib import Path
from m32f.common import *
class Regs(ctypes.Structure):
 _fields_=[(n,ctypes.c_ulonglong) for n in 'r15 r14 r13 r12 rbp rbx r11 r10 r9 r8 rax rcx rdx rsi rdi orig_rax rip cs eflags rsp ss fs_base gs_base ds es fs gs'.split()]
libc=ctypes.CDLL(None,use_errno=True)
libc.ptrace.restype=ctypes.c_long
layer=sys.argv[1];py=sys.argv[2]
os.environ.update(read(ROOT/'results/module32e_infrastructure_repair/frozen_validation_manifest.json')['plan']['environment_overrides'])
pid=os.fork()
if pid==0:
 if libc.ptrace(0,0,0,0)<0:os._exit(90)
 os.kill(os.getpid(),signal.SIGSTOP)
 fd=os.open(OUT/f'trace_{layer}_child.log',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600);os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
 os.execv(py,[py,str(OUT/'minimal_cuda.py'),layer])
os.waitpid(pid,0)
if libc.ptrace(0x4200,pid,0,1)<0:raise OSError(ctypes.get_errno(),'ptrace options failed')
entry=True;call=None;rows=[];exit_code=None
paths={2:(0,),21:(0,),76:(0,),90:(0,),92:(0,),94:(0,),132:(0,),188:(0,),189:(0,),197:(0,),198:(0,),257:(1,),260:(1,),261:(1,),268:(1,),280:(1,)}
def string(address):
 try:
  fd=os.open(f'/proc/{pid}/mem',os.O_RDONLY)
  try:return os.pread(fd,2048,address).split(b'\0',1)[0].decode(errors='replace')
  finally:os.close(fd)
 except OSError as exc:return repr(exc)
while True:
 if libc.ptrace(24,pid,0,0)<0:raise OSError(ctypes.get_errno(),'ptrace resume')
 _,status=os.waitpid(pid,0)
 if os.WIFEXITED(status):exit_code=os.WEXITSTATUS(status);break
 if os.WIFSIGNALED(status):exit_code=-os.WTERMSIG(status);break
 sig=os.WSTOPSIG(status)
 if sig==signal.SIGTRAP|0x80:
  r=Regs()
  if libc.ptrace(12,pid,0,ctypes.byref(r))<0:raise OSError(ctypes.get_errno(),'ptrace regs')
  if entry:
   args=[r.rdi,r.rsi,r.rdx,r.r10,r.r8,r.r9];nr=r.orig_rax
   call=dict(syscall=nr,args=args,paths=[string(args[j]) for j in paths.get(nr,())])
  else:
   ret=ctypes.c_longlong(r.rax).value
   if ret in (-errno.EACCES,-errno.EPERM) and call:rows.append(dict(call,return_value=ret,perf_ns=time.perf_counter_ns()))
  entry=not entry
 elif sig not in (signal.SIGTRAP,signal.SIGSTOP):
  # Pass unexpected signal and retain process failure evidence.
  libc.ptrace(24,pid,0,sig);_,status=os.waitpid(pid,0)
  if os.WIFEXITED(status):exit_code=os.WEXITSTATUS(status);break
save(f'syscall_trace_{layer}.json',dict(child_pid=pid,exit_code=exit_code,failed_syscalls=rows,scope='Own child main thread ptrace; diagnostic only, no service timing claim'))
print('TRACE_COMPLETE',layer,exit_code,rows,flush=True)
