"""Targeted CPU proof: delayed old snapshot cannot consume new-phase completions."""
import sys,threading
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m32f.isolation import install
install()
from m32f.common import *
from router_optimized import DeploymentRuntime
from m32f.monitor import IncrementalMonitor as RepairedMonitor
from m32f.runtime import configure
configure()


class GateLock:
 def __init__(self):self.raw=threading.RLock();self.captured=threading.Event();self.resume=threading.Event();self.gate=True
 def __enter__(self):
  if threading.current_thread().name=='delayed_snapshot' and self.gate:
   self.gate=False;self.captured.set();assert self.resume.wait(2)
  self.raw.acquire();return self
 def __exit__(self,*args):self.raw.release()

def main():
 rt=DeploymentRuntime(SimpleNamespace());rt.lock=GateLock();m=RepairedMonitor(rt,dict(run_id='CPU_PHASE_TRANSITION'),1,sensor=lambda:{})
 a=dict(request_id='old:0',status='completed',model=S,task='asr');b=dict(request_id='new:0',status='planned',model=S,task='asr')
 m.set_active(dict(label='old',request_records=[a]));result=[]
 t=threading.Thread(target=lambda:result.append(m.coherent_records()),name='delayed_snapshot');t.start();assert rt.lock.captured.wait(2)
 m.set_active(dict(label='new',request_records=[b]))
 with rt.lock:b['status']='completed';rt.refresh(b)
 rt.lock.resume.set();t.join(2);assert not t.is_alive()
 assert result[0][0]['label']=='old' and result[0][1][0]['request_id']=='old:0'
 assert len(rt.terminal_changes)==1,'Old snapshot consumed new-phase terminal event'
 header,rows=m.coherent_records();assert header['label']=='new' and rows[0]['status']=='completed' and not rt.terminal_changes
 save('CPU_phase_transition_acceptance.json',dict(passed=True,old_delayed_snapshot_coherent=True,new_phase_completion_not_lost=True,thread_exit=True,GPU_used=False,source=ref(OUT/'m32f/monitor.py')))
 print('PHASE_TRANSITION_PASS')
if __name__=='__main__':main()
