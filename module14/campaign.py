"""Reuse verified standalone13B orchestration, write only Module14 evidence."""
import argparse,json,os,subprocess,threading,time
from module14.common import *
from module13b import campaign as original

PROTO=read(OUT/'measurement_protocol_freeze.json');ENTRIES=PROTO['entries'];RAW.mkdir(exist_ok=True)
original.OUT=OUT;original.RAW=RAW;original.SHORT=ENTRIES;original.PROTO=PROTO
original.ENV.update(PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
def paddle_loader(argv):
 # Match the unchanged original run_static_hybrid.Worker.start() launcher.
 # Paddle's CUDA13 wheel dependencies already exist in its own environment.
 lib='/root/miniconda3/envs/ocr_env/lib/python3.11/site-packages/nvidia/cu13/lib'
 previous=os.environ.get('LD_LIBRARY_PATH','')
 return ['/usr/bin/env','LD_LIBRARY_PATH='+lib+(':'+previous if previous else ''),*argv]
def command(e,cpu=False):
 if e.get('control'):
  worker='joint_deploy/resnet_worker.py' if e['task']=='image' else 'joint_deploy/ocr_worker.py';python=ROOT/'envs/specialist_env/bin/python' if e['task']=='image' else Path('/root/miniconda3/envs/ocr_env/bin/python')
  argv=[str(python),str(ROOT/worker)]
  return paddle_loader(argv) if e['task']=='ocr' else argv
 python=Path('/root/miniconda3/envs/ocr_env/bin/python') if e['backend']=='paddle' else ROOT/'envs/module13b_transformers/bin/python'
 argv=[str(python),str(ROOT/'module14/worker.py'),'--model-id',e['model_id']]+(['--cpu-preflight'] if cpu else [])
 return paddle_loader(argv) if e['backend']=='paddle' else argv
original.command=command

def cpu():
 existing=OUT/'cpu_preflight_results.json'
 if existing.exists():
  i=1
  while (OUT/f'cpu_preflight_attempt{i}.json').exists():i+=1
  (OUT/f'cpu_preflight_attempt{i}.json').write_bytes(existing.read_bytes())
 records=[]
 for e in ENTRIES:
  deadline=time.monotonic()+1800;announced=False
  while not any(m['model_id']==e['model_id'] for m in read(OUT/'download_manifest.json')['models']):
   if not announced:print('WAIT_PINNED_DOWNLOAD_BEFORE_CPU_PREFLIGHT',e['model_id'],flush=True);announced=True
   assert time.monotonic()<deadline,'Pinned download not completed'
   threading.Event().wait(2)
  print('CPU_PREFLIGHT',e['model_id'],flush=True);p=RAW/(key(e)+'__cpu.log');i=1
  while p.exists():p=RAW/(key(e)+f'__cpu_attempt{i}.log');i+=1
  r=subprocess.run(command(e,True),cwd=ROOT,env={**original.ENV,'CUDA_VISIBLE_DEVICES':''},text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600);p.write_text(r.stdout);parts=[x.split('|',1)[1] for x in r.stdout.splitlines() if x.startswith('CPU_PREFLIGHT|')];records.append({'task':e['task'],'model_id':e['model_id'],'revision':e['revision'],'success':r.returncode==0 and len(parts)==1,'exit_code':r.returncode,'metadata':json.loads(parts[0]) if parts else None,'log':str(p)});save('cpu_preflight_results.json',{'at_utc':now(),'entries':records,'completed':False});print('CPU_RESULT',e['model_id'],records[-1]['success'],flush=True)
 save('cpu_preflight_results.json',{'at_utc':now(),'entries':records,'completed':True,'all_success':all(r['success'] for r in records)})
def gpu():
 assert (OUT/'execution_freeze.json').exists(),'Before performance lock every adapter'
 freeze=read(OUT/'execution_freeze.json')
 for r in freeze['implementation']+read(OUT/'discovery_freeze_manifest.json')['files']:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 pre=read(OUT/'cpu_preflight_results.json');ok={r['model_id']:r['success'] for r in pre['entries']};original.nv.nvmlInit()
 try:
  for e in ENTRIES:
   if ok[e['model_id']]:original.run(e,'quality')
  for rep in [1,2,3]:
   for task,controlid in [('image','resnet50'),('ocr','paddleocr')]:
    original.run({'task':task,'model_id':controlid,'control':True},'latency',rep)
    for e in ENTRIES:
     if e['task']==task and ok[e['model_id']]:original.run(e,'latency',rep)
 finally:original.nv.nvmlShutdown()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);a=p.parse_args();cpu() if a.mode=='cpu' else gpu()
