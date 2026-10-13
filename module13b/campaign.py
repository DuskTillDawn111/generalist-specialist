"""Sequential real standalone campaign; preserves every raw attempt."""
import argparse, datetime, hashlib, json, os, queue, statistics, subprocess, threading, time
from pathlib import Path
import pynvml as nv
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/module13_real_specialists';RAW=OUT/'raw13b'
RAW.mkdir(exist_ok=True)
ENV={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
SHORT=json.loads((OUT/'measurement_shortlist.json').read_text())['entries']
PROTO=json.loads((OUT/'measurement_protocol_freeze.json').read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def key(e):return e['task']+'__'+e['model_id'].replace('/','--')
def python(e):return ROOT/('envs/module13b_speechbrain/bin/python' if e['model_id'].startswith('speechbrain/') else 'envs/module13b_transformers/bin/python')
def command(e,cpu=False):
    if e.get('control'):
        worker={'asr':'candidate_pool_audit/worker_distil_whisper.py','lid':'joint_deploy/whisper_worker.py','text':'joint_deploy/distilbert_worker.py'}[e['task']]
        return [str(ROOT/'envs/specialist_env/bin/python'),str(ROOT/worker)]
    return [str(python(e)),str(ROOT/'module13b/worker.py'),'--model-id',e['model_id'],'--task',e['task'],'--directory',str(ROOT/'models/module13b'/e['model_id'].replace('/','--'))]+(['--cpu-preflight'] if cpu else [])
def cpu_preflight():
    records=[]
    previous=OUT/'cpu_preflight_results.json'
    if previous.exists():
        index=1
        while (OUT/f'cpu_preflight_attempt{index}.json').exists():index+=1
        (OUT/f'cpu_preflight_attempt{index}.json').write_bytes(previous.read_bytes())
    for e in SHORT:
        # CPU loading may overlap transport of later checkpoints. Never try
        # to load an incomplete snapshot; this is outside every model timer.
        deadline=time.monotonic()+1800;announced=False
        while True:
            manifest=json.loads((OUT/'download_manifest.json').read_text())
            downloaded=next((m for m in manifest['models'] if m['model_id']==e['model_id']),None)
            if downloaded is not None:break
            if not announced:print('WAIT_PINNED_DOWNLOAD_BEFORE_CPU_LOAD',e['model_id'],flush=True);announced=True
            assert time.monotonic()<deadline,'Pinned snapshot not ready'
            threading.Event().wait(2)
        log=RAW/(key(e)+'__cpu.log')
        attempt=1
        while log.exists():
            log=RAW/(key(e)+f'__cpu_attempt{attempt}.log');attempt+=1
        print('CPU_PREFLIGHT',key(e),flush=True)
        r=subprocess.run(command(e,True),cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':''},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
        log.write_text(r.stdout)
        lines=[x.split('|',1)[1] for x in r.stdout.splitlines() if x.startswith('CPU_PREFLIGHT|')]
        records.append({'task':e['task'],'model_id':e['model_id'],'revision':e['revision'],'exit_code':r.returncode,'success':r.returncode==0 and len(lines)==1,'metadata':json.loads(lines[0]) if lines else None,'log':str(log)})
        write(OUT/'cpu_preflight_results.json',{'at_utc':now(),'entries':records,'completed':False})
    write(OUT/'cpu_preflight_results.json',{'at_utc':now(),'entries':records,'completed':True,'all_success':all(r['success'] for r in records)})

def running(h):return nv.nvmlDeviceGetComputeRunningProcesses(h)
def rss(pid):
    try:
        for l in Path(f'/proc/{pid}/status').read_text().splitlines():
            if l.startswith('VmRSS:'):return int(l.split()[1])/1024
    except FileNotFoundError:pass
    return None
class Worker:
    def __init__(self,e,stem):
        self.e=e;self.stem=stem;self.q=queue.Queue();self.lines=(RAW/(stem+'.worker.log')).open('w')
        self.p=subprocess.Popen(command(e),cwd=ROOT,env=ENV,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        def reader():
            for line in self.p.stdout:
                self.lines.write(line);self.lines.flush()
                if '|' in line and line.split('|',1)[0] in ['READY','RESULT','ERROR','BYE','MEMORY_END']:self.q.put(line.strip())
            self.q.put('EXIT|'+str(self.p.poll()))
        self.thread=threading.Thread(target=reader,daemon=True);self.thread.start()
        try:self.ready=self.receive('READY',600)
        except Exception:
            self.p.kill();self.p.wait();self.thread.join(timeout=5);self.lines.close();raise
    def receive(self,kind,timeout=180):
        deadline=time.monotonic()+timeout
        while True:
            line=self.q.get(timeout=max(.001,deadline-time.monotonic()));tag,data=line.split('|',1)
            if tag in ['ERROR','EXIT']:raise RuntimeError(line)
            if tag==kind:return json.loads(data)
    def request(self,row,index):
        request={**row,'request_id':f'{self.stem}:{index}','arrival_index':index,'submit_ns':time.time_ns()}
        self.p.stdin.write(json.dumps(request)+'\n');self.p.stdin.flush()
        value=self.receive('RESULT',300);assert value['request_id']==request['request_id'];return value
    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write('{"op":"shutdown"}\n');self.p.stdin.flush()
            try:self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:self.p.kill();self.p.wait()
        self.thread.join(timeout=5);self.lines.close()
class Telemetry:
    def __init__(self,h,pid,stem):
        self.h=h;self.pid=pid;self.stop=threading.Event();self.phase='initial';self.rows=[];self.foreign=[]
        self.path=RAW/(stem+'.nvml.jsonl');self.f=self.path.open('w')
        def sample():
            while not self.stop.is_set():
                processes=running(h);foreign=[x.pid for x in processes if x.pid!=pid]
                if foreign:self.foreign.append(foreign)
                memory=next((x.usedGpuMemory/1048576 for x in processes if x.pid==pid),None)
                r={'time_ns':time.time_ns(),'phase':self.phase,'pid':pid,'process_memory_mib':memory,'host_rss_mib':rss(pid),'foreign_pids':foreign}
                self.rows.append(r);self.f.write(json.dumps(r)+'\n');self.f.flush()
                self.stop.wait(.01 if self.phase=='measurement' else .1)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
    def close(self):self.stop.set();self.thread.join();self.f.close()

def run(e,phase,rep=0):
    stem=key(e)+f'__{phase}__r{rep}';summary=RAW/(stem+'.summary.json')
    if summary.exists():
        d=json.loads(summary.read_text());print('REUSE_COMPLETED',stem,d['success'],flush=True);return d
    if (RAW/(stem+'.worker.log')).exists():raise RuntimeError('Raw attempt exists without completed summary: explicit infrastructure amendment required '+stem)
    h=nv.nvmlDeviceGetHandleByIndex(0);assert not running(h),'Foreign GPU model present before launch'
    rows=[json.loads(x) for x in (OUT/f'latency_samples_{e["task"]}.jsonl').read_text().splitlines()] if phase=='latency' else [{'task_type':e['task'],'pool_id':e['task']+':'+s['id'],'sample':s} for s in [json.loads(x) for x in Path(PROTO['datasets'][e['task']]['quality_manifest']['path']).read_text().splitlines()]]
    n=30 if phase=='latency' else 200;assert len(rows)==n
    print('START',stem,flush=True);record={'at_utc':now(),'task':e['task'],'model_id':e['model_id'],'control':e.get('control',False),'phase':phase,'repetition':rep,'success':False,'request_count':0,'raw_path':str(RAW/(stem+'.results.jsonl'))}
    worker=None;tele=None
    try:
        worker=Worker(e,stem);record['ready']=worker.ready
        tele=Telemetry(h,worker.p.pid,stem)
        if phase=='latency':
            worker.request(rows[0],-1);tele.phase='idle'
            # Untimed idle observation required by frozen memory protocol.
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:tele.stop.wait(.1)
            idle=[r['process_memory_mib'] for r in tele.rows if r['phase']=='idle' and r['process_memory_mib'] is not None]
            assert len(idle)>=45
            record['idle_model_vram_mib']=statistics.mean(idle);record['idle_samples']=len(idle)
        tele.phase='measurement'
        with (RAW/(stem+'.results.jsonl')).open('w') as f:
            for i,row in enumerate(rows):
                r=worker.request(row,i);f.write(json.dumps(r)+'\n');f.flush();record['request_count']=i+1
                if (i+1)%50==0:print('PROGRESS',stem,i+1,n,flush=True)
        tele.phase='post';record['success']=True
    except Exception as err:record['error']={'type':type(err).__name__,'reason':str(err)[:2000]}
    finally:
        if tele:
            tele.close();infer=[r for r in tele.rows if r['phase']=='measurement']
            record['peak_inference_vram_mib']=max((r['process_memory_mib'] for r in infer if r['process_memory_mib'] is not None),default=None)
            record['host_rss_peak_mib']=max((r['host_rss_mib'] for r in tele.rows if r['host_rss_mib'] is not None),default=None)
            record['foreign_process_samples']=tele.foreign
            if tele.foreign:record['success']=False;record['error']={'type':'ForeignGPUProcess','reason':str(tele.foreign)}
        if worker:worker.close()
        # Do not start another process until the complete model residency exits.
        if worker:
            for _ in range(100):
                if not any(x.pid==worker.p.pid for x in running(h)):break
                threading.Event().wait(.1)
            assert not running(h),'GPU residency did not release'
        record['completed_at_utc']=now();write(summary,record)
    print('FINISH',stem,record['success'],record.get('error'),flush=True);return record

def campaign():
    assert (OUT/'module13b_execution_freeze.json').exists(),'GPU results require prior execution lock'
    pre=json.loads((OUT/'cpu_preflight_results.json').read_text());ok={(x['task'],x['model_id']):x['success'] for x in pre['entries']}
    nv.nvmlInit()
    try:
        for e in SHORT:
            if ok[(e['task'],e['model_id'])]:run(e,'quality')
        # User 13B explicitly requires s0/M for every path; failed quality paths
        # stay descriptive only and never enter selection/Pareto admission.
        for rep in [1,2,3]:
            for task in ['asr','lid','text']:
                control={'task':task,'model_id':{'asr':'distil_whisper','lid':'whisper-small','text':'distilbert'}[task],'control':True}
                run(control,'latency',rep)
                for e in SHORT:
                    if e['task']==task and ok[(task,e['model_id'])]:run(e,'latency',rep)
    finally:nv.nvmlShutdown()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);a=p.parse_args()
    cpu_preflight() if a.mode=='cpu' else campaign()
