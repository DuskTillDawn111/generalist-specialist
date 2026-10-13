"""Sequential isolated frozen-subset quality measurement, not deployment search."""
import hashlib, json, os, queue, subprocess, threading, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from candidate_pool_audit.registry import registry
OUT = ROOT / 'results/quality_workload_audit'

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def gpu():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).strip()

def run(name, config, manifest):
    result_path = OUT / 'raw_quality' / (name + '.json')
    if result_path.exists() and json.loads(result_path.read_text()).get('status') == 'COMPLETE':
        existing=json.loads(result_path.read_text())
        if existing['manifest_sha256']!=hashlib.sha256((OUT/'quality_evaluation_manifest.json').read_bytes()).hexdigest():
            raise RuntimeError('Completed audit manifest changed')
        print('reuse audit', name, flush=True); return
    if gpu():
        raise RuntimeError('GPU occupied before quality audit')
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    if name=='paddleocr':
        # Exact library-path setup from the historical Worker.start.
        cuda_lib='/root/miniconda3/envs/ocr_env/lib/python3.11/site-packages/nvidia/cu13/lib'
        old=env.get('LD_LIBRARY_PATH','')
        env['LD_LIBRARY_PATH']=cuda_lib+(':'+old if old else '')
    script = config['script']
    if name in ('phi4mm','minicpm'):
        env['AUDIT_ORIGINAL_WORKER'] = str(script)
        script = ROOT / 'quality_workload_audit/vqa_worker.py'
    q = queue.Queue()
    p = subprocess.Popen([str(config['python']),str(script)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=ROOT,env=env)
    record = {'model':name,'status':'STARTED','pid':p.pid,'launcher':{'python':str(config['python']),'script':str(script)},
              'runtime_environment_overrides':{k:env[k] for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM','LD_LIBRARY_PATH'] if k in env},
              'audit_launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'start_unix':time.time(),'manifest_sha256':hashlib.sha256((OUT/'quality_evaluation_manifest.json').read_bytes()).hexdigest(),
              'tasks':{}}
    def reader():
        with (OUT/'raw_quality'/(name+'.log')).open('w') as log:
            for line in p.stdout:
                log.write(line); log.flush()
                if line.startswith(('READY|','RESULT|','ERROR|','BYE|')):
                    k,v=line.split('|',1); q.put((k,json.loads(v)))
        q.put(('EXIT',{'returncode':p.poll()}))
    threading.Thread(target=reader,daemon=True).start()
    def event(wanted, timeout=600):
        deadline=time.monotonic()+timeout
        while True:
            left=deadline-time.monotonic()
            if left<=0: raise TimeoutError(name+' '+wanted)
            k,v=q.get(timeout=left)
            if k in ('ERROR','EXIT'): raise RuntimeError(json.dumps(v))
            if k==wanted:return v
    try:
        record['ready']=event('READY')
        for task in manifest['models'][name]['tasks']:
            tm=manifest['tasks'][task]
            samples=[json.loads(l) for l in Path(tm['evaluation_manifest']).read_text().splitlines() if l]
            task_record={'status':'STARTED','responses':[], 'sample_ids':tm['sample_ids']}
            record['tasks'][task]=task_record
            for i,sample in enumerate(samples):
                request={'request_id':f'audit:{name}:{task}:{i}','pool_id':sample['id'],'arrival_index':i,
                         'task_type':task,'sample':sample}
                p.stdin.write(json.dumps(request)+'\n');p.stdin.flush()
                response=event('RESULT')
                if response['pool_id']!=sample['id']:raise RuntimeError('ID mismatch')
                task_record['responses'].append(response)
                if (i+1)%50==0:
                    save(result_path,record); print(name,task,i+1,'/',len(samples),flush=True)
            task_record['status']='COMPLETE'
            save(result_path,record)
        record['status']='COMPLETE'
    except Exception as e:
        record.update(status='PENDING_TECHNICAL_FAILURE',error=repr(e))
        print(name,'FAILED',repr(e),flush=True)
    finally:
        if p.poll() is None:
            try:
                p.stdin.write('{"op":"shutdown"}\n');p.stdin.flush();p.wait(timeout=30)
            except Exception:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        record.update(end_unix=time.time(),exit_code=p.returncode)
        save(result_path,record)
        if gpu():raise RuntimeError('GPU not unloaded after '+name)
    print(name,record['status'],flush=True)

def main():
    manifest=json.loads((OUT/'quality_evaluation_manifest.json').read_text())
    configs=registry()
    for name in manifest['models']:
        run(name,configs[name],manifest)
if __name__=='__main__':main()
