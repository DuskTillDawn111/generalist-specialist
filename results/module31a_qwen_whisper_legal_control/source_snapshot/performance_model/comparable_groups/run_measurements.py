"""Execute only the frozen Module 2G plan, sequentially, with bounded runs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .plan import OUT, sha


def main():
    plan_hash=json.loads((OUT/'comparison_plan_hash.json').read_text())
    plan_path=Path(plan_hash['path'])
    if sha(plan_path)!=plan_hash['sha256']: raise RuntimeError('comparison_plan.json changed after freeze')
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    if sha(OUT/'run_manifest.json')!=plan_hash['run_manifest_sha256']: raise RuntimeError('run_manifest changed after freeze')
    done=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for group in manifest['groups']:
        for candidate in group['selected_candidates']:
            for run in candidate['runs']:
                result=Path(run['result_path']); log=Path(run['log_path']); result.parent.mkdir(parents=True,exist_ok=True)
                if result.exists():
                    done.append(dict(run_id=run['run_id'],status='existing',result=str(result))); continue
                command=[sys.executable,'-m','joint_deploy.measurement.run_measurement','--config',run['config_path'],'--output',run['result_path']]
                start=time.time()
                with log.open('w') as stream:
                    proc=subprocess.run(command,cwd=str(Path(__file__).resolve().parents[2]),env=env,stdout=stream,stderr=subprocess.STDOUT,check=False)
                status='completed'
                if result.exists():
                    data=json.loads(result.read_text());status=data.get('status','missing')
                done.append(dict(run_id=run['run_id'],status=status,returncode=proc.returncode,result=str(result),elapsed_seconds=time.time()-start))
                print(json.dumps(done[-1]),flush=True)
    (OUT/'run_execution_summary.json').write_text(json.dumps(dict(plan_sha256=plan_hash['sha256'],runs=done),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(total=len(done),status_counts={s:sum(x['status']==s for x in done) for s in sorted({x['status'] for x in done})}),indent=2))


if __name__=='__main__': main()
