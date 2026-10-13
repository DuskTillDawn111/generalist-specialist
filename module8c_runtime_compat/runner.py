"""Original process runner with only the harness entry point changed."""
import os, signal, subprocess, sys, time
from pathlib import Path
from deployment_search.runtime_cache import save_json
from integer_replica_pilot.evaluator import ROOT


def runner(config,raw_path,log_path,config_path,context_path):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,'-m','module8c_runtime_compat.runtime','--config',str(config_path),
        '--context',str(context_path),'--output',str(raw_path)]
    timeout=sum(config['deployment'].values())*(config['startup_timeout_seconds']+config['shutdown_timeout_seconds'])+config['measurement_seconds']+config['warmup_seconds']+3*config['drain_timeout_seconds']+120
    start=time.time()
    with Path(log_path).open('x') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save_json(Path(raw_path).with_suffix('.process.json'),{'pid':proc.pid,'command':command,'start_unix':start})
        try:code=proc.wait(timeout=timeout);outcome={'returncode':code,'timed_out':False}
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();outcome={'returncode':proc.returncode,'timed_out':True}
    return dict(outcome,elapsed_seconds=time.time()-start)
