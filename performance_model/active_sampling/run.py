"""Bounded serial execution of the frozen 18-run acquisition plan."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
from performance_model.resources import source
from performance_model.stability import label_stability
from .common import *


def execute(output=OUT):
    output=Path(output);plan=json.loads((output/'selected_conditions.json').read_text());ex=extractor()
    for p,h in plan['protected_sources'].items():
        if source(ROOT/p)['sha256']!=h: raise ValueError('Frozen source changed: '+p)
    if len(plan['conditions'])!=6 or sum(len(c['runs']) for c in plan['conditions'])!=18: raise ValueError('Budget changed')
    for old in json.loads((output/'candidate_pool.json').read_text())['prior_sources']:
        if source(old['path'])['sha256']!=old['sha256']: raise ValueError('Prior data changed')
    state={'status':'running','completed_runs':[],'maximum_runs':18,'budget_expansion_allowed':False}
    save(output/'execution_state.json',state)
    for rep in range(3):
        # Within each repetition: stable -> boundary -> risky; seeded ordering within bands.
        ordered=[]
        for category in ('clearly_stable','near_boundary','likely_unstable'):
            group=[c for c in plan['conditions'] if c['expected_category']==category]
            random.Random(f'active:{rep}:{category}').shuffle(group);ordered.extend(group)
        for condition in ordered:
            spec=condition['runs'][rep];path=Path(spec['result']);path.parent.mkdir(parents=True,exist_ok=True)
            if source(spec['config'])['sha256']!=spec['config_sha256']: raise ValueError('Run config changed')
            timed_out=False
            if not path.exists():
                print('START '+spec['run_id']+' '+condition['expected_category'],flush=True)
                Path(spec['log']).parent.mkdir(parents=True,exist_ok=True)
                with open(spec['log'],'w') as log:
                    child=subprocess.Popen([sys.executable,'-m','joint_deploy.measurement.run_measurement','--config',spec['config'],'--output',spec['result']],
                        cwd=ROOT,env={**os.environ,**plan['environment']},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try: child.wait(timeout=plan['policy']['safe_run_timeout_seconds'])
                    except subprocess.TimeoutExpired:
                        timed_out=True;os.kill(child.pid,signal.SIGINT)
                        try: child.wait(timeout=plan['policy']['interrupt_grace_seconds'])
                        except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
                if not path.exists():
                    cfg=json.loads(Path(spec['config']).read_text())
                    save(path,dict(**cfg,status='failed',errors=['Harness did not save a result; see run log'],request_records=[],measurement_window=None,system_metrics=None))
                data=json.loads(path.read_text())
                data['active_sampling']=dict(configuration_id=condition['configuration_id'],deployment_id=condition['deployment_id'],
                    repetition_id=spec['repetition_id'],run_id=spec['run_id'],expected_category=condition['expected_category'],
                    actual_features=ex.extract(data['deployment'],data['workload_rps'],data['routing_rps']),
                    stability=label_stability(data,plan['policy']['stability_thresholds']).as_dict(),safety_timeout=timed_out,
                    safety_policy=plan['policy'],acquisition_source=str(output/'selected_conditions.json'))
                save(path,data)
            data=json.loads(path.read_text())
            if 'active_sampling' not in data:
                # Recover a valid harness result saved just before interruption, without rerunning it.
                data['active_sampling']=dict(configuration_id=condition['configuration_id'],deployment_id=condition['deployment_id'],
                    repetition_id=spec['repetition_id'],run_id=spec['run_id'],expected_category=condition['expected_category'],
                    actual_features=ex.extract(data['deployment'],data['workload_rps'],data['routing_rps']),
                    stability=label_stability(data,plan['policy']['stability_thresholds']).as_dict(),safety_policy=plan['policy'],recovered_result=True)
                save(path,data)
            print('END '+spec['run_id']+' '+data['active_sampling']['stability']['category'],flush=True)
            state['completed_runs'].append(spec['run_id']);save(output/'execution_state.json',state)
    state['status']='measurement_complete';save(output/'execution_state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args();execute(args.output_dir.resolve())
    from .audit import audit
    audit(args.output_dir.resolve())
