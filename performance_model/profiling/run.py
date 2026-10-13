"""Bounded, resumable serial execution using the unchanged Stage 2A harness."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from .plan import OUT,ROOT,save,source,extractor


def assess(data,policy):
    records=data.get('request_records',[]); window=data.get('measurement_window'); reasons=[]
    system=data.get('system_metrics') or {}
    ratio=system.get('completion_ratio')
    if ratio is None or ratio<policy['completion_ratio_threshold']: reasons.append('completion_ratio_missing_or_below_threshold')
    if data.get('status')!='completed' or data.get('errors'): reasons.append('worker_or_run_error')
    if any(r['status']=='error' for key in ('request_records','warmup_request_records','priming_request_records') for r in data.get(key,[])): reasons.append('request_error')
    if any(r['status']=='incomplete' for r in records): reasons.append('incomplete')
    if any(s.get('forced') or s.get('exit_code')!=0 or not s.get('bye_received') for s in data.get('shutdown',{}).values()): reasons.append('unclean_shutdown')
    bins=[]
    if window:
        start,end=window['start_time_ns'],window['end_time_ns']
        events=sorted([(r.get('enqueued_time_ns',r['scheduled_time_ns']),1) for r in records]+[(r['completion_time_ns'],-1) for r in records if r.get('completion_time_ns') is not None])
        active=cursor=0; last=start
        for i in range(10):
            edge=start+round((end-start)*(i+1)/10); left=start+round((end-start)*i/10); area=0
            while cursor<len(events) and events[cursor][0]<edge:
                t,d=events[cursor]; area+=active*(t-last); active+=d; last=t; cursor+=1
            area+=active*(edge-last); last=edge
            bins.append(dict(offset_seconds=(edge-start)/1e9,mean_backlog=area/(edge-left),backlog_at_end=active))
        means=[b['mean_backlog'] for b in bins[5:]]; threshold=max(10,.02*len(records))
        if means[-1]-means[0]>=threshold and sum(b>a for a,b in zip(means,means[1:]))>=3 and active>=threshold/2: reasons.append('sustained_backlog_growth')
        if window['drain_seconds']>policy['drain_overload_seconds']: reasons.append('drain_exceeds_threshold')
    else: reasons.append('measurement_unavailable')
    return dict(overloaded_or_unstable=bool(reasons),reasons=reasons,backlog_bins=bins,
        backlog_at_window_end=bins[-1]['backlog_at_end'] if bins else None,completion_ratio=ratio,
        drain_seconds=window.get('drain_seconds') if window else None)


def decorate(config,spec,plan,ex):
    path=Path(spec['result']); data=json.loads(path.read_text())
    actual=ex.extract(data['deployment'],data['workload_rps'],data['routing_rps'])
    counts=Counter((r['task'],r['model']) for r in data.get('request_records',[]))
    observed_r={k:{m:counts[k,m]/data['measurement_seconds'] for m in ex.catalog} for k in data['workload_rps']}
    observed_w={k:math.fsum(row.values()) for k,row in observed_r.items()}
    realized=ex.extract(data['deployment'],observed_w,observed_r)
    data['feature_directed_profiling']=dict(configuration=config['name'],experiment_family=config['experiment_family'],
        planned_features=config['expected_features'],actual_config_features=actual,
        observed_cohort_rate_features=realized,
        interpretation='Features recomputed from the actual run x/r/lambda by shared extractor; measured counts stored separately.',
        overload=assess(data,plan['measurement_policy']))
    save(path,data)
    return data


def execute_run(config,spec,plan,ex):
    path=Path(spec['result']); path.parent.mkdir(parents=True,exist_ok=True)
    if source(spec['config'])['sha256']!=spec['config_sha256']: raise ValueError('Config changed')
    if not path.exists():
        print('START '+spec['run_id'],flush=True)
        Path(spec['log']).parent.mkdir(parents=True,exist_ok=True)
        with open(spec['log'],'w') as log:
            child=subprocess.Popen([sys.executable,'-m','joint_deploy.measurement.run_measurement','--config',spec['config'],'--output',spec['result']],
                cwd=ROOT,env={**os.environ,**plan['execution_environment']},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: child.wait(timeout=plan['measurement_policy']['safe_run_timeout_seconds'])
            except subprocess.TimeoutExpired:
                # Let original harness catch KeyboardInterrupt and preserve request prefix.
                os.kill(child.pid,signal.SIGINT)
                try: child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=10)
                save(path.with_suffix('.timeout.json'),dict(run_id=spec['run_id'],safe_timeout=True,log=spec['log']))
        if not path.exists():
            cfg=json.loads(Path(spec['config']).read_text())
            save(path,dict(**cfg,status='failed',errors=['Child exited without result; see log'],request_records=[],system_metrics=None,measurement_window=None,shutdown={}))
    data=decorate(config,spec,plan,ex)
    print('END '+spec['run_id']+' '+json.dumps(data['feature_directed_profiling']['overload']),flush=True)
    return data['feature_directed_profiling']['overload']['overloaded_or_unstable']


def execute(output):
    plan=json.loads((output/'profiling_plan.json').read_text()); ex=extractor()
    for p,h in plan['protocol_hashes'].items():
        if source(ROOT/p)['sha256']!=h: raise ValueError('Frozen protocol changed: '+p)
    state=dict(status='running',completed_runs=[],skipped_configurations=[],stopped_load_axis_at=None)
    save(output/'execution_state.json',state)
    for config in [c for c in plan['configs'] if c['experiment_family']=='load_axis']:
        if state['stopped_load_axis_at'] is not None:
            # Balanced B anchor remains required; defer it to B even after A overload stop.
            state['skipped_configurations'].append(config['name']); continue
        unstable=False
        for spec in config['runs']:
            unstable |= execute_run(config,spec,plan,ex)
            state['completed_runs'].append(spec['run_id']); save(output/'execution_state.json',state)
        if unstable: state['stopped_load_axis_at']=config['expected_u']; save(output/'execution_state.json',state)
    for family in ('task_mix_axis','colocation_axis'):
        group=[c for c in plan['configs'] if c['experiment_family']==family]
        if family=='task_mix_axis':
            anchor=next(c for c in plan['configs'] if c['name']=='A_u_0.40')
            if not all(Path(s['result']).exists() for s in anchor['runs']): group=[anchor]+group
        for rep in range(3):
            ordered=list(group); random.Random(f'feature-profile:{family}:{rep}').shuffle(ordered)
            for config in ordered:
                spec=config['runs'][rep]; execute_run(config,spec,plan,ex)
                state['completed_runs'].append(spec['run_id']); save(output/'execution_state.json',state)
    state['status']='measurement_complete'; save(output/'execution_state.json',state)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args(); execute(args.output_dir.resolve())
