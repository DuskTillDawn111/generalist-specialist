"""Sequential locked characterization runner; never calls a deployment search."""
import os,fcntl,subprocess,time
from .common import *
from replica_scaling.telemetry import snapshot

def launch(slot):
    directory=OUT/'slots';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"{slot['run_id']}.json";output=OUT/'raw_runs'/f"{slot['run_id']}.json"
    if output.exists():
        raw=read(output);assert raw['slot']==slot,'Exact slot identity mismatch'
        return raw
    assert not snapshot()['processes'],'Unexpected resident GPU process'
    if path.exists():assert read(path)==slot
    else:save_json(path,slot)
    ledger=read(OUT/'execution_ledger.json') if (OUT/'execution_ledger.json').exists() else {'attempts':[]}
    assert not any(r['run_id']==slot['run_id'] for r in ledger['attempts']),'Started attempt without raw file; do not outcome-retry'
    record={'run_id':slot['run_id'],'slot':ref(path),'start_utc':now(),'status':'STARTED',
            'harness_sources':[ref(p) for p in (ROOT/'module11b').glob('*.py')]}
    ledger['attempts'].append(record);save('execution_ledger.json',ledger)
    (OUT/'raw_runs').mkdir(parents=True,exist_ok=True)
    log=OUT/'raw_runs'/f"{slot['run_id']}.log";env=os.environ.copy()
    env.update(read(OUT/'measurement_plan.json')['environment_overrides'])
    with log.open('w') as stream:
        module='module11b.closed_loop' if slot['kind']=='closed_loop_profile' else 'module11b.measure'
        p=subprocess.run(['/root/miniconda3/envs/gsdeploy/bin/python','-m',module,'--slot',str(path)],cwd=ROOT,env=env,
            stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
    record.update(status='RETURNED',returncode=p.returncode,end_utc=now(),raw=ref(output) if output.exists() else None)
    save('execution_ledger.json',ledger)
    assert output.exists(),f'Measurement returned without raw result: {log}'
    raw=read(output);assert raw['gpu_release']['verified'],'GPU release failed; cannot continue measurement'
    print(json.dumps({'run':slot['run_id'],'valid':raw['valid_session'],'elapsed_seconds':raw['elapsed_seconds'],
        'capacity':raw.get('phases',{}).get('closed_loop_capacity',{}).get('throughput_rps',raw.get('phases',{}).get('capacity',{}).get('metrics',{}).get('completed_throughput_rps')),
        'low_stability':raw.get('phases',{}).get('low_load',{}).get('stability',{}).get('stability_state_v2'),
        'errors':raw['errors']}),flush=True)
    return raw

def profile_frontier(required,g_previous,tag='initial'):
    plan=read(OUT/'measurement_plan.json');m=required['model'];x=required['required_x'];task=required['primary_task']
    s=plan['models'][m]['service_time_ms'][task]/1000;rate=(g_previous+1.2)/s;low=.45*g_previous/s
    for ri,seed in enumerate(plan['seeds'],1):
        counts=[x-1,x] if ri%2 else [x,x-1]
        for n in counts:
            slot={'kind':'replica_profile','run_id':f'{m}_frontier{x}_{tag}_x{n}_rep{ri}','model':m,'required_x':x,
                'repetition':ri,'seed':seed,'primary_task':task,'g_previous_frozen':g_previous,
                'deployment':{m:n},'lambda_rps':{task:rate},'routing':{task:{m:rate}},
                'warm_lambda':{task:low},'warm_routing':{task:{m:low}},'low_lambda':{task:low},'low_routing':{task:{m:low}},
                'measurement_seconds':plan['measurement_seconds'],'affected_contexts':required['affected_contexts'],
                'affected_boundary':required['affected_boundary'],'reason':required['reason'],
                'capacity_session_comparison':'same paired rate/seed/input pool, different independent process count'}
            launch(slot)

def initial():
    plan=read(OUT/'measurement_plan.json');c,h,roles,cal=inputs()
    for required in plan['required_initial_frontier']:
        profile_frontier(required,cal.capacity_factors[required['model']][required['required_x']-1])
    save('initial_frontier_status.json',{'status':'COMPLETED','at_utc':now(),'required_points':len(plan['required_initial_frontier'])})

def closed_loop_frontier(required,g_previous,tag='closed_loop'):
    plan=read(OUT/'measurement_plan.json');supp=read(OUT/'closed_loop_protocol.json')
    m=required['model'];x=required['required_x'];task=required['primary_task']
    low=.45*g_previous/(plan['models'][m]['service_time_ms'][task]/1000)
    for ri,seed in enumerate(supp['seeds'],1):
        for n in ([x-1,x] if ri%2 else [x,x-1]):
            slot={'kind':'closed_loop_profile','run_id':f'{m}_frontier{x}_{tag}_x{n}_rep{ri}',
                'model':m,'required_x':x,'repetition':ri,'seed':seed,'primary_task':task,
                'g_previous_frozen':g_previous,'deployment':{m:n},
                'lambda_rps':{task:low},'routing':{task:{m:low}},
                'warm_lambda':{task:low},'warm_routing':{task:{m:low}},
                'low_lambda':{task:low},'low_routing':{task:{m:low}},
                'measurement_seconds':supp['measurement_seconds'],'affected_contexts':required['affected_contexts'],
                'affected_boundary':required['affected_boundary'],'reason':required['reason'],
                'closed_loop_protocol':ref(OUT/'closed_loop_protocol.json'),
                'capacity_session_comparison':'same paired seed/samples, serial per-replica closed loop, independent processes'}
            launch(slot)

def closed_initial():
    from .analyze_replica import main as analyze
    plan=read(OUT/'measurement_plan.json');c,h,roles,base=inputs()
    for required in plan['required_initial_frontier']:
        closed_loop_frontier(required,base.capacity_factors[required['model']][required['required_x']-1])
        analyze()
    save('closed_loop_initial_status.json',{'status':'COMPLETED','at_utc':now()})

def costs():
    plan=read(OUT/'measurement_plan.json');manifest=read(OUT/'resident_set_manifest.json')
    bg=read(OLD/'input_freeze.json')['background'];rate=manifest['low_load_protocol']['Generalist_total_rate_rps']
    for state in manifest['sets']:
        x=state['deployment'];gs=sorted(m for m in x if plan['models'][m]['role']=='generalist');ss=sorted(set(x)-set(gs))
        base={'visual_qa':rate};route={'visual_qa':{m:rate/len(gs) for m in gs}}
        active=dict(base);ar={k:dict(v) for k,v in route.items()}
        for m in ss:
            k=min(plan['models'][m]['supported_tasks'],key=lambda k:(-bg[k]*plan['models'][m]['service_time_ms'][k],k))
            r=.001/(plan['models'][m]['service_time_ms'][k]/1000);active[k]=active.get(k,0.)+r;ar.setdefault(k,{})[m]=r
        for ri,seed in enumerate(plan['seeds'],1):
            slot={'kind':'fixed_cost','run_id':state['set_id']+f'_rep{ri}','set_id':state['set_id'],'repetition':ri,'seed':seed,
                'deployment':x,'lambda_rps':base,'routing':route,'warm_lambda':active,'warm_routing':ar,
                'active_lambda':active,'active_routing':ar,'measurement_seconds':30/rate,
                'predicted_G_nominal_pressure':{m:rate/len(gs)*plan['models'][m]['service_time_ms']['visual_qa']/1000 for m in gs},
                'predicted_S_nominal_pressure_each':.001,'safe_memory_mib':state['safe_memory_mib']}
            launch(slot)
    save('fixed_cost_stage_status.json',{'status':'COMPLETED','sets':len(manifest['sets']),'at_utc':now()})

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initial','closed_initial','costs']);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'GPU_execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze=read(OUT/'input_freeze.json');fast_verify(freeze['protected_references'])
    assert not subprocess.run(['pgrep','-af','[n]vidia-cuda-mps'],capture_output=True,text=True).stdout.strip()
    {'initial':initial,'closed_initial':closed_initial,'costs':costs}[args.stage]()

if __name__=='__main__':main()
