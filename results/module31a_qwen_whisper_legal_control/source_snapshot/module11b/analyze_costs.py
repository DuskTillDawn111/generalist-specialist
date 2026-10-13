"""Independent resource measurements; no equivalent-memory or penalty score."""
from collections import defaultdict
from scipy.stats import t
from .common import *
from joint_deploy.measurement.metrics import percentile

def paired(values):
    values=list(values);out=stat(values)
    if len(values)==3:
        half=float(t.ppf(.975,2))*statistics.stdev(values)/math.sqrt(3)
        out['mean_95percent_t_interval']=[statistics.mean(values)-half,statistics.mean(values)+half]
        out['all_three_positive']=all(v>0 for v in values)
        out['repeatable_positive_supported']=out['all_three_positive'] and out['mean_95percent_t_interval'][0]>0
    return out

def idle(raw,key):
    samples=raw.get(key,[]);PIDs={i:w['pid'] for i,w in raw['worker_identities'].items()}
    per_worker={}
    for i,pid in PIDs.items():
        per_worker[i]={'GPU_mib':stat(next((p['used_mib'] for p in s['processes'] if p['pid']==pid),None) for s in samples),
            'host_RSS_bytes':stat(sum(p['rss_bytes'] for p in s['host_processes'] if p['root_worker_pid']==pid) for s in samples),
            'CPU_core_percent':stat(sum(p['cpu_core_percent'] or 0. for p in s['host_processes'] if p['root_worker_pid']==pid) for s in samples)}
    return {**{key:stat(s[key] for s in samples)['mean'] for key in ['used_mib','owned_used_mib','host_RSS_bytes','worker_CPU_core_percent','owned_process_tree_count','gpu_util_percent']},
        'compute_PID_count':stat(len(s['processes']) for s in samples)['mean'], 'per_worker':per_worker}

def main():
    plan=read(OUT/'measurement_plan.json');manifest=read(OUT/'resident_set_manifest.json');fixed=[];low=[];raws={}
    for p in sorted((OUT/'raw_runs').glob('*.json')):
        raw=read(p);slot=raw['slot']
        if slot['kind']!='fixed_cost':continue
        raws[(slot['set_id'],slot['seed'])]=raw
        for stage in ['cold_idle','postwarm_idle']:
            ev=idle(raw,stage)
            fixed.append({'set_id':slot['set_id'],'seed':slot['seed'],'repetition':slot['repetition'],'idle_stage':stage,
                'deployment':slot['deployment'],'N_G':sum(n for m,n in slot['deployment'].items() if plan['models'][m]['role']=='generalist'),
                'N_S':sum(n for m,n in slot['deployment'].items() if plan['models'][m]['role']=='specialist'),**ev,
                'safe_owned_GPU_mib':factor()*ev['owned_used_mib'] if ev['owned_used_mib'] is not None else None,
                'startup_seconds':raw.get('startup_seconds'),
                'startup_ready_observation_ns':raw['telemetry'].get('startup_READY_observation_ns'),
                'exact_CUDA_context_count':None,'context_proxy':'NVML compute PIDs; exact context count unobservable',
                'valid_session':raw['valid_session'],'raw':ref(p)})
        for mode,ph in raw['phases'].items():
            records=[r for r in ph['request_records'] if r['task']=='visual_qa' and r['status']=='completed']
            tele=[s for s in raw['telemetry']['samples'] if s['phase']==mode]
            low.append({'set_id':slot['set_id'],'seed':slot['seed'],'repetition':slot['repetition'],'mode':mode,
                'deployment':slot['deployment'],'G_mean_E2E_ms':stat(r['e2e_latency_ms'] for r in records)['mean'],
                'G_P95_E2E_ms':percentile([r['e2e_latency_ms'] for r in records],95),
                'G_mean_worker_ms':stat(r['worker_latency_ms'] for r in records)['mean'],
                'G_throughput_rps':sum(r['completion_time_ns']<=ph['window']['end_time_ns'] for r in records)/ph['window']['duration_seconds'],
                'G_completed_requests':len(records),'stability':ph['stability']['stability_state_v2'],
                'G_nominal_pressure':slot['predicted_G_nominal_pressure'],'S_nominal_pressure_each':.001 if mode=='G_PLUS_LOW_S' else 0.,
                'GPU_util_percent':stat(s['gpu_util_percent'] for s in tele)['mean'],
                'CPU_core_percent':stat(s['worker_CPU_core_percent'] for s in tele)['mean'],
                'valid_session':raw['valid_session'],'raw':ref(p)})
    csvsave('fixed_cost_measurements.csv',fixed);csvsave('low_load_runtime_results.csv',low)
    by={(r['set_id'],r['seed']):r for r in fixed if r['idle_stage']=='postwarm_idle' and r['valid_session']}
    lby={(r['set_id'],r['seed'],r['mode']):r for r in low if r['valid_session'] and r['stability']=='steady'}
    increments=[];effects=[];own=defaultdict(list)
    states=manifest['sets']
    for state in states:
        x=state['deployment'];gs=tuple(sorted(m for m in x if plan['models'][m]['role']=='generalist'));ss=sorted(m for m in x if plan['models'][m]['role']=='specialist')
        for seed in plan['seeds']:
            row=by.get((state['set_id'],seed))
            if row:
                for i,v in row['per_worker'].items():
                    m=i.split('#')[0]
                    if plan['models'][m]['role']=='specialist':own[m].append(v)
        predecessors=[s for s in states if tuple(sorted(m for m in s['deployment'] if plan['models'][m]['role']=='generalist'))==gs and
                   sum(s['deployment'][m] for m in s['deployment'] if plan['models'][m]['role']=='specialist')<len(ss)]
        prev=max(predecessors,key=lambda s:(sum(s['deployment'][m] for m in s['deployment'] if plan['models'][m]['role']=='specialist'),s['set_id'])) if predecessors else None
        baseline=next((s for s in states if tuple(sorted(m for m in s['deployment'] if plan['models'][m]['role']=='generalist'))==gs and
                       not any(plan['models'][m]['role']=='specialist' for m in s['deployment'])),None)
        if prev:
            added=sorted(set(x)-set(prev['deployment']))
            diffs={k:[] for k in ['owned_used_mib','host_RSS_bytes','worker_CPU_core_percent','owned_process_tree_count','compute_PID_count','startup_seconds']}
            for seed in plan['seeds']:
                a=by.get((state['set_id'],seed));b=by.get((prev['set_id'],seed))
                if a and b:
                    for k in diffs:diffs[k].append(a[k]-b[k])
            increments.append({'set_id':state['set_id'],'previous_set_id':prev['set_id'],'added_specialist':added,
                'paired_resource_increments':{k:paired(v) for k,v in diffs.items()}})
        if baseline and ss:
            for mode in ['G_ONLY','G_PLUS_LOW_S']:
                diffs={k:[] for k in ['G_mean_E2E_ms','G_P95_E2E_ms','G_mean_worker_ms','G_throughput_rps','GPU_util_percent','CPU_core_percent']}
                for seed in plan['seeds']:
                    a=lby.get((state['set_id'],seed,mode));b=lby.get((baseline['set_id'],seed,'G_ONLY'))
                    if a and b:
                        for k in diffs:diffs[k].append(a[k]-b[k])
                effects.append({'set_id':state['set_id'],'baseline_set_id':baseline['set_id'],'mode':mode,
                    'scope':'paired independent repetitions under frozen low-load visual_qa; all contributing phases steady',
                    'paired_differences':{k:paired(v) for k,v in diffs.items()}})
            diffs={k:[] for k in ['G_mean_E2E_ms','G_P95_E2E_ms','G_mean_worker_ms','G_throughput_rps','GPU_util_percent','CPU_core_percent']}
            for seed in plan['seeds']:
                a=lby.get((state['set_id'],seed,'G_PLUS_LOW_S'));b=lby.get((state['set_id'],seed,'G_ONLY'))
                if a and b:
                    for k in diffs:diffs[k].append(a[k]-b[k])
            effects.append({'set_id':state['set_id'],'baseline_set_id':state['set_id'],'mode':'RARE_ACTIVATION_MINUS_IDLE_RESIDENCY',
                'scope':'same resident set and PIDs; paired phase order alternates by seed; actual rare Specialist work included, not a pure idle-context cost',
                'paired_differences':{k:paired(v) for k,v in diffs.items()}})
    c,hw,roles,base=inputs();bg=read(OLD/'input_freeze.json')['background'];trade=[]
    for task in sorted(bg):
        gs=[m for m in c if roles[m]=='generalist' and task in c[m].supported_tasks];ss=[m for m in c if roles[m]=='specialist' and task in c[m].supported_tasks]
        if not gs or not ss:continue
        G=min(gs,key=lambda m:(c[m].service_time_ms[task],m));S=min(ss,key=lambda m:(c[m].service_time_ms[task],m))
        resources={k:stat(v[key]['mean'] for v in own[S]) for k,key in [('S_GPU_mib','GPU_mib'),('S_host_RSS_bytes','host_RSS_bytes'),('S_CPU_core_percent','CPU_core_percent')]}
        for lam in [0.]+[bg[task]*10**e for e in range(-6,1)]:
            trade.append({'task':task,'Generalist':G,'Specialist':S,'lambda_rps':lam,
                'service_demand_saving_seconds_per_second':lam*(c[G].service_time_ms[task]-c[S].service_time_ms[task])/1000,
                'service_time_advantage_ms':c[G].service_time_ms[task]-c[S].service_time_ms[task],
                'measured_S_GPU_mib':resources['S_GPU_mib']['median'],'measured_S_host_RSS_bytes':resources['S_host_RSS_bytes']['median'],
                'measured_S_idle_CPU_core_percent':resources['S_CPU_core_percent']['median'],
                'additional_worker_process':1,'additional_NVML_compute_PID':1,
                'resource_observation_scope':'per-worker residency in frozen measured Phi/Qwen ladder; not a new optimizer penalty',
                'resource_estimation_count':resources['S_GPU_mib']['n']})
    csvsave('specialist_low_qps_tradeoff.csv',trade)
    complete=len(raws)==len(states)*3 and all(r['valid_session'] for r in raws.values())
    findings={'stage_complete':complete,'independent_sessions':len(raws),'expected_sessions':len(states)*3,
        'resource_increments':increments,'low_load_effects':effects,
        'per_specialist_resources':{m:{k:stat(v[k]['mean'] for v in values) for k in ['GPU_mib','host_RSS_bytes','CPU_core_percent']} for m,values in own.items()},
        'low_QPS_benefit_limit':'DeltaD=lambda*(sG-sS) tends to zero; resident resources do not scale with lambda',
        'no_universal_low_QPS_not_worth_claim':'different resource units kept separate; opportunity cost depends on actual budgets and co-residents',
        'cost_attribution':'G_ONLY measures idle residency; G_PLUS_LOW_S includes real rare Specialist service plus scheduling/host effects; active-minus-idle isolates adding requests, not pure context-switch cost',
        'no_optimizer_change':True,'limitations':['Three independent repetitions; latency uncertainty reported, no performance threshold tuned.',
            'Host RSS sums may count shared pages repeatedly; RSS is not PSS.',
            'CPU percentage unit: 100 equals one core; controller excluded.',
            'Exact CUDA context count unavailable; NVML process count is only a proxy.',
            'Paired t intervals descriptive; no multiple-testing claim of universal context interference.',
            'Low-load workload uses frozen admitted visual_qa only; no new task.']}
    save('fixed_cost_characterization.json',findings)
    print(json.dumps({'sets_measured':len(raws),'complete':complete,'low_load_rows':len(low)}),flush=True)
if __name__=='__main__':main()
