"""Add measured prefix points, retain historical tables byte-for-value."""
import copy
from collections import defaultdict,Counter
from scipy.stats import t
from deployment_core.replica_calibration import ReplicaCalibration
from .common import *

def calibration():
    c,hw,roles,base=inputs()
    path=OUT/'replica_scaling.json'
    if not path.exists():return base
    data=read(path)
    g={m:{int(n):v for n,v in tab.items()} for m,tab in data['capacity_factors'].items()}
    M={m:{int(n):v for n,v in tab.items()} for m,tab in data['memory_table_mib'].items()}
    return ReplicaCalibration(g,M,data['evidence_bounds'],data['model_evidence'],data['source_hashes']).validate(c)

def full_capacity(raw):
    slot=raw['slot']
    if slot['kind']=='closed_loop_profile':
        phase=raw.get('phases',{}).get('closed_loop_capacity')
        return bool(raw['valid_session'] and phase and phase['all_requests_completed_after_drain'] and phase['window']['planned_duration_seconds']==slot['measurement_seconds'] and phase['throughput_rps']>0)
    phase=raw.get('phases',{}).get('capacity')
    return bool(raw['valid_session'] and phase and phase['window']['stop'] is None and
        abs(phase['window']['duration_seconds']-slot['measurement_seconds'])<1e-9 and phase['metrics']['completed_throughput_rps']>0)

def primary_rows():
    rows=[]
    for p in sorted((OUT/'raw_runs').glob('*.json')):
        raw=read(p)
        if raw['slot']['kind'] not in ('replica_profile','closed_loop_profile'):continue
        slot=raw['slot'];closed=slot['kind']=='closed_loop_profile'
        phase=raw.get('phases',{}).get('closed_loop_capacity' if closed else 'capacity')
        metrics={'completed_throughput_rps':phase['throughput_rps']} if closed and phase else phase['metrics'] if phase else {}
        low=raw.get('phases',{}).get('low_load');samples=raw['telemetry']['samples']
        probes=raw.get('path_probe_records',[]);latencies={}
        for task in read(OUT/'measurement_plan.json')['models'][slot['model']]['supported_tasks']:
            latencies[task]=stat(r.get('worker_latency_ms') for r in probes if r['task']==task and r['status']=='completed')
        rows.append({'run_id':slot['run_id'],'model':slot['model'],'required_x':slot['required_x'],
            'replica_count':slot['deployment'][slot['model']],'repetition':slot['repetition'],'seed':slot['seed'],
            'primary_task':slot['primary_task'],'metric_protocol':'CLOSED_LOOP_SERVICE_CAPACITY' if closed else 'PERIODIC_ARRIVAL_LATENCY_STABILITY',
            'offered_rps':None if closed else slot['lambda_rps'][slot['primary_task']],
            'throughput_rps':metrics.get('completed_throughput_rps'),'within_window_completion_ratio':metrics.get('within_window_completion_ratio'),
            'mean_worker_latency_ms':metrics.get('mean_worker_latency_ms'),'p95_worker_latency_ms':metrics.get('p95_worker_latency_ms'),
            'mean_E2E_ms':metrics.get('mean_e2e_ms'),'P95_E2E_ms':metrics.get('p95_e2e_ms'),
            'full_capacity_repetition':full_capacity(raw),'capacity_stability':phase['stability']['stability_state_v2'] if phase and not closed else None,
            'low_load_stability':low['stability']['stability_state_v2'] if low else None,
            'low_load_rps':next(iter(slot['low_lambda'].values())),
            'owned_idle_VRAM_mib':stat(s['owned_used_mib'] for s in raw.get('postwarm_idle',[]))['mean'],
            'safe_idle_VRAM_mib':factor()*stat(s['owned_used_mib'] for s in raw.get('postwarm_idle',[]))['mean'] if raw.get('postwarm_idle') else None,
            'peak_owned_VRAM_mib':max((s['owned_used_mib'] for s in samples),default=None),
            'peak_device_VRAM_mib':max((s['used_mib'] for s in samples),default=None),
            'idle_host_RSS_bytes':stat(s['host_RSS_bytes'] for s in raw.get('postwarm_idle',[]))['mean'],
            'idle_CPU_core_percent':stat(s['worker_CPU_core_percent'] for s in raw.get('postwarm_idle',[]))['mean'],
            'startup_seconds':raw.get('startup_seconds'),'per_task_serial_probe_latencies':latencies,
            'independent_worker_PIDs':{i:w['pid'] for i,w in raw['worker_identities'].items()},
            'valid_session':raw['valid_session'],'errors':raw['errors'],'raw':ref(p)})
    return rows

def minimum_cover_memory(c,roles,cal,m,xmemory,bg):
    import itertools
    required={k for k,v in bg.items() if v};others=sorted(set(c)-{m});best=None
    for bits in itertools.product([0,1],repeat=len(others)):
        active=[n for n,b in zip(others,bits) if b];covered=set(c[m].supported_tasks)
        for n in active:covered.update(c[n].supported_tasks)
        if not required<=covered:continue
        mem=factor()*(xmemory+sum(cal.memory_table_mib[n][1] for n in active))
        key=(mem,tuple(active))
        if best is None or key<best:best=key
    return {'minimum_joint_safe_memory_mib':best[0],'background_cover':list(best[1]),
        'scope':'minimum admitted quality/capability cover, ignoring capacity; measured family count memory and unchanged measured single-copy others'}

def main():
    c,hw,roles,base=inputs();plan=read(OUT/'measurement_plan.json');rows=primary_rows();csvsave('replica_profile_results.csv',rows)
    grouped=defaultdict(list)
    csvsave('periodic_profile_results.csv',[r for r in rows if r['metric_protocol']=='PERIODIC_ARRIVAL_LATENCY_STABILITY'])
    csvsave('closed_loop_profile_results.csv',[r for r in rows if r['metric_protocol']=='CLOSED_LOOP_SERVICE_CAPACITY'])
    for r in rows:
        if r['metric_protocol']=='CLOSED_LOOP_SERVICE_CAPACITY':grouped[(r['model'],r['required_x'])].append(r)
    g=copy.deepcopy(base.capacity_factors);M=copy.deepcopy(base.memory_table_mib);bounds=dict(base.evidence_bounds)
    model_evidence=copy.deepcopy(base.model_evidence);points=[];terminals={}
    bg=read(OLD/'input_freeze.json')['background']
    for (m,x),records in sorted(grouped.items(),key=lambda p:(p[0][1],p[0][0])):
        if x!=bounds[m]+1:continue
        prev=[r for r in records if r['replica_count']==x-1];new=[r for r in records if r['replica_count']==x]
        if len(prev)!=3 or len(new)!=3:continue
        prev=sorted(prev,key=lambda r:r['seed']);new=sorted(new,key=lambda r:r['seed']);assert [r['seed'] for r in prev]==[r['seed'] for r in new]
        low_stable=all(r['low_load_stability']=='steady' for r in new)
        low_unstable=all(r['low_load_stability']=='unstable' for r in new)
        ready=all(r['full_capacity_repetition'] for r in prev+new)
        paired=[b['throughput_rps']/a['throughput_rps'] for a,b in zip(prev,new)] if ready else None
        previous_pressured=ready # serial work-conserving closed loop, no external offered-rate cap
        new_arrival_censored=False
        ci=None
        if paired:
            mean=statistics.mean(paired);half=float(t.ppf(.975,2))*statistics.stdev(paired)/math.sqrt(3)
            ci=[mean-half,mean+half]
        memory=stat(r['owned_idle_VRAM_mib'] for r in new)
        cover=minimum_cover_memory(c,roles,base,m,memory['mean'],bg) if memory['n']==3 else None
        affected=hw.gpu_memory_mib
        memory_terminal=bool(cover and cover['minimum_joint_safe_memory_mib']>affected+1e-7)
        saturated=bool(ready and previous_pressured and all(v<=1. for v in paired))
        eligible=bool(ready and previous_pressured and not low_unstable)
        point={'model':m,'x':x,'previous_x':x-1,'g_previous':g[m][x-1],
            'paired_throughput_gains':paired,'gain_statistics':stat(paired or []),'gain_95percent_t_interval':ci,
            'previous_count_capacity_pressured_all_three':previous_pressured,'new_count_arrival_censored_any_repeat':new_arrival_censored,
            'all_three_low_load_steady':low_stable,'all_three_low_load_unstable':low_unstable,
            'count_memory_observation':memory,'minimum_joint_memory_cover':cover,
            'raw_repetitions':[r['raw'] for r in prev+new],'g_ratio_to_previous':statistics.median(paired) if eligible else None,
            'g_new':g[m][x-1]*statistics.median(paired) if eligible else None,'measured_point_added':eligible,
            'mixed_sign_increment_inconclusive_not_terminal':bool(eligible and min(paired)<=1<max(paired)),
            'capacity_scope':'closed-loop serial service-capacity throughput ratio on primary task ONLY; periodic workload E2E/stability reported separately',
            'capacity_protocol':ref(OUT/'closed_loop_protocol.json')}
        if low_unstable:point['classification']='unstable';terminals[m]={'reason':'LOW_LOAD_UNSTABLE_ALL_THREE','x':x,'evidence':point}
        elif memory_terminal:point['classification']='memory_infeasible';terminals[m]={'reason':'MEASURED_SAFE_MEMORY_INFEASIBLE_WITH_MINIMUM_COVER','x':x,'evidence':point}
        elif saturated:point['classification']='saturated';terminals[m]={'reason':'NO_POSITIVE_INCREMENT_IN_ALL_THREE_CLOSED_LOOP_PAIRS','x':x,'evidence':point}
        elif not eligible:point['classification']='EVIDENCE_INCOMPLETE_OR_REFERENCE_ARRIVAL_CENSORED'
        elif all(v>1 for v in paired):
            ideal=x/(x-1)
            point['classification']='near-linear' if ci[0]<=ideal<=ci[1] else 'sublinear' if ci[1]<ideal else 'superlinear_observed'
        else:point['classification']='INCONCLUSIVE_INCREMENT'
        if eligible:
            g[m][x]=point['g_new'];M[m][x]=memory['mean'];bounds[m]=x
            model_evidence[m]['Module11B_points']=model_evidence[m].get('Module11B_points',[])+[point]
        points.append(point)
    for m in c:
        for n,v in base.capacity_factors[m].items():assert g[m][n]==v
        for n,v in base.memory_table_mib[m].items():assert M[m][n]==v
    ReplicaCalibration(g,M,bounds,model_evidence,base.source_hashes).validate(c)
    save('replica_scaling.json',{'at_utc':now(),'capacity_factors':g,'memory_table_mib':M,'evidence_bounds':bounds,
        'model_evidence':model_evidence,'source_hashes':base.source_hashes,'new_points':points,'terminal_frontiers':terminals,
        'historical_points_unchanged':True,'Module9_loader_not_modified':True,'new_profile_runs':len(rows),
        'terminal_rule_source':'user explicitly authorized sequential stopping on measured memory/saturation/low-load instability',
        'higher_unmeasured_counts_never_extrapolated':True})
    save('replica_point_status.json',{'points':[{k:p[k] for k in ['model','x','classification','g_new','measured_point_added']} for p in points],
        'terminals':{m:{k:v[k] for k in ['reason','x']} for m,v in terminals.items()},'runs':len(rows)})
    print(json.dumps({'runs':len(rows),'points':[(p['model'],p['x'],p['classification'],p['g_new']) for p in points],'terminals':list(terminals)}),flush=True)

if __name__=='__main__':main()
