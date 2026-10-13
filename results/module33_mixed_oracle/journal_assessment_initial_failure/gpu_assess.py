from .common import *
import numpy as np
from types import SimpleNamespace
from module26.common import arithmetic
from module20.optimizer import components
from module21.trajectory import choose
from performance_model.local_calibration.stability_v2 import reconstruct,classify,DEFAULT_THRESHOLDS

def assess(raw):
    fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);d=read(sp['pools']['SMALL6']['path']);weights=read(fm['utility_definition']['path'])['weights'];refs=read(fm['utility_reference']['path'])['latency_reference_ms']
    checks=dict(valid_session=raw.get('valid_session',False),nonblocking_control=raw.get('control_nonblocking',False),write_isolation=not raw.get('unexpected_write_violations',[]))
    ph=raw.get('phases',{}).get('measurement');workers={};value={};errors=[]
    if ph:
        rs=ph['request_records'];w=ph['window'];tr=read(raw['slot']['trajectories']['measurement']['path'])['events'];rates=raw['slot']['lambda_rps'];mapping=raw['replica_to_family']
        checks.update(frozen_duration=w['planned_duration_seconds']==raw['slot']['measurement_seconds'] and w['end_time_ns']-w['start_time_ns']==round(raw['slot']['measurement_seconds']*1e9),
            full_cohort=len(rs)==len(tr) and all(r['status']=='completed' for r in rs),six_tasks=set(r['task'] for r in rs)==set(rates),
            trace_and_route=all(r['task']==e['task'] and r['sample_id']==e['item']['sample'].get('id') and r['frozen_offset_seconds']==e['offset_seconds'] and r['model']==choose(e,raw['slot']['routing'],raw['slot']['deployment']) for r,e in zip(rs,tr)),
            native_timestamps=all(all(r.get('worker_response',{}).get(k) is not None for k in ['submit_ns','start_ns','end_ns']) and abs(r['worker_latency_ms']-(r['worker_response']['end_ns']-r['worker_response']['start_ns'])/1e6)<1e-6 and abs(r['worker_queue_wait_ms']-(r['worker_response']['start_ns']-r['worker_response']['submit_ns'])/1e6)<1e-6 for r in rs if r['status']=='completed'),
            bounded_drain=w['drain_seconds']<=60.1)
        ss=[s for s in raw['telemetry']['samples'] if w['start_time_ns']<=s['time_ns']<=w['end_time_ns']];ts=[s['time_ns'] for s in ss]
        checks.update(sensor_values=bool(ss) and all(all(s.get(k) is not None and np.isfinite(s[k]) for k in ['time_ns','unix_ns','used_mib','owned_used_mib','gpu_util_percent','worker_CPU_core_percent','host_RSS_bytes','temperature_gpu_c','sm_clock_mhz']) for s in ss),
            no_external_GPU=all(not s['foreign_pids'] for s in ss),
            telemetry_coverage=bool(ts) and (ts[0]-w['start_time_ns'])/1e9<2 and (w['end_time_ns']-ts[-1])/1e9<2 and max(np.diff(ts)/1e9,default=0)<5,
            monotonic_timestamps=all(b>a for a,b in zip(ts,ts[1:])),
            quality_capability=all(r['task'] in d['models'][mapping[r['model']]]['supported_tasks'] for r in rs))
        aliases={i:SimpleNamespace(service_time_ms=d['models'][m]['service_time_ms']) for i,m in mapping.items()}
        for i in mapping:
            rrs=[r for r in rs if r['model']==i]
            if not rrs:workers[i]=dict(stability_state_v2='IDLE_NO_TRAFFIC',final_recovery_confirmed=True);continue
            tmp=dict(request_records=rrs,measurement_window=w,status='completed' if all(r['status']=='completed' for r in rrs) else 'completed_with_failures',errors=raw['errors'],drain_timeout_seconds=60.,shutdown=raw['shutdown'])
            ev=reconstruct(tmp,aliases,DEFAULT_THRESHOLDS);workers[i]={**classify(ev,DEFAULT_THRESHOLDS),'evidence':ev}
        value=arithmetic(raw,d,refs,weights)
        checks['safe_memory_budget']=value.get('safe_memory_mib',float('inf'))<=raw['slot']['safe_memory_budget_mib']+1e-6
        if 'raw_arithmetic_U' in value:
            u=components(value['per_task'],rates,refs,value['T_norm'],value['M_norm'],weights)['U'];checks['independent_U_arithmetic']=abs(u-value['raw_arithmetic_U'])<1e-12
        else:checks['independent_U_arithmetic']=False
        # Durable journal must contain each full phase's terminal record exactly once.
        jp=OUT/'request_journals'/(raw['slot']['run_id']+'.jsonl')
        lines=[json.loads(s) for s in jp.read_text().splitlines()] if jp.exists() else []
        terms=[l['record'] for l in lines if l['kind']=='REQUEST_TERMINAL' and l['phase']=='measurement']
        by={r['request_id']:r for r in terms}
        checks['durable_terminal_journal']=len(terms)==len(by)==len(rs) and all(by.get(r['request_id'])==r for r in rs)
    else:checks['measurement_present']=False
    technical=all(checks.values());system=ph['stability'] if ph else None
    stable=bool(system and system['stability_state_v2']!='unstable' and all(v['stability_state_v2']!='unstable' and v['final_recovery_confirmed'] for v in workers.values()))
    eligible=technical and stable
    env={k:float(np.mean([s[k] for s in ss])) for k in ['temperature_gpu_c','sm_clock_mhz','power_draw_w','gpu_util_percent','owned_used_mib','worker_CPU_core_percent']} if ph and ss else {}
    return dict(run_id=raw['slot']['run_id'],candidate_id=raw['slot']['candidate_id'],stage=raw['slot']['stage'],U=value.get('raw_arithmetic_U') if eligible else None,
        eligible=eligible,technical_valid=technical,stable=stable,checks=checks,components=value,system_stability=system,worker_stability=workers,
        environment=env,raw=ref(OUT/'raw_runs'/(raw['slot']['run_id']+'.json')))
