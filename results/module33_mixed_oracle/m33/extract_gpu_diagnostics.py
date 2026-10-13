"""Read-only native timing diagnostics and measured static baseline comparisons."""
from .common import *
import numpy as np

def main():
    candidates={c['candidate_id']:c for c in read(OUT/'gpu_candidates.json')};requests=[];paths=[];sensors=[]
    observations=[read(p) for p in (OUT/'gpu_observations').glob('*.json')]
    for o in observations:
        raw=read(o['raw']['path']);ph=raw.get('phases',{}).get('measurement')
        if not ph:continue
        rs=ph['request_records'];slot=raw['slot'];mapping=raw['replica_to_family'];candidate=candidates[o['candidate_id']]
        for r in rs:
            wr=r.get('worker_response',{});family=mapping[r['model']]
            service=(wr['end_ns']-wr['start_ns'])/1e6 if all(wr.get(k) is not None for k in ['start_ns','end_ns']) else None
            requests.append(dict(run_id=o['run_id'],candidate_id=o['candidate_id'],stage=o['stage'],eligible=o['eligible'],request_id=r['request_id'],task=r['task'],worker=r['model'],family=family,sample_id=r['sample_id'],status=r['status'],
                frozen_offset_seconds=r['frozen_offset_seconds'],scheduled_time_ns=r['scheduled_time_ns'],actual_submit_time_ns=r.get('actual_submit_time_ns'),completion_time_ns=r.get('completion_time_ns'),
                worker_submit_ns=wr.get('submit_ns'),worker_start_ns=wr.get('start_ns'),worker_end_ns=wr.get('end_ns'),
                service_ms=service,worker_wait_ms=r.get('worker_queue_wait_ms'),E2E_ms=r.get('e2e_latency_ms'),runtime_ms=r.get('runtime_latency_ms'),schedule_delay_ms=r.get('schedule_delay_ms')))
        for task,family in sorted({(r['task'],mapping[r['model']]) for r in rs}):
            rr=[r for r in requests if r['run_id']==o['run_id'] and r['task']==task and r['family']==family and r['status']=='completed'];n=slot['deployment'][family]
            sp=read(OUT/'candidate_space_definition.json');d=read(sp['pools']['SMALL6']['path']);predsvc=d['models'][family]['service_time_ms'][task]*n/d['capacity_factors'][family][str(n)]
            vals={field:[r[field] for r in rr if r[field] is not None] for field in ['service_ms','worker_wait_ms','E2E_ms','schedule_delay_ms']}
            paths.append(dict(run_id=o['run_id'],candidate_id=o['candidate_id'],stage=o['stage'],task=task,family=family,replicas=n,completed=len(rr),eligible=o['eligible'],
                predicted_constant_service_ms=predsvc,**{field+'_'+stat:float(fn(v)) if v else None for field,v in vals.items() for stat,fn in [('mean',np.mean),('p95',lambda z:np.percentile(z,95)),('max',np.max)]},
                observed_service_over_prediction=float(np.mean(vals['service_ms']))/predsvc if vals['service_ms'] else None))
        sensors.extend(dict(run_id=o['run_id'],candidate_id=o['candidate_id'],**s) for s in raw['telemetry']['samples'])
    csvsave('gpu_all_requests.csv',requests);csvsave('gpu_worker_service_wait.csv',paths);csvsave('gpu_sensor_samples.csv',sensors)
    dev=read(OUT/'development_observations.json') if (OUT/'development_observations.json').exists() else []
    by={o['candidate_id']:o for o in dev};bench=[]
    for dec in read(OUT/'optimizer_decisions.json'):
        if dec['pool']!='SMALL6' or dec['load']!='ASR32B':continue
        c=next((v for v in candidates.values() if identity(v['deployment'],v['routing'])==identity(dec['deployment'],dec['routing'])),None)
        o=by.get(c['candidate_id']) if c else None
        bench.append(dict(algorithm=dec['algorithm'],candidate_id=c['candidate_id'] if c else None,predicted_U=dec['predicted_U'],measured_development_U=o['U'] if o else None,eligible=o['eligible'] if o else None,
            scope='Baseline plan uses the exact corresponding measurement from the ALL8 competition; same plans reuse a measurement, not independent trials.'))
    csvsave('gpu_measured_baseline_comparison.csv',bench)
    save('gpu_native_timing_diagnostics.json',dict(runs_with_measurements=len({r['run_id'] for r in requests}),full_request_count=len(requests),native_service='(worker.end_ns-worker.start_ns)/1e6',
        worker_wait='worker.start_ns-worker.submit_ns; host send-start, not worker acceptance ACK',E2E='host completion perf time - scheduled perf time',
        causal_interference_identified=False,components_never_refit=True))
    print('GPU_DIAGNOSTICS',len(requests),'requests',flush=True)
if __name__=='__main__':main()
