import itertools
import numpy as np
from scipy.optimize import minimize
from .common import *
from .latency import lower_bound

REQUIRED=['hotspot_sets.csv','normalized_directions.json','multihotspot_capacity_envelopes.csv',
    'oneG_vs_twoG_j.csv','second_generalist_regions.csv','structural_failure_reasons.csv',
    'primary_multihotspot_candidate.json','future_hotspot_points.json','module11a_gate.json','module11a_summary.md']

def main():
    freeze=read(OUT/'input_freeze.json');verify(freeze['protected_references'])
    for name in REQUIRED:assert (OUT/name).is_file(),name
    c,hw,roles,cal=inputs();checks=[]
    # Check the pruning relaxation independently, with one task and multiple
    # model choices, against direct deterministic convex optimization.
    for task in freeze['eligible_tasks']:
        x={m:int(task in c[m].supported_tasks) for m in c}
        options=[(c[m].service_time_ms[task]/1000,cal.capacity_factors[m][n]) for m,n in x.items() if n]
        s=np.array([p[0] for p in options]);g=np.array([p[1] for p in options])
        for fraction in [.1,.5,.9]:
            rate=fraction*sum(g/s)
            def f(r):return sum((r*s)/(1-r*s/g))/rate
            def jac(r):return s/(1-r*s/g)**2/rate
            result=minimize(f,fraction*g/s,jac=jac,bounds=[(0,(1-1e-9)*gg/ss) for ss,gg in options],
                constraints={'type':'eq','fun':lambda r:sum(r)-rate,'jac':lambda r:np.ones(len(r))},
                method='SLSQP',options={'maxiter':1000,'ftol':1e-12})
            assert result.success,result.message
            lb=lower_bound(x,{task:rate},c,cal)
            assert abs(lb-result.fun)<1e-8,(task,fraction,lb,result.fun)
            checks.append({'task':task,'fraction_of_taskwise_physical_capacity':fraction,
                'analytical_lower_bound_seconds':lb,'independent_convex_seconds':float(result.fun)})
    assert len(read(OUT/'capacity_witnesses.json')['rows'])==360
    assert len(read(OUT/'windows.json')['rows'])==120
    assert len(read(OUT/'latency_witnesses.json')['rows'])==500
    assert read(OUT/'cpu_validation.json')['status']=='PASS'
    assert read(OUT/'future_hotspot_points.json')['GPU_runs']==0
    save('completion_verification.json',{'status':'PASS','at_utc':now(),'protected_files_verified':len(freeze['protected_references']),
        'Module9_runtime_and_prior_Module10_including_invalid_archive_unchanged':True,
        'all_required_outputs_present':True,'lower_bound_independent_convex_checks':checks,
        'GPU_runs':0,'MPS_enabled':False,'new_profiles':0,'no_new_task_or_neighborhood':True,
        'execution_complete_scientific_gate_evidence_incomplete':True})
    outputs=[ref(p) for p in OUT.rglob('*') if p.is_file() and p.name!='output_manifest.json']
    code=[ref(p) for p in (ROOT/'module11').glob('*.py')]
    save('output_manifest.json',{'at_utc':now(),'outputs':outputs,'code':code,'new_GPU_runs':0})
    print(json.dumps({'completion':'PASS','protected_files':len(freeze['protected_references']),'output_files':len(outputs),'new_GPU_runs':0}),flush=True)

if __name__=='__main__':main()
