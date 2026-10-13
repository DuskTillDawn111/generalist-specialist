"""Conditional, fixed Proposed-deployment routing crossover. No search."""
import argparse
import os
from dataclasses import asdict
from pathlib import Path

from module8d3_diagnosis import OUT, D2, C, F, CAT, read, ref, counts, flat, rates, verify_inputs, validate_raw, save_json, file_hash, Workload


def gate_freeze():
    row={'gate':'Part G: only if A/B/C cannot distinguish worker-presence changes from routing changes, measure fixed Proposed deployment with deterministic greedy-compatible routing.',
        'reasoning_condition':'C and B change whether Phi is deployed AND receives VQA traffic; D is missing. D is needed to isolate routing at fixed Proposed deployment, avoiding attribution of a route-mediated gain solely to deleting a model.',
        'speech_condition':'If A-B and B-C point differences have opposite signs, the two conditional effects compensate; D is needed to check canonical versus greedy routing on Proposed deployment as well.',
        'scope':['reasoning_vqa_heavy','speech_heavy'],'maximum_new_states':2,'repetitions':3,'maximum_new_runs':6,
        'existing_plan_unchanged':ref(OUT/'crossover_plan.json'),'no_method_or_workload_patch':True,'implementation':ref(Path(__file__).resolve()),'threshold':'No new numeric gate; structural ambiguity and opposite contrast signs only'}
    path=OUT/'crossover_stage2_gate_policy.json'
    if path.exists():assert read(path)==row
    else:save_json(path,row);(OUT/(path.name+'.sha256')).write_text(file_hash(path)+'\n')


def prepare():
    gate_freeze();verify_inputs();stage1=read(OUT/'crossover_results.json');assert stage1['status']=='COMPLETE'
    vals=read(OUT/'routing_crossover_audit.json')['variants'];planned=[]
    for n in ('reasoning_vqa_heavy','speech_heavy'):
        a=next(v for v in vals if v['workload']==n and v['variant']=='A')
        c=next(v for v in vals if v['workload']==n and v['variant']=='C')
        d=next(v for v in vals if v['workload']==n and v['variant']=='D')
        b=next(v for v in stage1['new_states'] if v['workload']==n)
        ab=a['evidence']['runtime']['weighted_mean_e2e']-b['runtime']['weighted_mean_e2e']
        bc=b['runtime']['weighted_mean_e2e']-c['evidence']['runtime']['weighted_mean_e2e']
        if not d['evidence'] and (n=='reasoning_vqa_heavy' or ab*bc<0):
            planned.append(dict(d,reason='Separate VQA route choice from Phi presence' if n=='reasoning_vqa_heavy' else 'Opposite A-B and B-C contrasts require fixed-Proposed routing check',stage1_mean_contrasts={'A_minus_B':ab,'B_minus_C':bc}))
    path=OUT/'crossover_plan_stage2.json'
    plan={'status':'FROZEN' if planned else 'NOT_RUN','states':planned,'settings':read(OUT/'crossover_plan.json')['settings'],'seeds':[7101,7202,7303],'repetitions':3,'maximum_new_timed_runs':3*len(planned),'gate_policy':ref(OUT/'crossover_stage2_gate_policy.json'),'stage1_evidence':ref(OUT/'crossover_results.json'),'implementation':ref(Path(__file__).resolve()),'no_search_no_ranking':True}
    if path.exists():assert read(path)==plan
    else:save_json(path,plan);(OUT/(path.name+'.sha256')).write_text(file_hash(path)+'\n')
    print({'stage2_gate': 'PASS' if planned else 'NOT_NEEDED', 'states':[v['workload'] for v in planned],'new_run_cap':3*len(planned)},flush=True)


def run():
    verify_inputs();path=OUT/'crossover_plan_stage2.json';plan=read(path)
    assert file_hash(path)==(OUT/(path.name+'.sha256')).read_text().strip()
    assert file_hash(plan['implementation']['path'])==plan['implementation']['sha256']
    assert file_hash(plan['stage1_evidence']['path'])==plan['stage1_evidence']['sha256']
    for k,v in read(F/'baseline_definitions.json')['measurement_protocol']['execution_environment_overrides'].items():os.environ[k]=v
    from replica_scaling.telemetry import snapshot
    gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128
    from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator
    results=[]
    for state in plan['states']:
        verify_inputs();n=state['workload'];ev=ReplicaRuntimeEvaluator(OUT/'crossover_raw_D'/n,CAT,C/n/'runtime_context.json',plan['settings'],plan['seeds'],1)
        x=counts(state['deployment']);route=flat(state['identity']['routing']);w=Workload(rates(n));assert ev.identity(x,w,route)==state['identity']
        rt=ev.evaluate(x,w,route)
        for p,seed in zip(rt.raw_result_paths,plan['seeds']):validate_raw(p,state['identity'],seed,n)
        results.append({'workload':n,'variant':'D','identity':state['identity'],'runtime':asdict(rt),'raw_sources':[ref(p) for p in rt.raw_result_paths],'measurement_source':'newly_measured','search':False})
        save_json(OUT/'crossover_stage2_results.json',{'status':'RUNNING','plan':ref(path),'new_states':results,'new_timed_runs':3*len(results)})
    save_json(OUT/'crossover_stage2_results.json',{'status':'COMPLETE' if results else 'NOT_RUN','plan':ref(path),'new_states':results,'new_timed_runs':3*len(results),'no_search_no_ranking':True})
    print({'stage2_complete':True,'new_timed_runs':3*len(results)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze-gate','prepare','run'));a=p.parse_args()
    {'freeze-gate':gate_freeze,'prepare':prepare,'run':run}[a.action]()
