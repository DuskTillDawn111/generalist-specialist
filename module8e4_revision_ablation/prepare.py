"""Freeze first, verify endpoints, then replay only strictly cached observations."""
from dataclasses import asdict
from collections import Counter
import json
from pathlib import Path
from .common import *
from .variants import FLAGS, flags_dict, make, config
from deployment_core import Workload
from deployment_search.runtime import RuntimeEvaluator
from deployment_search.result import jsonable
from deployment_search.tests.fixtures import forbid_enumeration
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator, summarize, aggregate
from joint_deploy.measurement.schema import MeasurementConfig


class MissingState(Exception):
    def __init__(self, record):
        self.record = record
        super().__init__('No frozen compatible raw cache for independently selected state')


class CachedEvaluator(RuntimeEvaluator):
    def __init__(self, w, catalog, method):
        self.w, self.catalog, self.method = w, catalog, method
        self.donors = old_donors(); self.calls = []
        self.identifier = ReplicaRuntimeEvaluator(OUT/'preflight'/w['id']/method, catalog, CONTEXT,
            read(F/'method_comparison_freeze.json')['measurement_settings'])

    def evaluate(self, deployment, workload, routing):
        ident = self.identifier.identity(deployment, workload, routing)
        reps = []; missing = []; hashes = []
        for i, seed in enumerate(read(F/'method_comparison_freeze.json')['search_seeds'], 1):
            cfg = MeasurementConfig(ident['deployment'], ident['workload'], ident['routing'],
                random_seed=seed, **read(F/'method_comparison_freeze.json')['measurement_settings']).as_dict()
            h = digest(identity(cfg, digest(self.w))); hashes.append(h)
            if h not in self.donors: missing.append(seed); continue
            donor = self.donors[h]
            assert donor['identity'] == identity(cfg, digest(self.w))
            assert file_hash(donor['raw_path']) == donor['raw_sha256']
            rep = summarize(read(donor['raw_path']), workload, self.catalog, donor['outcome'], donor['raw_path'])
            rep.update(repetition=i, random_seed=seed); reps.append(rep)
        record = {'workload': self.w['id'], 'method': self.method, 'index': len(self.calls),
            'deployment': deployment, 'routing': jsonable(routing), 'routing_sha256': digest(jsonable(routing)),
            'measurement_identity_hashes': hashes, 'missing_seeds': missing}
        if missing: raise MissingState(record)
        self.calls.append(record)
        return aggregate(reps, digest(ident), OUT/'preflight'/'virtual_aggregate.json', ident['deployment'])


def semantic(result):
    """Complete search decisions; ignore only audit metadata and cache file labels."""
    j = result.to_dict()
    def runtime(r):
        return None if r is None else {k:r[k] for k in ['stability_state_v2','weighted_mean_e2e','weighted_p95_e2e','completion_ratio','throughput']}
    iterations = []
    for tr in j['iterations']:
        cs = []
        for c in tr['candidates']:
            mu = c['mutation']
            cs.append({'mutation':mu, 'canonical':c['canonical'], 'delta_z':c['delta_z'],
                'memory_mib':c['memory_mib'], 'rank':c['rank'], 'selected_top_b':c['selected_top_b'],
                'status':c['status'], 'reason':c['reason'], 'runtime':runtime(c['runtime'])})
        iterations.append({k:tr[k] for k in ['iteration','deployment','ranking','selected_top_b','accepted','num_runtime_candidates','no_improvement_iterations']}
            | {'canonical':tr['canonical'],'current_runtime':runtime(tr['current_runtime']), 'candidates':cs})
    seed = {k:v for k,v in j['seed'].items() if k != 'solver_time_ms'}
    return ({k:j[k] for k in ['status','deployment','canonical','runtime_evaluation_count','search_iterations','milp_solve_count','visited_deployments','rejected_mutations']}
        | {'seed':seed,'runtime':runtime(j['runtime']),'seed_runtime':runtime(j['seed_runtime']),'iterations':iterations})


def execute_cpu(method, w, catalog, hardware, roles, cal, endpoint=False):
    evaluator = CachedEvaluator(w, catalog, method)
    snapshots = []
    obj = make(method, evaluator, lambda r:snapshots.append(r.to_dict()), flag_endpoint=endpoint)
    try:
        result = obj.search(Workload(w['lambda_rps']),catalog,hardware,config(method),roles=roles,
            capacity_mode='empirical',memory_mode='empirical',calibration=cal)
        return result, evaluator, snapshots, None
    except MissingState as e:
        return None, evaluator, snapshots, e.record


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    verify_inputs()
    protocol = read(F/'method_comparison_freeze.json')
    spec = {'module':'8E-4','frozen_at_utc':utc(),'workload_role':'POST-VALIDATION ABLATION SET',
        'workloads':ref(F/'heldout_workloads.json'),'workload_hashes':ref(F/'heldout_workload_hashes.json'),
        'flags':{m:flags_dict(m) for m in FLAGS},'predictor':None,'search_config':protocol['search_config'],
        'search_repetitions':3,'search_seeds':protocol['search_seeds'], 'confirmation_order':protocol['confirmation_order'],
        'confirmation_seeds':protocol['confirmation_seeds'],'seed_confirmations':2,'final_confirmations':2,
        'endpoint_rule':'M00 direct official v1, M11 direct official v2; no GPU reruns. Flagged endpoints must have identical complete semantic decisions.',
        'implementation':'Exact immutable native RevisedGuidedSearch.search code object, instance-local diversified_order binding; original neighborhood or native targeted neighborhood. No copied or edited search loop.',
        'R1_OFF':'Original x±1 neighborhood only; no SWAP/closure.',
        'R2_OFF':'Native deployment_search.scoring.rank_candidates; global exact delta_z Top-2, original stable model tie order.',
        'R2_ON':'Native v2 diversified_order; with R1 off, Class R contains only original REMOVE/SCALE_DOWN.',
        'reuse':'Full 8E-3 measurement identity, phase/role and seed partition unchanged. Earliest compatible physical donor. No future cached metric exposed before own candidate commitment.',
        'schedule':'Frozen 8E-2 order filtered to M1/M2, remapped to M10/M01; original iteration/rank/confirmation slots; no B2 measurement. Independent both-round commitments precede value release.',
        'failure_policy':protocol['failure_policy'],'automatic_retries':0,
        'claim_policy':{
            'factor_contrasts':{'R1':['M10_vs_M00','M11_vs_M01'],'R2':['M01_vs_M00','M11_vs_M10']},
            'factor_ordered_rules':[
                {'if':'incomplete or protocol invalid','status':'INSUFFICIENT'},
                {'if':'LOSS>WIN or any stability regression or any steady mean regression>=20%','status':'NOT_SUPPORTED'},
                {'if':'WIN>=2 and LOSS==0 and positive mechanism chain present','status':'SUPPORTED'},
                {'if':'WIN>LOSS and WIN>=1','status':'PARTIAL'},
                {'if':'otherwise complete six-workload contrast set','status':'NOT_SUPPORTED'}],
            'interaction_rules':[
                {'if':'incomplete or protocol invalid','status':'INSUFFICIENT'},
                {'if':'joint-only WIN and exact generated-rank-missed-selected-accepted chain on >=2 workloads','status':'SUPPORTED'},
                {'if':'joint-only WIN or exact mechanism chain on >=1 workload','status':'PARTIAL'},
                {'if':'otherwise complete six-workload set','status':'NOT_SUPPORTED'}],
            'joint_only':'M11_vs_M00 WIN while neither M10_vs_M00 nor M01_vs_M00 is WIN on same workload',
            'chain':'Accepted beneficial M11 replacement exact deployment+routing generated by M10 with rank>2 and not evaluated, absent from M01 generation.',
            'necessity_scope':'Only necessity for observed full-v2 wins under these frozen workloads and budget; no universal algorithm necessity claim.',
            'static_freeze_readiness':'Complete valid campaign, M11 retains 2/4/0 vs M00, no M11 LOSS versus either ablation, at least one supported factor and supported/partial interaction. Freeze current static method only; routing limitation remains.',
            'dynamic_readiness':'Same static readiness permits a separate Module 8F protocol design; this task executes no dynamic work.',
            'no_significance_claim':True},
        'ROUTING_REVISION_PENDING':True,'new_revision_allowed':False}
    immutable(OUT/'ablation_spec.json',spec)
    Path(str(OUT/'ablation_spec.json')+'.sha256').write_text(file_hash(OUT/'ablation_spec.json')+'\n')


def regression_and_preflight():
    assert file_hash(OUT/'ablation_spec.json') == Path(str(OUT/'ablation_spec.json')+'.sha256').read_text().strip()
    catalog,hardware,roles,cal = inputs(); workloads = read(F/'heldout_workloads.json')['workloads']
    flags = {'status':'PASS','flags':{m:flags_dict(m) for m in FLAGS},'same_config':all(asdict(config(m))==asdict(config('M00')) for m in FLAGS),
        'native_loop_code_unchanged':True,'only_global_binding_difference':'diversified_order','GPU_runs':0}
    assert flags['same_config']; save_json(OUT/'feature_flag_validation.json',flags)
    checks=[]; preflights=[]
    with forbid_enumeration() as guards:
        for w in workloads:
            for method, source in [('M00','M1'),('M11','M2')]:
                original, _, _, missing = execute_cpu(method,w,catalog,hardware,roles,cal)
                flagged, _, _, fm = execute_cpu(method,w,catalog,hardware,roles,cal,True)
                assert missing is None and fm is None
                assert semantic(original)==semantic(flagged), (w['id'],method,'flagged endpoint differs')
                old = read(E3/('m1_search_traces' if source=='M1' else 'm2_search_traces')/w['id']/'search_complete.json')
                from types import SimpleNamespace
                class Serialized:
                    def to_dict(self):return old
                assert semantic(original)==semantic(Serialized()), (w['id'],method,'8E-3 trace differs')
                checks.append({'workload':w['id'],'method':method,'full_candidate_generation_ranking_TopB_routing_budget_stopping_equal':True,
                    'semantic_sha256':digest(semantic(original)), 'original_trace':ref(E3/('m1_search_traces' if source=='M1' else 'm2_search_traces')/w['id']/'search_complete.json')})
            for method in NEW:
                result, ev, snapshots, missing = execute_cpu(method,w,catalog,hardware,roles,cal)
                last = snapshots[-1] if snapshots else None
                save_json(OUT/'preflight'/w['id']/method/'cached_only_trace.json',result.to_dict() if result else last)
                cached=len(ev.calls); outstanding=6-cached if missing else 0
                confirmation_reuse=2
                if result:
                    ident=ev.identifier.identity(result.deployment,Workload(w['lambda_rps']),result.canonical.routing)
                    for seed in read(F/'method_comparison_freeze.json')['confirmation_seeds']:
                        cfg=MeasurementConfig(ident['deployment'],ident['workload'],ident['routing'],random_seed=seed,
                            **read(F/'method_comparison_freeze.json')['measurement_settings']).as_dict()
                        confirmation_reuse+=digest(identity(cfg,digest(w),'confirmation_final')) in ev.donors
                preflights.append({'workload':w['id'],'method':method,'status':'CACHE_COMPLETE' if result else 'FIRST_MISSING_STATE',
                    'reusable_search_states_in_selected_prefix':cached,'new_search_state_upper_bound':outstanding,
                    'first_missing_state':missing,'future_candidates_not_predicted':bool(missing),
                    'guaranteed_confirmation_repetition_reuse':confirmation_reuse,
                    'new_confirmation_run_upper_bound':4-confirmation_reuse,
                    'worst_case_new_timed_runs':outstanding*3+4-confirmation_reuse,
                    'termination_reason':result.status if result else None})
        assert sum(g.call_count for g in guards)==0
    save_json(OUT/'regression_equivalence.json',{'status':'PASS','checks':checks,'GPU_runs':0,
        'scope':'Exact six real-workload trajectories using frozen raw cache, plus flag endpoints against direct native methods.'})
    reusable=sum(r['reusable_search_states_in_selected_prefix'] for r in preflights)
    missing=sum(r['first_missing_state'] is not None for r in preflights)
    cap=sum(r['worst_case_new_timed_runs'] for r in preflights)
    reuse={'status':'PASS','preflight_at_utc':utc(),'method_role':'POST-VALIDATION ABLATION SET',
        'M00_M11_reusable_workload_method_results':12,'M00_M11_new_runs_allowed':0,
        'expected_reusable_search_states':reusable,'known_missing_frontier_states':missing,
        'missing_physical_states_upper_bound':sum(r['new_search_state_upper_bound'] for r in preflights),
        'worst_case_new_timed_runs':cap,'workloads_methods':preflights,
        'estimate_scope':'Cache-only replay stops at first independently selected missing state; no fabricated runtime for future decisions. Upper bound assumes remaining search slots and non-guaranteed final confirmations miss cache.',
        'existing_E3_raw_runs':181,'cache_complete_trajectory_count':sum(r['status']=='CACHE_COMPLETE' for r in preflights),
        'preflight_GPU_runs':0}
    save_json(OUT/'reuse_audit.json',reuse)
    print(json.dumps({k:reuse[k] for k in ['expected_reusable_search_states','known_missing_frontier_states','missing_physical_states_upper_bound','worst_case_new_timed_runs','cache_complete_trajectory_count']}),flush=True)


def main():
    if not (OUT/'ablation_spec.json').exists():freeze()
    else:verify_inputs()
    regression_and_preflight()


if __name__=='__main__':main()
