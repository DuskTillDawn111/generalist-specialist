"""Freeze domain/protocol before solving or observing new GPU utility."""
from .common import *
def main():
    assert (OUT/'history_sha256_before.json').exists()
    assert not (OUT/'frozen_problem_manifest.json').exists()
    src=ROOT/'results/module32b_vqa_aware_routing/selected_pool_catalog.json';full=read(src)
    S='openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1'
    OCR='baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt'
    small=['phi4mm',S,'roberta_sst2','convnext_tiny',OCR,'whisper-small']
    pools={}
    for name,models in [('SMALL6',small),('LARGE7',small+['qwen'])]:
        d=copy.deepcopy(full)
        for key in ['models','capacity_factors','memory_table_mib','evidence_bounds','model_evidence']:
            d[key]={m:d[key][m] for m in models}
        for m in models:
            maxn=2 if name=='LARGE7' or m==OCR else 1
            ns=[n for n in d['capacity_factors'][m] if int(n)<=maxn]
            for key in ['capacity_factors','memory_table_mib']:d[key][m]={n:d[key][m][n] for n in ns}
            d['evidence_bounds'][m]=max(map(int,ns))
        # Explicit finite benchmark path restriction, not a quality-policy edit.
        d['models']['whisper-small']['supported_tasks']=['lid']
        save('pools/'+name+'.json',d);pools[name]=ref(OUT/'pools'/ (name+'.json'))
    hist18=ROOT/'results/module18_cross_task/frozen_task_manifests.json'
    f18=read(hist18);low=next(s for s in f18['slots'] if s['run_id']=='text_low_rep1')['lambda_rps']
    hist32=ROOT/'results/module32b_vqa_aware_routing/frozen_gpu_manifest.json';f32=read(hist32)
    loads={'LOW18':dict(rates=low,source=ref(hist18),source_run='text_low_rep1'),
           'ASR32B':dict(rates=f32['slots'][0]['lambda_rps'],source=ref(hist32),source_run='A0')}
    space=dict(pools=pools,loads=loads,budget_mib=34567.5534,budget_source=ref(hist32),
        route_grid_denominator=2,route_rule='For every admitted task path, nonnegative integer units summing to 2; QPS = units*offered_QPS/2; absent paths zero. Enumerate every weak composition.',
        roles='At least one resident Generalist and one Specialist. Zero-demand residents are legal and count against memory.',
        path_restrictions={'whisper-small':'LID only, as in Module32B native task routing; ASR capability remains historically admitted but excluded from this restricted benchmark.'},
        hard_constraints=['historical quality/capability','six positive tasks covered','measured counts only','routing conservation and linkage','safe_factor*M<=budget','nominal rho<1','no incomplete-after-drain predicted cohort'],
        operational_08='Original V3 retains rho<=.8. 20A nominal eligibility is rho<1; its .8/.95 generation caps are search design, not rewritten feasibility.',
        completeness='Finite restricted model/count/path/grid space only; no continuous or full historical model-pool optimum claim.')
    save('candidate_space_definition.json',space)
    audit=dict(formal_V3=dict(entry='module9bc.frozen_v3.create',ranking='delta_J',routing='min J=sum_m D_m/(1-rho_m)/total_QPS',
        acceptance='GPU runtime NoiseAwareRuntimeObjective(1ms,.05): stability then measured mean/P95/memory. CPU dry_run uses planning transition, not a real U optimum.',
        U_optimized=False,source=ref(ROOT/'module9bc/frozen_v3.py')),
        utility_reference_algorithm=dict(entry='module20.optimizer.Generator + module20.search U ranking',objective='max original20A predicted U',
            promotion_status='20A UTILITY_MODEL_INSUFFICIENT; never relabel as promoted formal V3',
            sources=[ref(ROOT/'module20/optimizer.py'),ref(ROOT/'module20/search.py')]),
        unpromoted_extensions=[dict(entry='module25.optimizer.optimize',guard='MODULE25_PREDICTIVE_GATE_FAILED',executed=False)],
        grid_adapter='Change MILP variable units and integrality solely to impose frozen finite domain; objective, candidate schedule, ranking, caps, weights and predictor unchanged. Native unconstrained decisions separately reported.',
        V3_grid_handling='Native V3 is run unchanged. An off-grid native route is recorded as DOMAIN_MISMATCH, never silently rounded or treated as an in-space solution; its deployment with best frozen J grid route is an explicitly labelled baseline projection.')
    save('optimizer_objective_audit.json',audit)
    manifest=dict(module=33,frozen_unix_ns=time.time_ns(),space=ref(OUT/'candidate_space_definition.json'),
        objective_audit=ref(OUT/'optimizer_objective_audit.json'),utility_definition=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),
        utility_reference=ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),
        CPU_evaluator='Unchanged module20.optimizer.predict_scenarios, nominal phase=.37 horizon=120s; same interface for every candidate. Four scenarios counted.',
        J_evaluator='Unchanged deployment_core.routing.e2e_surrogate.evaluate_surrogate',
        GPU_protocol=dict(pool='SMALL6',load='ASR32B',all_domain_candidates=True,expected_feasible_candidates=8,
            development_seconds=120,warmup_seconds=30,drain_seconds=60,development_seed=330101,development_order_seed=330102,
            independent_pair_seconds=240,independent_seeds=[330201,330202,330203],
            pairs='Development observed maximum versus finite-grid20A choice AND formal native V3 grid-compatible choice; deduplicate identical arms; fresh worker PIDs per run. AB/BA/AB order. Reevaluate winners only, never refit optimizer.',
            uncertainty='Paired 30s arrival-block bootstrap 2000 reps plus paired independent-run t95 CI; microgaps whose CI includes zero are unresolved.',
            runtime='Original native DeploymentRuntime + SafeMonitorV2 timing wrapper, Module32F repaired CUDA/cache write confinement; no dynamic router is invoked or edited.',
            failure='Technical error: retain raw and NULL U, stop GPU campaign without replacing candidates or outcome retries. Scientifically unstable candidates: retain NULL U and continue remaining frozen candidates if clean release.',
            scope='Development empirical maximum in entire finite measured set; independent pair checks do not certify population/continuous/global optimality.'),
        runtime_benchmark_repetitions=3,no_algorithm_refit_after_GPU=True,no_module34=True,
        science_sources=[ref(src),ref(ROOT/'results/module14f_memory_ocr_x3/heavy_empirical_x3_reconciled.json'),ref(ROOT/'results/module8b_quality_pool_finalization/quality_policy.json')],
        source_hashes=[ref(p) for p in (OUT/'m33').glob('*.py')])
    save('frozen_problem_manifest.json',manifest)
    print('FROZEN',[(k,v['rates']) for k,v in loads.items()],flush=True)
if __name__=='__main__':main()
