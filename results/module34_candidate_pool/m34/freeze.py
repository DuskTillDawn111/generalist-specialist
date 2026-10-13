"""Only candidate membership changes across A/B/C; freeze before predictions."""
from .common import *
def main():
    assert (OUT/'history_sha256_before.json').exists()
    assert not (OUT/'frozen_problem_manifest.json').exists()
    fullpath=ROOT/'results/module14f_memory_ocr_x3/heavy_empirical_x3_reconciled.json'
    full=read(fullpath);addedpath=ROOT/'results/module32b_vqa_aware_routing/selected_pool_catalog.json';added=read(addedpath)
    large='openai/whisper-large-v3@06f233fe06e710322aca913c1bc4249a0d71fce1'
    medium='openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e'
    ocr='baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt'
    bart='textattack/facebook-bart-large-SST-2@8a4baa9662afa056a97a2e9441ae665a06504979'
    members={'A':['qwen','minicpm','phi4mm'],'B':['qwen','minicpm','phi4mm',large,medium],
             'C':['qwen','minicpm','phi4mm',large,medium,'roberta_sst2','convnext_tiny',ocr,'whisper-small',bart]}
    all_data=copy.deepcopy(full)
    for key in ['models','capacity_factors','memory_table_mib','evidence_bounds','model_evidence']:
        all_data[key]['whisper-small']=copy.deepcopy(added[key]['whisper-small'])
    pools={}
    for name,models in members.items():
        d=copy.deepcopy(all_data)
        for key in ['models','capacity_factors','memory_table_mib','evidence_bounds','model_evidence']:
            d[key]={m:d[key][m] for m in models}
        for m in models:
            for key in ['capacity_factors','memory_table_mib']:d[key][m]={n:v for n,v in d[key][m].items() if int(n)<=1}
            d['evidence_bounds'][m]=1
        save(f'pools/{name}.json',d);pools[name]=ref(OUT/f'pools/{name}.json')
    assert set(members['A'])<set(members['B'])<set(members['C'])
    l18path=ROOT/'results/module18_cross_task/frozen_task_manifests.json';l18=read(l18path)
    low=next(s['lambda_rps'] for s in l18['slots'] if s['run_id']=='text_low_rep1')
    mid=next(s['lambda_rps'] for s in l18['slots'] if s['run_id']=='text_mid_rep1')
    text10=copy.deepcopy(mid);text10['text']=mid['text']/2
    m33=read(ROOT/'results/module33_mixed_oracle/candidate_space_definition.json')
    scenes={'LOW18':dict(rates=low,source=ref(l18path),derivation='Original text_low_rep1'),
            'ASR32B':dict(rates=m33['loads']['ASR32B']['rates'],source=m33['loads']['ASR32B']['source'],derivation='Original A0; directly comparable to Module33 input/arrival distribution'),
            'TEXT_HALF_MID18':dict(rates=text10,source=ref(l18path),derivation='text_mid_rep1 target20QPS divided by2; other5 tasks unchanged. Predeclared moderate text demand avoids trivial single-replica G infeasibility at original20QPS.')}
    assert all(q>0 for s in scenes.values() for q in s['rates'].values())
    policy=ROOT/'results/module8b_quality_pool_finalization/quality_policy.json'
    space=dict(pools=pools,members=members,scenarios=scenes,budget_mib=49140.,
        budget_source=ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),
        budget_reason='Original20A full-GPU budget49140MiB; measured physical total identical. Same budget in all3 pools; minimum pureG cover safe~42059MiB exceeds old33 budget.',
        route_grid_denominator=1,route_rule='All endpoint task routes; one resident admitted model per task. No continuous splitting. Enumerate every task-path assignment.',
        replica_rule='Controlled binary0/1 residency for every model across all pools; original measured one-replica g/M/s0 unchanged. Does not certify optimality over higher counts.',
        constraints=['original task quality admission','six positive tasks covered','route linkage/conservation','same endpoint routing domain','safe memory<=49140MiB','rho<=.8','nominal complete120s cohort+60s drain'],
        benchmark_scope='Nested candidate-set expansion with fixed inputs, budget, load per scenario, weights, normalization, counts and route domain. PureS infeasible without VQA path; no synthetic Specialist VQA.',
        supported_paths_policy='Use all original admitted catalog paths. Whisper-small ASR restored from original8B admission score0.042180839129459274<=0.042181; Module33 LID-only restriction was a benchmark restriction, never a failed quality path.')
    save('candidate_space_definition.json',space)
    save('frozen_problem_manifest.json',dict(module=34,frozen_unix_ns=time.time_ns(),space=ref(OUT/'candidate_space_definition.json'),
        utility_definition=ref(ROOT/'results/module20_utility_optimizer/utility_definition.json'),
        utility_reference=ref(ROOT/'results/module20_utility_optimizer/utility_reference_manifest.json'),quality_policy=ref(policy),
        historical_profiles=[ref(fullpath),ref(addedpath)],
        historical33=[ref(ROOT/'results/module33_mixed_oracle'/n) for n in ['module33_gate.json','cpu_oracles.json','development_observations.json','pair_observations.json','formal_route_domain_audit.json']],
        methods=['CPU_U_ORACLE','CPU_J_ORACLE','V3_NATIVE_CONTINUOUS_PRESERVED','V3_J_GRID_CONTROL','20A_U_GRID','20A_NATIVE_CONTINUOUS_PRESERVED','FASTEST_PATH_STATIC','MIN_MEMORY_COVER_FASTEST','G_ONLY_CLASS_ORACLE','S_ONLY_CLASS_ORACLE'],
        formal_objective='J for route/rank; unchanged original noise-aware runtime mean/P95/memory acceptance, not a global scalarJ or U objective.',
        V3_domain_control='Independent34 adapter restricting both private V3 ranker and serving J routing to exact endpoint grid. Original candidate generation, top2 selection, stopping, noise acceptance unchanged. Original continuous V3 also retained; never relabel adapter as untouched native.',
        U_method='Existing20A generation schedule and U scoring; finite route units/integrality adapter only. No new U optimizer or outcome-based retuning.',
        predictor='Unchanged20A predict_scenarios,120s phase.37; identical per-candidate evaluation plus commonrho.8 guard',
        runtime_repetitions=3,
        GPU_budget=dict(max_distinct_development_plans=6,development_seconds=120,independent_seconds=240,
            development_seed=340101,independent_seeds=[340201,340202,340203],
            selection='After CPU results, choose one feasible B scenario with maximal heuristic U regret (prefer G-only Oracle if available), and C ASR32B Oracle versus fastest-service heuristic; deduplicate. GPU plans/seeds frozen before GPU observation. No new scenario scan after GPU.',
            full_GPU_oracle_claim=False,scope='Independent key-competitor comparisons; only fully measured predeclared small restricted competition may get an empirical best label. No oracle over all CPU candidates.',
            historical_reuse='Recompute original33 raw U with unchanged20A formula at new fixedbudget49140, retain original raw/Gate/U. Development reuse is not independent new-pool ranking validation; endpoint/replica/path membership checked.'),
        deadline_local='2026-10-16 Asia/Shanghai',priority='Core CPU9 cells+few independent GPU contrasts, then paper; no automaticModule35',no_algorithm_refit_after_GPU=True,
        source_hashes=[ref(p) for p in (OUT/'m34').glob('*.py')]))
    print('FROZEN_POOLS', {k:len(v) for k,v in members.items()},flush=True)
if __name__=='__main__':main()
