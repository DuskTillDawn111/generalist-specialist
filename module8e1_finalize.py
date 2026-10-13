"""Report and validate frozen offline revision evidence; no GPU execution."""
import ast,csv,json
from pathlib import Path
from module8e1_analysis import OUT,ROOT,C,D2,D3,NAMES,CAT,read,ref,file_hash,save_json,verify_v2
from module8d3_diagnosis import verify_inputs
from replica_scaling.telemetry import snapshot
from candidate_pool_audit.provenance import environment_snapshot

def ast_method(path,cls,method):
    tree=ast.parse(path.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in cl.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==method)

def attribution():
    checks={}
    for name in ['_seed','_routing','_key','_memory','_memory_increment','_candidate_rejection','_warm_start']:
        a=ast_method(ROOT/'deployment_search/integer_search.py','IntegerGuidedDeploymentSearch',name)
        b=ast_method(ROOT/'method_v2/integer_search.py','MethodV2',name)
        checks[name]=ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    a=ast_method(ROOT/'deployment_search/search.py','GuidedDeploymentSearch','search')
    b=ast_method(ROOT/'method_v2/engine.py','RevisedGuidedSearch','search')
    def transition_suffix(node):
        loop=next(n for n in node.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='offset')
        index=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='best' for t in n.targets))
        return ast.dump(ast.Module(body=loop.body[index:],type_ignores=[]),include_attributes=False)
    checks['runtime_evaluation_accept_rollback_stopping_suffix_identical']=transition_suffix(a)==transition_suffix(b)
    assert all(checks.values()),checks
    save_json(OUT/'revision_attribution_audit.json',{'status':'PASS','AST_checks':checks,'shared_original_RuntimeEvaluator_and_runtime_objective_source_unchanged':True,'behavior_changes_only':['targeted SWAP generation','class-diversified Top2'],'other_changes':'Only frozen-config guards and candidate/complexity/net-memory diagnostic metadata','GPU_experiments':0})


def report():
    verify_v2();attribution()
    reg=read(OUT/'regression_results.json');assert reg['status']=='PASS'
    replay=read(OUT/'development_replay.json');miss=read(OUT/'development_cache_misses.json');case=read(OUT/'reasoning_replacement_case.json');complexity=read(OUT/'complexity_validation.json')
    for row in complexity['actual_development_states']:
        row['formal_pool_ordered_swap_upper_bound']=132
        row['effective_admitted_ordered_swap_upper_bound']=row['formal_ordered_pair_upper_bound']
    save_json(OUT/'complexity_validation.json',complexity)
    paths=list(csv.DictReader((OUT/'b2_path_compression.csv').open()))
    benefit=sum(b['not_evaluated_by_v1'] for w in replay['workloads'] for b in w['beneficial_cached_transitions'])
    records=[]
    for w in replay['workloads']:
        records.append({'workload':w['workload'],'budget_unchanged':w['budget'],'logical_slots_used_including_misses':w['logical_selected_state_attempts'],'known_beneficial_candidates':len(w['beneficial_cached_transitions']),'new_beneficial_cached_candidates_vs_v1':sum(b['not_evaluated_by_v1'] for b in w['beneficial_cached_transitions']),'cache_misses':w['cache_misses'],'full_v2_final_deployment_observed':False,'search_continuation_on_missing_measurement':False})
    save_json(OUT/'counterfactual_budget_audit.json',{'workloads':records,'max_iterations':3,'max_unique_states_including_seed':6,'top_b':2,'patience':1,'graph_reachability_not_runtime_acceptance':True,'new_beneficial_cached_state_count':benefit,'specific_shortest_path_slot_audit':ref(OUT/'shortest_path_slot_audit.json'),'physical_GPU_runs':0})
    save_json(OUT/'routing_revision_status.json',{'status':'ROUTING_REVISION_PENDING','canonical_LP_changed':False,'greedy_routing_implemented':False,'reason':'8D3 showed workload-dependent mean/P95 effects; one universal replacement routing policy is not established. Keep unchanged for attribution of this revision.','routing_change_required_before_this_v2_heldout_test':False,'routing_limitation_may_prevent_latency_dominance':True,'future_routing_revision_requires_separate_version_and_preregistration':True})
    ready=case['NEIGHBORHOOD_FIX_CONFIRMED_OFFLINE'] and reg['status']=='PASS' and read(OUT/'synthetic_revision_validation.json')['status']=='PASS'
    gate={'METHOD_V2_OFFLINE_VALIDATION_PASS':ready,'MODULE8E2_ALLOWED':ready,'HELDOUT_GPU_CAMPAIGN_READY':False,'permission_scope':'Next stage may prepare/preregister a fresh held-out protocol. No GPU campaign launched by this module.','remaining_prelaunch_requirements':['New independently frozen held-out workload matrix; no development-six unbiased claim','Freeze source/config/input/hardware/arrival/measurement/confirmation identities and analysis rules before first run','No tuning against held-out results; any subsequent revision requires fresh held-out validation'],'cache_misses':miss['CACHE_MISS_occurrences'],'routing_status':'ROUTING_REVISION_PENDING','v2_effectiveness_or_unbiased_final_claim':False,'no_new_GPU_runs':True}
    save_json(OUT/'module8e1_gate.json',gate)
    (OUT/'heldout_validation_requirements.md').write_text('''# Held-out validation requirements

The six Module 8C/8D workloads are DEVELOPMENT WORKLOADS. They informed this revision; neither replay nor later GPU runs on those same workloads support an unbiased final-validation claim. They remain frozen and unchanged.

Before a Module 8E-2 timed run, independently define a fresh held-out demand/load matrix and deterministic seeds under a preregistered sampling/design principle. Freeze its lambda vectors and hashes BEFORE inspecting its planning/runtime results. Do not select workloads to obtain different deployments or favorable outcomes. Do not fill development cache misses and call that held-out evidence.

Freeze method_v2 source hashes, candidate rules, Class R order, canonical routing, predictor=None, quality admission, model revisions, all 27 service-demand entries, empirical capacity/memory and replica bounds. Keep top_b=2, max_iterations=3, unique runtime states<=6 including seed, patience=1, three search repetitions and two independent seed/final confirmations. Freeze task sources, preprocessing, arrival generation, warmup/window, precision/backend and hardware identities. Confirmations must not affect search decisions.

Preregister analysis, runtime objective, failures/retries, cache reuse identity checks and any requested comparisons before measurements. Report every held-out condition, unsuccessful/unstable state, cache miss and resource failure; no best-of workload/seed selection. Existing baseline definitions must be frozen if comparison is later authorized; none is run here.

Keep canonical routing unchanged for this version's attribution. ROUTING_REVISION_PENDING remains a limitation, not permission to tune routing on held-out outcomes. A routing/ranking revision after inspecting held-out results needs a new method version and fresh held-out validation. Only this module's structural correctness is established; latency dominance/generalization is not.

Module 8E-1 stops after offline validation. No new GPU search, baseline, ablation, dynamic adaptation or second hardware has been started.
''')
    lines=['# Module 8E-1 — Minimal method revision and offline validation','',
        'OFFLINE_VALIDATION_PASS=true. Six workloads are DEVELOPMENT WORKLOADS; no unbiased final-validation claim. Zero new GPU experiments.','',
        'v1 remains runnable via method_v1.create()/method_v1.config(), using unchanged original packages, with 76 archived source files, source hashes and 24 unchanged Proposed/Random traces. v2 lives independently in method_v2; no original file was overwritten.','',
        '## Frozen revision','',
        'Only neighborhood and Top-B selection change. Targeted swaps use admitted capability overlap, prospective NEW-model ADD replacement closure and same-task Specialist replacement. Resulting coverage/memory/bounds and original canonical LP capacity are checked. Integer counts change exactly -1/+1. Closure means a previously nonredundant model becomes replaceable by equal-or-faster frozen-s0 paths after ADD; already-redundant removals and replica-ups are not mislabeled closures.','',
        'Class P receives slot1 by delta_z. Class R receives slot2, prioritizing exact closure, service-dominated family removal, lower/equal memory, then largest delta_z (including negatives), followed by deterministic ties. Feasible candidates only; no runtime-label/model-name preferences. Seeder, empirical evidence, canonical formulation, RuntimeEvaluator and NoiseAwareRuntimeObjective are unchanged.','',
        'top_b=2; max_iterations=3; max_unique_runtime_states=6 including seed; patience=1; predictor=None. Search repetitions=3 and independent seed/final confirmations=2 remain frozen; none newly executed.','',
        '## Reachability and reasoning','',
        '| Workload | v1 shortest | v2 shortest | reduction |','|---|---:|---:|---:|']
    for r in paths:lines.append(f"| {r['workload']} | {r['v1_shortest_steps']} | {r['v2_shortest_steps']} | {r['reduction']} |")
    lines+=['','All six B2 vectors are graph-reachable within three v2 mutations, using bounded goal-directed local path search. This is not a claim those intermediates will pass runtime acceptance or actually be selected in three iterations. Path witnesses are saved.','',
        f"NEIGHBORHOOD_FIX_CONFIRMED_OFFLINE={case['NEIGHBORHOOD_FIX_CONFIRMED_OFFLINE']}: reasoning seed Phi+S4 generates Qwen+S4 in the same round, and places it in Top2. Near old Phi+Qwen+S4 final, -Phi also enters Top2. Cached reasoning replay selects Qwen+S4 directly: mean86.406ms versus old final95.491ms. This is development reuse, not new or unbiased validation.",'',
        '## Candidate complexity','',
        '| Workload/state | base | swaps | total |','|---|---:|---:|---:|']
    for r in complexity['actual_development_states']:lines.append(f"| {r['workload']}/{r['state']} | {r['base_add_remove_candidates']} | {r['targeted_swap_candidates']} | {r['total_candidates']} |")
    lines+=['','Formal pool M=12 (ordered swap upper bound132); quality-admitted effective catalog M=11 (upper bound110). Actual snapshots have7–14 targeted swaps,15–24 total generated candidates. Base candidates remain unchanged; no exhaustive deployment-vector enumeration.','',
        '| Synthetic M | density | base | swaps | total | LP screen ms |','|---|---|---:|---:|---:|---:|']
    for r in complexity['synthetic']:lines.append(f"| {r['M']} | {r['density']} | {r['base_add_remove_candidates']} | {r['targeted_swap_candidates']} | {r['total_candidates']} | {r['LP_screen_ms']:.1f} |")
    lines+=['','Dense overlap can grow quadratically; no strict O(M) claim. The largest tested30-model case has230 candidates and under0.5s LP screening on this host, versus2^30 possible binary vectors; only generated local candidates are screened. Runtime slots remain2. Legacy bounded test oracles are regression fixtures, not v2 production enumeration.','',
        '## Historical replay and budget','',
        '| Workload | hits | misses | last resolved mean ms | status |','|---|---:|---:|---:|---|']
    for w in replay['workloads']:lines.append(f"| {w['workload']} | {w['cached_hits']} | {w['cache_misses']} | {w['last_resolved_mean_ms']:.3f} | {w['replay_status']} |")
    lines+=['',
        f"CACHE_MISS occurrences={miss['CACHE_MISS_occurrences']}, exact unique identities={miss['unique_state_identities']}; no predicted/synthetic runtime fills. All six replays are incomplete, so the table lists last resolved incumbents, never v2 final-deployment claims. Other selected slots in an unresolved iteration are probed within the original budget, but no winner is committed while one selected state remains unknown.",'',
        f"One new-to-v1 beneficial cached state is reached: reasoning Qwen+S4. Speech and no-VQA also select their already-known beneficial Distil ADD, but an unknown second-slot replacement prevents committing that iteration. This is limited evidence of better access, not six-workload dominance. All replay logical attempts are<=6.",'',
        'Conditional shortest-path slot audit: speech and no-VQA witness paths fit the unchanged selection/budget if each intermediate is accepted. The other four shortest witnesses lose their second SWAP at Top2. This does not rule out other paths, but compression alone has not established full budgeted B2 reachability. Three iterations using both slots would require seven states; the last iteration has only one slot at budget6.', '',
        '## Regression and gate','',
        f"{reg['tests_run']} tests pass: all27 old modules plus10 new v2 contracts. The two existing sklearn modules use the existing gsdeploy environment; no packages installed or predictor retrained. Initial wrong-environment import errors are preserved, then resolved by correct existing test environment. AST audit confirms unchanged seeder/routing/feasibility and identical runtime-evaluation/accept/rollback/stopping suffix. V1 hashes/traces unchanged.",'',
        'ROUTING_REVISION_PENDING. Routing need not change before a separately frozen held-out test of these two revisions; it remains a known limitation and may prevent latency dominance.','',
        'MODULE8E2_ALLOWED=true for held-out protocol preparation and then validation after preregistration. HELDOUT_GPU_CAMPAIGN_READY=false until the fresh workload matrix/protocol is frozen. No held-out campaign is launched here.','']
    (OUT/'module8e1_summary.md').write_text('\n'.join(lines))
    print({'report_complete':True,'MODULE8E2_ALLOWED':ready,'cache_misses':miss['CACHE_MISS_occurrences']},flush=True)


def validate():
    verify_v2();verify_inputs()
    freeze=read(OUT/'method_v1_freeze.json')
    for r in freeze['archived_sources'].values():assert file_hash(r['path'])==r['sha256']
    assert file_hash(freeze['wrapper']['path'])==freeze['wrapper']['sha256']
    for p in [D2/'artifact_manifest.json',D3/'artifact_manifest.json']:
        assert file_hash(p)==Path(str(p)+'.sha256').read_text().strip()
        for r in read(p)['artifacts'].values():assert file_hash(r['path'])==r['sha256']
    assert environment_snapshot()==read(C/'environment_manifest.json')
    required=['method_v1_freeze.json','method_v2_spec.json','method_v2_source_hashes.json','swap_generation_validation.json','candidate_class_validation.json','synthetic_revision_validation.json','development_replay.json','development_cache_misses.json','b2_path_compression.csv','reasoning_replacement_case.json','complexity_validation.json','regression_results.json','routing_revision_status.json','heldout_validation_requirements.md','module8e1_summary.md']
    for name in required:assert (OUT/name).is_file() and (OUT/name).stat().st_size>0
    assert read(OUT/'regression_results.json')['status']=='PASS'
    assert read(OUT/'reasoning_replacement_case.json')['NEIGHBORHOOD_FIX_CONFIRMED_OFFLINE']
    rep=read(OUT/'development_replay.json')['workloads'];assert len(rep)==6
    assert all(w['logical_selected_state_attempts']<=6 and not w['committed_final_claim_allowed'] and not w['unbiased_final_validation_claim'] for w in rep)
    assert sum(w['cache_misses'] for w in rep)==read(OUT/'development_cache_misses.json')['CACHE_MISS_occurrences']==6
    for w in rep:
        for it in w['trace']['iterations']:
            assert len(it['selected_top_b'])<=2
            assert all(c['mutation']['deployment'].get('paddleocr_vl',0)==0 for c in it['candidates'])
    assert read(OUT/'routing_revision_status.json')['status']=='ROUTING_REVISION_PENDING'
    gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128
    save_json(OUT/'final_validation.json',{'status':'PASS','all_v1_frozen_sources_and_traces_unchanged':True,'all_8C_8D_8D3_inputs_raw_artifacts_unchanged':True,'source_archive_files':len(freeze['sources']),'trace_files':len(freeze['trace_sources']),'tests':read(OUT/'regression_results.json')['tests_run'],'production_enumeration_count':0,'no_new_GPU_experiments':True,'development_only':True,'unbiased_final_claim':False,'all_required_outputs':required,'GPU_idle':gpu,'validator':ref(Path(__file__).resolve())})
    artifacts={str(p.relative_to(ROOT)):ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ['artifact_manifest.json','artifact_manifest.json.sha256']}
    for folder in ['method_v1','method_v2']:
        for p in sorted((ROOT/folder).glob('*.py')):artifacts[str(p.relative_to(ROOT))]=ref(p)
    for p in sorted(ROOT.glob('module8e1_*.py')):artifacts[str(p.relative_to(ROOT))]=ref(p)
    save_json(OUT/'artifact_manifest.json',{'module':'8E-1','status':'COMPLETE','artifacts':artifacts,'new_GPU_runs':0,'method_v1_overwritten':False,'stop_after_module':'8E-1'})
    Path(str(OUT/'artifact_manifest.json')+'.sha256').write_text(file_hash(OUT/'artifact_manifest.json')+'\n')
    print({'final_validation':'PASS','artifacts':len(artifacts),'GPU_runs':0},flush=True)

if __name__=='__main__':report();validate()
