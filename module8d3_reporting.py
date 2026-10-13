"""Derived offline reporting; does not change the frozen crossover runner."""
import csv
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from module8d3_diagnosis import (OUT, D2, C, P, F, ROOT, NAMES, CAT, ROLES, BOUNDS,
    POLICY, Offline, read, ref, counts, key, identity, flatten_trace_route, flat,
    better, csvwrite, save_json, digest, validate_raw, verify_inputs)


def supplementary_offline():
    audit = Offline()
    # Required changes already accomplished in an incumbent are not missing
    # ADDs/REMOVEs. Preserve all trace rows, annotate their scope correctly.
    with (OUT / 'reachability_trace.csv').open(newline='') as f: rows = list(csv.DictReader(f))
    trace_iterations={(method,str(seed),n,str(it['iteration'])):it for method,seed,n,path,t in audit.traces for it in t['iterations']}
    for row in rows:
        if row['method']=='Random':
            it=trace_iterations['Random',row['random_seed'],row['workload'],row['iteration']]
            ranked=sorted((c for c in it['candidates'] if c.get('delta_z') is not None),key=lambda c:(-c['delta_z'],c['mutation']['model']))
            row['analytical_rank']=next((i for i,c in enumerate(ranked,1) if c['mutation']['model']==row['required_mutation_model'] and c['mutation']['deployment'][c['mutation']['model']]-it['deployment'][c['mutation']['model']]==int(row['delta'])),None)
        row['analytical_rank_semantics']='Post-campaign unchanged LP ordering for diagnosis; never used for Random candidate selection' if row['method']=='Random' else 'Frozen Proposed trace rank'
        target = counts(read(D2/'b2'/row['workload']/'result.json')['state']['deployment'])
        current = json.loads(row['current_vector']); m=row['required_mutation_model']; sign=int(row['delta'])
        required = (target[m]-current[m])*sign > 0
        row['still_required_in_this_incumbent'] = required
        if not required:
            row['trace_status'] = 'TARGET_COMPONENT_ALREADY_SATISFIED'
            row['reason_not_evaluated'] = 'Mutation is no longer required to reach B2; generation/evaluation columns describe the trace candidate, not a missing target step'
        row['failure_stage'] = ('NONE' if not required or row['trace_status']=='ACCEPTED' else
            'NEIGHBORHOOD' if row['trace_status'] in ('NOT_GENERATED','REJECT_INFEASIBLE') else
            'RANKING' if row['trace_status']=='PRUNED_TOP_B' else
            'RUNTIME' if row['trace_status']=='ROLLBACK' else
            'BUDGET' if row['trace_status']=='NOT_EVALUATED_BUDGET' else 'NONE')
    csvwrite('reachability_trace.csv', rows)
    proof_rows=[]
    for state in read(OUT/'routing_crossover_audit.json')['variants']:
        ev=state['evidence']
        if not ev: continue
        proofs=[validate_raw(p['path'],state['identity'],seed,state['workload']) for p,seed in zip(ev['raw_sources'],(7101,7202,7303))]
        proof_rows.append({'workload':state['workload'],'variant':state['variant'],'all_12_conditions_equal':True,'proofs':proofs})
    save_json(OUT/'routing_crossover_reuse_manifest.json',{'states':proof_rows,'new_GPU_measurements':0})
    # All requested scope metrics, with exact shared-incumbent pair units.
    groups={}
    for method,seed,n,path,t in audit.traces:
        for it in t['iterations']:
            context=digest(identity(n,it['deployment'],flatten_trace_route(it['canonical']['routing'])))
            g=groups.setdefault((n,context),{'n':n,'context':context,'current':it['current_runtime'],'x':it['deployment'],'feasible':{},'measured':{}})
            for c in it['candidates']:
                if c.get('delta_z') is None: continue
                x=c['mutation']['deployment'];route=flatten_trace_route(c['canonical']['routing']);cid=digest(identity(n,x,route))
                item={'id':cid,'model':c['mutation']['model'],'role':ROLES[c['mutation']['model']], 'dz':c['delta_z'],'x':x}
                g['feasible'][cid]=item
                ev=audit.evidence(n,x,route)
                if ev:
                    rt=ev['runtime'];item=dict(item,rt=rt,dm=it['current_runtime']['weighted_mean_e2e']-rt['weighted_mean_e2e'],dp=it['current_runtime']['weighted_p95_e2e']-rt['weighted_p95_e2e'],runtime_better=better(rt,it['current_runtime'],x,it['deployment'])[0])
                    g['measured'][cid]=item
    def scope(ns=None,role=None):
        pair=Counter(); hit=[]
        for (n,context),g in groups.items():
            if ns and n!=ns:continue
            fs=[v for v in g['feasible'].values() if not role or v['role']==role]
            ms=[v for v in g['measured'].values() if not role or v['role']==role]
            for a,b in itertools.combinations(ms,2):
                dx,dy=a['dz']-b['dz'],a['dm']-b['dm']
                if not dx or not dy:pair['ties']+=1
                else:pair['eligible']+=1;pair['correct']+=int(dx*dy>0)
            if len(ms)<2:continue
            best={a['id'] for a in ms if not any(better(b['rt'],a['rt'],b['x'],a['x'])[0] for b in ms if b is not a)}
            ranked=sorted(fs,key=lambda a:(-a['dz'],a['model']))
            hit.append({'context':context,'workload':n,'complete':{a['id'] for a in fs}<={a['id'] for a in ms},'measured_n':len(ms),'feasible_n':len(fs),'Top1':ranked[0]['id'] in best,'Top2':bool({a['id'] for a in ranked[:2]}&best),'opportunity':any(a['runtime_better'] for a in ms)})
        def hs(gs):
            return {'n_choice_sets':len(gs),'status':'EXPLORATORY' if len(gs)>=3 else 'INSUFFICIENT','Top1_hit_rate':sum(g['Top1'] for g in gs)/len(gs) if gs else None,'Top2_hit_rate':sum(g['Top2'] for g in gs)/len(gs) if gs else None}
        return {'pairwise_ordering_accuracy':pair['correct']/pair['eligible'] if pair['eligible'] else None,'pairwise_counts':dict(pair),'pairwise_status':'EXPLORATORY' if pair['eligible']>=3 else 'INSUFFICIENT','Top_hits_complete_choice_sets':hs([g for g in hit if g['complete']]),'Top_hits_observed_partial_sets':hs(hit),'Top_hits_complete_runtime_improvement_opportunities':hs([g for g in hit if g['complete'] and g['opportunity']]),'choice_sets':hit}
    path=OUT/'lp_runtime_alignment_summary.json';s=read(path)
    s['stratified_ordering_and_hits']={'pooled':scope(),'per_workload':{n:scope(n) for n in NAMES},'by_family_role':{r:scope(role=r) for r in ('generalist','specialist')}}
    s['deduplication_interpretation']='Per-workload repeated event n=8 with two unique edges is INSUFFICIENT for independent alignment, despite identical event correlations; no p-values or independent RNG replication claim.'
    save_json(path,s)
    budget=read(OUT/'search_budget_audit.json')
    for row in budget['workloads']:
        row['budgeted_one_more_iteration_evaluable_count']=min(2,row['remaining_state_slots'],len(row['remaining_feasible_unvisited_candidates']))
        row['not_actually_budget_exhausted']=row['termination_reason']!='RUNTIME_EVALUATION_BUDGET_REACHED'
        if row['workload']=='reasoning_vqa_heavy':
            row['minus_Phi_one_more_iteration_rank']=2
            row['minus_Phi_requires_state_budget_7_if_original_ranking_kept']=True
            row['reason']='After taboo removal, -Whisper-small is rank1 and -Phi rank2; only one slot remains at 5/6. Patience alone cannot evaluate -Phi under unchanged ordering.'
    save_json(OUT/'search_budget_audit.json',budget)
    pool=read(P/'service_demand_v2_1.json')['models']
    save_json(OUT/'resource_constraint_summary.json',{'GPU_capacity_mib':49140,'formal_12_model_idle_sum_mib':sum(p['measured_idle_vram_mib'] for p in pool.values()),'effective_admitted_11_model_single_replica_memory_mib':POLICY.deployment_memory({m:1 for m in CAT}),'B2_five_VQA_workloads_memory_mib':25188,'B2_noVQA_memory_mib':7210,'all_pool_exceeds_GPU_does_not_imply_useful_subset_is_resource_bound':True})
    print(json.dumps({'event':'supplementary_offline_complete','reuse_states':len(proof_rows),'ordering_scopes_complete':True}),flush=True)


def report():
    verify_inputs();audit=Offline();variants=read(OUT/'routing_crossover_audit.json')['variants'];new=read(OUT/'crossover_results.json')
    extra=read(OUT/'crossover_stage2_results.json') if (OUT/'crossover_stage2_results.json').exists() else {'status':'NOT_RUN','new_states':[],'new_timed_runs':0}
    lookup={(v['workload'],v['variant']):v['evidence'] for v in variants}
    for v in new.get('new_states',[])+extra.get('new_states',[]):lookup[v['workload'],v['variant']]=v
    decomposed=[]
    for n in NAMES:
        entries={label:lookup[n,label] for label in 'ABCD'}
        def contrast(first,second):
            a,b=entries[first],entries[second]
            if not a or not b:return {'status':'CROSSOVER_MEASUREMENT_MISSING'}
            x,y=a['runtime'],b['runtime']
            return {'status':'MEASURED','mean_difference_ms':x['weighted_mean_e2e']-y['weighted_mean_e2e'],'P95_difference_ms':x['weighted_p95_e2e']-y['weighted_p95_e2e'],'first_minus_second':first+'-'+second,'interpretation':'A-B routing on B2 deployment; B-C deployment change with canonical recomputed, not pure physical presence; D-C routing on Proposed deployment; A-D deployment under greedy-compatible policy; B-D is a fixed-routing deployment contrast only when exact routing equality is confirmed'}
        row={'workload':n,'states':{label: {'status':'MEASURED','mean_ms':e['runtime']['weighted_mean_e2e'],'P95_ms':e['runtime']['weighted_p95_e2e'],'stability':e['runtime']['stability_state_v2'],'mean_std_ms':e['runtime']['replicate_statistics']['weighted_mean_e2e']['sample_std'],'source_paths':e['runtime']['raw_result_paths']} if e else {'status':'CROSSOVER_MEASUREMENT_MISSING'} for label,e in entries.items()},'contrasts':{a+b:contrast(a,b) for a,b in [('A','B'),('B','C'),('D','C'),('A','D'),('B','D')]},'interaction_identified':bool(entries['D'] and entries['B']),'B_vs_D_exact_routing_equal':bool(entries['B'] and entries['D'] and entries['B']['identity']['routing']==entries['D']['identity']['routing'])}
        decomposed.append(row)
    save_json(OUT/'crossover_decomposition.json',{'workloads':decomposed,'new_timed_runs':new.get('new_timed_runs',0)+extra.get('new_timed_runs',0),'scope':'Conditional contrasts for fixed frozen deployments/routes; no overall pure deployment percentage without full D','no_new_search':True})
    # Combined index preserves the original stage-1 file referenced by the stage-2 frozen plan.
    save_json(OUT/'all_crossover_results.json',{'status':'COMPLETE','stage1':ref(OUT/'crossover_results.json'),'stage2':ref(OUT/'crossover_stage2_results.json') if extra['status']=='COMPLETE' else None,'plans':[ref(OUT/'crossover_plan.json'),ref(OUT/'crossover_plan_stage2.json')],'new_states':new.get('new_states',[])+extra.get('new_states',[]),'new_timed_runs':new.get('new_timed_runs',0)+extra.get('new_timed_runs',0),'no_search_no_ranking':True})
    failures=[]
    for n in NAMES:
        tr=read(C/n/'search_complete.json');reach=read(OUT/'reachability_paths.json')[n];stages=[];evidence=[]
        if n in ('demand_balanced','vision_heavy','text_ocr_heavy'):
            stages=['NEIGHBORHOOD','RUNTIME','STOPPING'];cause='MULTIPLE'
            components=['NEIGHBORHOOD_LIMITATION','OBJECTIVE_ACCEPTANCE_BARRIER']
            evidence=['Seed pressure_tasks=[visual_qa], so +Distil-Whisper is not generated; actual generated graph shortest path is 5, exceeding max 3 accepted iterations.','+Qwen is measured but rolls back under unchanged objective; no accepted incumbent enables -Phi.','After first no-improvement stop, no feasible unvisited generated neighbor remains; more patience alone is insufficient.','B2/canonical missing for this workload; deployment versus routing attribution cannot be generalized from other workloads.']
        elif n=='speech_heavy':
            stages=['RUNTIME','STOPPING'];cause='MULTIPLE';components=['TWO_STEP_SWAP_ACCEPTANCE_BARRIER','ROUTING_MISMATCH']
            evidence=['+Distil-Whisper accepted first; +Qwen from final is generated rank1 and measured 63.1205ms versus current55.6846ms, then rolled back.','-Phi is infeasible before +Qwen; x±1 replacement requires two incumbent transitions, with an observed rejected intermediate.','Raising patience alone does not retry the already rejected +Qwen edge; crossover establishes conditional routing/deployment contrasts only.']
        elif n=='reasoning_vqa_heavy':
            stages=['RANKING','NEIGHBORHOOD','STOPPING','BUDGET'];cause='MULTIPLE';components=['ANALYTICAL_RANKING_MISMATCH','ROUTING_MISMATCH','NEIGHBORHOOD_LIMITATION','STOPPING_POLICY','CONDITIONAL_SEARCH_BUDGET_LIMIT']
            evidence=['+Qwen improves103.2430→95.4905ms (7.509%) and is accepted.','-Phi is generated, LP feasible, analytical rank4/delta_z=-0.08891564, pruned by Top2, not runtime evaluated in Proposed.','Random-2207 evaluates -Phi and accepts86.4063ms; state is steady and memory21516 vs33288MiB.','Fixed Proposed deployment greedy-compatible routing reduces mean95.4905→88.8546ms while Phi stays physically deployed; n=3/std3.4522ms is descriptive evidence of a routing contribution.','Proposed stops at5/6 states due patience=1. One extra iteration with original ranking would evaluate -Whisper-small first and lack the seventh slot for -Phi.','+Distil-Whisper is not generated with VQA-only bottleneck; exact B2 vector requires a pressure-gated detour even after Qwen replacement.']
        else:
            stages=['NONE'];cause='NONE';components=[];evidence=['Proposed already has exact B2 vector; difference is routing only.','A/B/C/D exact historical evidence exists. Primary mean21.0027 vs21.3554ms; frozen objective and confirmations classify tie.','Do not manufacture a formal failure from a below-gate difference.']
        failures.append({'workload':n,'primary_failure_cause':cause,'failure_stage':stages,'components':components,'evidence':evidence,'structural_seed_distance':reach['seed']['structural_l1'],'pressure_gated_seed_min_length':reach['seed']['pressure_gated_min_length'],'resource_constraint_excludes_B2':False,'crossover':next(r for r in decomposed if r['workload']==n)})
    classification={'workloads':failures,'DOMINANT_CAUSE':'MULTIPLE','dominant_components':['PRESSURE_GATED_NEIGHBORHOOD','PRESSURE_OBJECTIVE_RANKING_ROUTING_MISMATCH','TWO_STEP_RUNTIME_ACCEPTANCE_BARRIER'],'stopping_budget':'Amplifier, not sole global explanation; no Proposed job actually exhausted the six-state cap.','resource_constraint':'B2 useful subset is not resource constrained; all-pool infeasibility is not evidence of B2 infeasibility.','seed_objective':'Frozen empirical MILP minimizes feasible memory first, then replica count, rather than runtime latency. It chooses Phi and avoids unnecessary Distil at seed; this objective preference is distinct from a binding GPU memory limit.','unmeasured_workload_attribution':'Balanced/vision/text canonical-B2 crossovers remain missing; no universal deployment-vs-routing percentages claimed.','recommended_next_stage':'METHOD_REVISION_DESIGN_AND_MINIMAL_VALIDATION_BEFORE_ORIGINAL_ABLATION','method_changed':False}
    save_json(OUT/'failure_classification.json',classification)
    t=read(C/'reasoning_vqa_heavy'/'search_complete.json');it=t['iterations'][1];phi=next(c for c in it['candidates'] if c['mutation']['model']=='phi4mm' and c['mutation']['operation']=='SCALE_DOWN')
    case=['# Reasoning/VQA-heavy diagnosis','','Seed Phi+S4 → Proposed Phi+Qwen+S4 → Random-2207 Qwen+S4; B2=Qwen+S4+Distil-Whisper.','',
        '+Qwen accepted: mean103.243→95.491ms (7.509%), steady; frozen 5% gate passes. After this correct ADD, -Phi is generated and coverage/memory/canonical LP feasible, but analytical rank=4, delta_z=-0.08891564427; Top-B=2 excludes it. Proposed never evaluates this REMOVE.','',
        'Rank1=-DistilBERT and rank2=-ResNet50 both worsen runtime and roll back; rank3=-Whisper-small and rank4=-Phi remain untested. Removal raises min-max pressure, which ranks it poorly, but removal also sends VQA entirely to the faster Qwen path; Random evaluates and accepts it.','',
        'Random-2207 -Phi: mean86.406ms, P95=147.757ms, steady, memory21516MiB; Proposed95.491ms/157.304ms/33288MiB. Independent confirmations also show improvement. Correct ADD without completing matching REMOVE is directly observed.','',
        'A SWAP (-Phi,+Qwen) requires two x±1 incumbent transitions. In this workload +Qwen already passes acceptance, so no intermediate acceptance barrier is observed for this SWAP. In balanced/vision/text the +Qwen intermediate rolls back; in speech after Distil+1 it becomes slower, a direct barrier. Do not generalize that barrier to reasoning.','',
        'Stopping is NO_RUNTIME_IMPROVEMENT after two iterations/5 states, with one state slot remaining. Without changing ranking, one more iteration would place -Whisper-small rank1 and -Phi rank2; only the former fits the sixth slot. A future diagnostic needs budget7 plus patience2 (max_iterations3 unchanged) to isolate this path; not executed here.','',
        'B2 is another step farther structurally (+Distil), but that ADD is absent while VQA dominates. Actual pressure-gated graph needs4 steps from Proposed final to exact B2 vector, including a Whisper-small detour. Feasible graph reachability does not prove runtime-accepted reachability.','']
    reason_cross=next(r for r in decomposed if r['workload']=='reasoning_vqa_heavy')
    speech_cross=next(r for r in decomposed if r['workload']=='speech_heavy')
    if reason_cross['states']['D']['status']=='MEASURED':
        rr=reason_cross['states'];dc=reason_cross['contrasts']['DC'];bd=reason_cross['contrasts']['BD']
        case += ['Fixed-deployment routing crossover (n=3 per new state):', '',
            f"C canonical={rr['C']['mean_ms']:.3f}ms, D greedy-compatible={rr['D']['mean_ms']:.3f}ms; D-C={dc['mean_difference_ms']:.3f}ms, P95 difference={dc['P95_difference_ms']:.3f}ms. Phi remains physically deployed in D but receives no VQA; memory vector is unchanged. This is routing evidence at fixed Proposed deployment.", '',
            f"B versus D exact routing equality={reason_cross['B_vs_D_exact_routing_equal']}; B-D mean={bd['mean_difference_ms']:.3f}ms, P95={bd['P95_difference_ms']:.3f}ms. With equality true, this changes deployed Phi versus Distil-Whisper while preserving all task routing. B2 greedy A={rr['A']['mean_ms']:.3f}ms. Do not attribute the entire C-to-A improvement to deleting Phi.", '',
            f"D mean sample std={rr['D']['mean_std_ms']:.3f}ms, B std={rr['B']['mean_std_ms']:.3f}ms. These small, separately timed n=3 contrasts are descriptive, without an inferential significance or exact percentage-of-causality claim.", '']
    (OUT/'reasoning_vqa_case_study.md').write_text('\n'.join(case))
    rec=['# Method revision recommendations — design only','','1. Reconcile min-max pressure routing/ranking with the measured E2E objective. Δz is useful for capacity relief but can favor slower paths at low utilization and penalize beneficial removals. Standalone latency is an LP input, but minimizing maximum pressure is not minimizing weighted mean E2E/P95. The empirical seed separately minimizes memory, then replica count; a cheaper feasible seed need not have lower latency. Report objective-specific gaps; do not train a predictor or patch ranking in this module.','',
        '2. Audit candidate admission beyond only bottleneck tasks: quality-admitted ASR savings can remain invisible when VQA sets z. Consider a bounded additional specialist-offload candidate set as a future design, independently frozen before testing.','',
        '3. Consider SWAP for coverage-preserving Generalist replacement. For effective M=11, at most M(M-1)=110 ordered cross-family down/up pairs before filtering, versus at most2M=22 one-hop mutations; if restricted to one deployed Generalist and two absent Generalists, only2 alternatives. In reasoning there are2 deployed Generalists and1 absent one, at most2 Generalist swaps; a same-family pair is a no-op and should be excluded. Measure complexity after feasibility filtering.','',
        'SWAP addresses rejected ADD-before-REMOVE intermediates. It does not by itself fix positive-only pressure preference, missing non-bottleneck +Distil, or poor canonical routing; do not claim it solves B2 dominance. No SWAP implemented.','',
        '4. Minimal future budget sensitivity: freeze 6 versus7 unique states and patience1 versus2 with same max_iterations3 in reasoning. More patience alone under budget6 cannot reach rank2 -Phi in the next iteration. Balanced/vision/text have no remaining generated feasible unvisited candidate, so increasing budget alone is not a supported global cure. No sensitivity run executed.','',
        '5. Establish revised method validity before continuing the original ablation campaign. B2 remains the strongest latency baseline and Random found a better replacement state. Preserve current method/results as frozen evidence; any revision needs a new version and independently frozen protocol. No new predictor, quality threshold, workload, model, routing optimizer or deployment search implemented.','']
    (OUT/'method_revision_recommendations.md').write_text('\n'.join(rec))
    alignment=read(OUT/'lp_runtime_alignment_summary.json');raw=alignment['all_event_correlations']['pooled'];unique=alignment['deduplicated_sensitivity']['pooled']
    resources=[]
    with (OUT/'resource_constraint_audit.csv').open(newline='') as f:
        resources=list(csv.DictReader(f))
    rb=[r for r in resources if r['method']=='B2'];head=[float(r['memory_headroom_mib']) for r in rb]
    lines=['# Module 8D-3 — B2 dominance and search diagnosis','','Frozen algorithms/λ/quality/pool/revisions are unchanged. No new deployment search, ablation, dynamic adaptation or second hardware.','',
        '## Distance and actual generated-graph reachability','',
        '| Workload | seed→B2 L1 | final→B2 L1 | seed pressure-gated shortest | final pressure-gated shortest |','|---|---:|---:|---:|---:|']
    paths=read(OUT/'reachability_paths.json')
    for n in NAMES:
        s,f=paths[n]['seed'],paths[n]['final'];lines.append(f"| {n} | {s['structural_l1']} | {f['structural_l1']} | {s['pressure_gated_min_length']} | {f['pressure_gated_min_length']} |")
    lines+=['','All six B2 vectors are feasible and graph-reachable, but structural distance is not generated-neighborhood distance and neither is accepted runtime reachability. Four VQA regimes require removing/readding Whisper-small to expose ASR pressure and allow +Distil; shortest seed paths need5 steps, more than max3 iterations. Path witnesses and lattice-parity lower bounds are in reachability_paths.json.','',
        '## LP/runtime alignment','',
        f"69 evaluated transition events retained;30 exact distinct transitions. Pooled event Spearman(Δz,Δmean)={raw['Spearman_delta_z_delta_mean']:.3f}, Kendall={raw['Kendall_tau_delta_z_delta_mean']:.3f}, Spearman(Δz,ΔP95)={raw['Spearman_delta_z_delta_p95']:.3f}. Deduplicated sensitivity: {unique['Spearman_delta_z_delta_mean']:.3f}/{unique['Kendall_tau_delta_z_delta_mean']:.3f}/{unique['Spearman_delta_z_delta_p95']:.3f}. Shared raw samples are not independent.",
        'Balanced/vision/text each have only2 unique transitions, so independent per-workload correlation is INSUFFICIENT; repeated-event correlation1.0 cannot be treated as robust evidence. Generalist and Specialist scopes, pairwise ordering and Top1/Top2 hit rates with measurement coverage are reported in lp_runtime_alignment_summary.json.','',
        'Exact known misses below Top2: speech +Whisper-small replica (rank5) is a better-than-seed candidate, though Proposed evaluates a stronger Distil candidate; reasoning -Phi (rank4) is a decisive missed beneficial REMOVE. Do not call every unknown unmeasured candidate a missed winner or claim pervasive Top-B failure from two cases.','',
        '## Random path analysis','',
        'Balanced/vision/text have only2 feasible generated alternatives; all random seeds evaluate the same2 candidates as Proposed, and all retain the same seed. Speech alternatives differ but final objectives tie. Reasoning-2207 randomly selects -Phi after +Qwen and escapes the poor retained Phi+Qwen state. no-VQA1103/2207 omit +Distil and stop atseed, while47 matches Proposed. Fifteen objective ties are not fifteen independent replications; exact cache/state overlap and per-seed path sets are preserved.','',
        '## Routing/deployment crossover','',
        f"New fixed-state timed runs={new.get('new_timed_runs',0)+extra.get('new_timed_runs',0)}. A=B2/greedy, B=B2/canonical, C=Proposed/canonical, D=Proposed/greedy-compatible. Every reused point passed all12 identity checks; every new state uses3 frozen repetitions. Missing cells remain missing.",
        '| Workload | A mean | B mean | C mean | D mean |','|---|---:|---:|---:|---:|']
    for row in decomposed:
        lines.append('| '+row['workload']+' | '+' | '.join(f"{row['states'][l]['mean_ms']:.3f}" if row['states'][l]['status']=='MEASURED' else 'MISSING' for l in 'ABCD')+' |')
    lines+=['','A−B measures routing at fixed B2 deployment; D−C measures routing at fixed Proposed deployment. B−C changes deployment and recomputes canonical routing, so it is not a pure worker-presence effect. Interactions/generalization remain unidentified whereD/B is missing. Full P95/std/source-path contrasts are in crossover_decomposition.json.','',
        f"Reasoning fixed Proposed deployment: C→D mean {reason_cross['states']['C']['mean_ms']:.3f}→{reason_cross['states']['D']['mean_ms']:.3f}ms; Phi remains deployed but VQA goes entirely to Qwen. B/D exact routing equality={reason_cross['B_vs_D_exact_routing_equal']}. This directly separates a routing contrast from the B/D deployment contrast; n=3 descriptive evidence and observed std do not establish an exact causal allocation.", '',
        f"Speech A/B/C mean={speech_cross['states']['A']['mean_ms']:.3f}/{speech_cross['states']['B']['mean_ms']:.3f}/{speech_cross['states']['C']['mean_ms']:.3f}ms, P95={speech_cross['states']['A']['P95_ms']:.3f}/{speech_cross['states']['B']['P95_ms']:.3f}/{speech_cross['states']['C']['P95_ms']:.3f}ms. B2 canonical lowers mean versus Proposed but raises P95; greedy routing lowers the B2 tail. D was not triggered by the frozen mean-sign gate, so the speech interaction remains unidentified.", '',
        '## Greedy strength and resource constraint','',
        'Fastest admitted Specialist vs fastest Generalist s0 ratios: text11.84×, image25.39×, OCR26.43×, ASR4.47×, LID5.17×; VQA has no Specialist. All choices satisfy frozen quality admission. B2 combines five intrinsically faster paths and the fastest admitted VQA Generalist; it is a naturally strong latency heuristic, not a weak baseline.',
        f'B2 sampled peak headroom is {min(head):.0f}–{max(head):.0f}MiB. Its first-five peaks are about26400MiB (~54% of49140MiB capacity); no-VQA peak7205MiB. All-pool memory exceeds GPU, but useful B2 subsets fit comfortably. No GPU budget reduced.','',
        'The frozen empirical MILP seed minimizes feasible memory first, then replica count. It can prefer Phi and omit extra Distil despite ample headroom; this is a seed objective choice, not evidence that the 49GB hard limit binds. The runtime search then optimizes a different E2E objective, through a pressure-gated neighborhood.', '',
        '## Stopping and classification','',
        'All Proposed jobs terminate NO_RUNTIME_IMPROVEMENT, at3/6 or5/6 states, never by hard state cap/max_iterations. Speech/reasoning/noVQA have remaining unvisited feasible choices; balanced/vision/text do not under unchanged generation+tabu. Patience amplifies missed candidates, but budget expansion cannot by itself fix generator gating, rejected ADD intermediates or routing misalignment.','',
        'DOMINANT_CAUSE=MULTIPLE: pressure-gated neighborhood, pressure-objective ranking/routing mismatch and two-step acceptance barriers; stopping/budget amplify selected cases. Resource capacity does not exclude B2. no-VQA is NONE under the formal objective, with a small routing-only primary-metric difference.','',
        'Recommend method revision design and minimal separately frozen validation before original ablation; no revision or ablation implemented. SWAP, ranking and routing changes are recommendations only.','']
    (OUT/'module8d3_summary.md').write_text('\n'.join(lines))
    verify_inputs();print(json.dumps({'event':'diagnosis_report_complete','new_timed_runs':new.get('new_timed_runs',0)+extra.get('new_timed_runs',0),'DOMINANT_CAUSE':'MULTIPLE'}),flush=True)


if __name__=='__main__':
    import sys
    (supplementary_offline if sys.argv[1]=='supplementary' else report)()
