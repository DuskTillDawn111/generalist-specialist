"""Frozen development evidence only; no imports of a GPU-serving runner."""
import itertools,math,time
from types import SimpleNamespace
import numpy as np
from scipy.stats import spearmanr,kendalltau
from deployment_core import Workload
from deployment_search.routing import solve_canonical_routing
from deployment_core.routing.e2e_surrogate import evaluate_surrogate
from deployment_core.routing.e2e_router import solve_e2e_routing
from deployment_core.search.e2e_candidate_ranker import E2ECandidateRanker
from method_v2.selection import diversified_order,candidate_class
from .common import *

def correlations(rows,predictor,outcome):
    rows=[r for r in rows if r.get(predictor) is not None and r.get(outcome) is not None]
    x=[r[predictor] for r in rows];y=[r[outcome] for r in rows]
    if len(rows)<2 or len(set(x))<2 or len(set(y))<2:
        return {'n':len(rows),'Spearman':None,'Kendall':None,'pairwise_ordering_accuracy':None,'comparable_pairs':0}
    comparable=[(a,b) for a,b in itertools.combinations(rows,2) if a[outcome]!=b[outcome]]
    accuracy=sum(.5 if a[predictor]==b[predictor] else float((a[predictor]-b[predictor])*(a[outcome]-b[outcome])>0)
                 for a,b in comparable)/len(comparable) if comparable else None
    return {'n':len(rows),'Spearman':float(spearmanr(x,y).statistic),'Kendall':float(kendalltau(x,y).statistic),
            'pairwise_ordering_accuracy':accuracy,'comparable_pairs':len(comparable)}

def frozen_refs():
    refs=list(read(PROTOCOL)['frozen_inputs'].values())
    refs+=list(read(ROOT/'results/module8e1_method_revision/method_v2_source_hashes.json')['sources'].values())
    refs+=read(HISTORY/'execution_freeze.json')['protected_references']
    refs+=list(read(HISTORY/'execution_freeze.json')['execution_sources'].values())
    refs+=[ref(p) for d in ['module9a_interference','module9a_scheduling_tax','module8f3_structural_dominance']
           for p in (ROOT/'results'/d).rglob('*') if p.is_file() and p.suffix not in ['.pyc','.log']]
    # All existing native worker/transport references from the closed 9A audit.
    integrity=read(ROOT/'results/module9a_scheduling_tax/native_runtime_integrity.json')
    def collect(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:refs.append({'path':value['path'],'sha256':value['sha256']})
            else:
                for v in value.values():collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    collect(integrity)
    return list({r['path']:r for r in refs}.values())

def historical_states():
    physical=read(HISTORY/'physical_measurements.json')['runs'];byraw={r['raw_path']:r for r in physical}
    groups={};used=set()
    for path in sorted(HISTORY.glob('C*/**/*aggregate.json')):
        data=read(path)
        for rep in data.get('repetitions',[]):
            raw=rep.get('raw_result_path')
            if raw in used or raw not in byraw:continue
            used.add(raw);source=byraw[raw];config=source['identity']['config']
            route={(k,m):v for k,rs in config['routing_rps'].items() for m,v in rs.items()}
            key=statekey(source['workload'],config['deployment'],route)
            g=groups.setdefault(key,{'id':key,'cell':source['workload'],'deployment':config['deployment'],
                         'routing':serial(route),'repetitions':[],'owners':[],'source_refs':[]})
            g['repetitions'].append(dict(labels(rep),raw_path=raw,seed=rep['random_seed']))
            g['owners'].append(source['owner']);g['source_refs'].append(ref(path))
    for g in groups.values():
        reps=g['repetitions'];g['mean_e2e_ms']=float(np.mean([r['mean_e2e_ms'] for r in reps]));g['weighted_p95_ms']=float(np.mean([r['weighted_p95_ms'] for r in reps]))
        g['stability']=min((r['stability'] for r in reps),key=lambda tier:TIER[tier])
    assert len(used)==sum(r['valid'] for r in physical),'Missing historical physical repetitions'
    return groups

def main():
    assert not (OUT/'experiment_freeze.json').exists(),'Existing 9BC freeze must not be overwritten'
    OUT.mkdir(parents=True,exist_ok=True)
    catalog,hardware,roles,cal=inputs();grid=cells();factor=read(ROOT/'results/module8f1a_memory_breakpoints/memory_safety_factor.json')['safety_factor']
    refs=frozen_refs();verify(refs)
    assert read(ROOT/'results/module9a_scheduling_tax/fitted_model.json')['selected_model']=='I0'
    freeze={'at_utc':now(),'protected_references':refs,'development_evidence':['module8f2_resource_static'],
            'excluded':['module8e3 heldout labels','future final heldout','9A pair labels'],
            's0':'frozen candidate_pool_v2_1/service_demand_v2_1.json; no 9A matched-anchor substitution',
            'interference_model':'I0','objective':'sum_m D_m/(1-D_m/g_m) / sum_k lambda_k',
            'formula_weights':None,'solver':'SLSQP analytic gradient; objective expressed in ms for conditioning; canonical initial point, fixed maxiter=1000, ftol=1e-12 ms, epsilon=1e-9; convex linearization gap <=1e-8 seconds',
            'routing_state_selection':'four distinct exact historical states: minimum-z I1 routing-changing 1G; minimum-z I2 routing-changing 1G; maximum-z I3 routing-changing RC-SG-owned if available otherwise 1G; minimum-z historical multi-G routing-changing state. Deterministic cell/vector tie-break. Slots without routing-changing states use same category with no change. No new GPU labels used.',
            'routing_repetitions':3,'routing_seeds':read(PROTOCOL)['search_seeds'],
            'routing_order':'per state rep1 R0,R1; rep2 R1,R0; rep3 R0,R1',
            'routing_gate_rule':'NOT_SUPPORTED on more state-level stability degradations than improvements, or more matched-steady mean regressions than improvements, or at least two states R0 steady and R1 nonsteady; SUPPORTED requires no stability degradation, nonnegative aggregate matched-steady ordering, and one all-three matched-steady improvement; otherwise PARTIAL. No magnitude threshold.',
            'ranking_gate_rule':'compare equal-iteration measured-only rank correlation and Top-2 hit; SUPPORTED if improvements without conflicting degradation and GPU sanity no worse frozen objective; NOT_SUPPORTED on consistent replay regression or systematic sanity regression; otherwise PARTIAL. No formula tuning.',
            'ranking_sanity_selection':'at most two cells sorted by descending count of iterations with changed class-diversified Top-2, then cell id; exact frozen budget per search. Canonical routing in both arms to isolate ranking.',
            'generation':'unchanged frozen method_v2','Top_B':2,'runtime_objective':'unchanged NoiseAwareRuntimeObjective(1ms,.05); this existing accept/rollback rule is separate from new gate with no 5/10 percent threshold',
            'new_GPU_before_freeze':0}
    save('experiment_freeze.json',freeze)
    states=historical_states();save('historical_states.json',{'states':list(states.values()),'deduplicated_physical_repetitions':sum(len(s['repetitions']) for s in states.values())})
    ranker=E2ECandidateRanker(catalog,cal,factor);audit=[];replay=[];validation=[]
    def analyze(identifier,cell,x,route,origin,truth=None):
        w=Workload(grid[cell]['lambda_rps']);route=flat(route)
        before=evaluate_surrogate(x,route,w,catalog,cal)
        canonical=solve_canonical_routing(x,w,catalog,.8,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
        if canonical is None:return None
        fit=ranker.evaluate(x,w,canonical)
        assert fit['converged'],(identifier,fit)
        # The replay R0 is canonical for exact x. Labelled route is retained in audit.
        zero=evaluate_surrogate(x,canonical.routing,w,catalog,cal);after=fit['evaluation'];r1=fit['routing']
        distance=math.fsum(abs(r1.get(pair,0.)-canonical.routing.get(pair,0.)) for pair in set(r1)|set(canonical.routing))
        bytask={k:math.fsum(abs(r1.get((k,m),0.)-canonical.routing.get((k,m),0.)) for m in catalog) for k in w.active_tasks}
        row={'id':identifier,'cell':cell,'origin':origin,'deployment':x,'N_G':sum(bool(x.get(m)) for m in catalog if roles[m]=='generalist'),
             'N_S':sum(bool(x.get(m)) for m in catalog if roles[m]=='specialist'),'R0_J_seconds':zero['J_seconds'],'R1_J_seconds':after['J_seconds'],
             'R0_z':zero['z'],'R1_z':after['z'],'routing_L1_rps':distance,'routing_L1_over_Lambda':distance/before['Lambda'],
             'task_L1_rps':bytask,'task_shift_fraction':{k:bytask[k]/(2*w.arrival_rate[k]) for k in bytask},
             'R0_D':zero['D'],'R1_D':after['D'],'R0_rho':zero['rho'],'R1_rho':after['rho'],
             'predicted_mean_improvement_ms':1000*(zero['J_seconds']-after['J_seconds']),
             'R0_routing':serial(canonical.routing),'R1_routing':serial(r1),'solver_converged':fit['converged'],
             'optimality_gap_seconds':fit['optimality_gap_seconds'],'solve_seconds':fit['solve_seconds']}
        if truth:row.update(truth)
        replay.append(row);return row
    for key,s in states.items():
        before=evaluate_surrogate(s['deployment'],flat(s['routing']),Workload(grid[s['cell']]['lambda_rps']),catalog,cal)
        audit.append({'id':key,'cell':s['cell'],'deployment':s['deployment'],'z':before['z'],'J_seconds':before['J_seconds'],
                      'J_domain_valid':before['domain_valid'],'mean_e2e_ms':s['mean_e2e_ms'],'weighted_p95_ms':s['weighted_p95_ms'],
                      'stability':s['stability'],'instability_tier':2-TIER[s['stability']],'physical_repetitions':len(s['repetitions'])})
        analyze(key,s['cell'],s['deployment'],s['routing'],'historical_runtime',labels({'weighted_mean_e2e':s['mean_e2e_ms'],'weighted_p95_e2e':s['weighted_p95_ms'],'stability_state_v2':s['stability']}))
    report={}
    for subset,tiers in [('steady',['steady']),('steady_plus_recoverable',['steady','recoverable_congested']),('all_development',list(TIER))]:
        rows=[r for r in audit if r['stability'] in tiers]
        report[subset]={p:{t:correlations(rows,p,t) for t in ['mean_e2e_ms','weighted_p95_ms','instability_tier']} for p in ['z','J_seconds']}
        within=[]
        for cell in grid:
            within.extend((a,b) for a,b in itertools.combinations([r for r in rows if r['cell']==cell],2) if a['mean_e2e_ms']!=b['mean_e2e_ms'])
        report[subset]['within_cell_pairwise_mean_ordering']={p:{'pairs':len(within),'accuracy':sum(.5 if a[p]==b[p] else float((a[p]-b[p])*(a['mean_e2e_ms']-b['mean_e2e_ms'])>0) for a,b in within)/len(within) if within else None} for p in ['z','J_seconds']}
    csvsave('surrogate_audit.csv',audit);save('surrogate_audit.json',{'metrics':report,'states':len(audit),'J_domain_invalid':sum(not r['J_domain_valid'] for r in audit),'unit':'J in seconds; labels in ms','replication_unit':'deduplicated exact cell/deployment/routing; repetitions not independent correlation observations','pooled_intensity_confound':'cross-cell load levels can inflate correlations; within-cell ordering separately reported','stability_aggregation':'worst independent repetition tier; equal repetition means','no_final_heldout_data':True})
    candidate_rows=[];iterations=[];oracle=[];changes={}
    for tracepath in sorted(HISTORY.glob('C*/static_v2/search_complete.json')):
        cell=tracepath.parts[-3];trace=read(tracepath);w=Workload(grid[cell]['lambda_rps'])
        for tr in trace['iterations']:
            iteration=tr['iteration']+1;current=tr['deployment'];base=ranker.evaluate(current,w)
            feasible=[c for c in tr['candidates'] if c['canonical'] is not None and c['rank'] is not None]
            objects=[];mapping={}
            for c in feasible:
                mu=SimpleNamespace(**c['mutation']);obj=SimpleNamespace(mutation=mu,memory_mib=c['memory_mib'],delta_z=c['delta_z'],canonical=None)
                objects.append(obj);mapping[id(obj)]=c
                x=c['mutation']['deployment'];identifier=statekey(cell,x,flat(c['canonical']['routing']))
                if not any(row['id']==identifier for row in replay):analyze(identifier,cell,x,c['canonical']['routing'],'generated_candidate')
            old=diversified_order(objects,current,catalog,w,tr['deployment_memory_mib'])
            new=ranker.order(objects,current,catalog,w,tr['deployment_memory_mib'])
            oldkeys=[digest(c.mutation.deployment) for c in old[:2]];newkeys=[digest(c.mutation.deployment) for c in new[:2]]
            changed=set(oldkeys)!=set(newkeys)
            changes[cell]=changes.get(cell,0)+int(changed)
            measured=[]
            for obj in objects:
                c=mapping[id(obj)];x=obj.mutation.deployment;fit=ranker.evaluate(x,w)
                truth=c.get('runtime');key=statekey(cell,x,flat(c['canonical']['routing']))
                if truth:
                    labelled=labels(truth);source='measured_this_iteration'
                elif key in states:
                    s=states[key];labelled={k:s[k] for k in ['mean_e2e_ms','weighted_p95_ms','stability']};source='known_exact_historical_state'
                else:labelled={};source='UNMEASURED_NO_LABEL'
                row={'cell':cell,'iteration':iteration,'deployment':x,'operation':obj.mutation.operation,'class':candidate_class(obj),
                     'delta_z':obj.delta_z,'delta_J_seconds':obj.delta_J,'J_star_seconds':fit['evaluation']['J_seconds'],
                     'old_rank':old.index(obj)+1,'new_rank':new.index(obj)+1,'label_source':source,
                     'mean_improvement_ms':tr['current_runtime']['weighted_mean_e2e']-labelled['mean_e2e_ms'] if labelled else None,
                     'p95_improvement_ms':tr['current_runtime']['weighted_p95_e2e']-labelled['weighted_p95_ms'] if labelled else None,
                     **labelled}
                candidate_rows.append(row)
                if labelled:measured.append(row)
            # Evaluate Top-k only on actual known-label candidates, reordering that
            # restricted set by the same class P/R policies. Full-rank availability
            # is reported separately; unknown selected outcomes never imputed.
            def restricted(rows,score):
                if score=='old':return sorted(rows,key=lambda r:r['old_rank'])
                globalrows=sorted(rows,key=lambda r:(-r['delta_J_seconds'],ranker.evaluate(r['deployment'],w)['evaluation']['z'],factor*sum(cal.memory_table_mib[m][n] for m,n in r['deployment'].items()),tuple(sorted(r['deployment'].items()))))
                p=[r for r in globalrows if r['class']=='P'];r=[r for r in globalrows if r['class']=='R']
                first=p[0] if p else r[0] if r else None
                selected=[first] if first else [];second=next((a for a in r if a is not first),None)
                if second is None:second=next((a for a in globalrows if a is not first),None)
                if second:selected.append(second)
                return selected+[a for a in globalrows if all(a is not b for b in selected)]
            # Old known-set order is recomputed, because unlabelled candidates
            # removed from the class-diversified set can change reserved slots.
            knownobjects=[o for o in objects if any(digest(o.mutation.deployment)==digest(r['deployment']) for r in measured)]
            oldknown=diversified_order(knownobjects,current,catalog,w,tr['deployment_memory_mib'])
            oldmeasured=[next(r for r in measured if r['deployment']==o.mutation.deployment) for o in oldknown]
            newmeasured=restricted(measured,'new')
            best=min(measured,key=lambda r:(-TIER[r['stability']],r['mean_e2e_ms'] if r['stability']=='steady' else 0.,r['weighted_p95_ms'] if r['stability']=='steady' else 0.,factor*sum(cal.memory_table_mib[m][n] for m,n in r['deployment'].items()),tuple(sorted(r['deployment'].items())))) if measured else None
            record={'cell':cell,'iteration':iteration,'feasible_candidates':len(feasible),'known_labels':len(measured),'Top2_changed':changed,'old_Top2':oldkeys,'new_Top2':newkeys}
            for tag,ordered in [('old',oldmeasured),('new',newmeasured)]:
                for k in [1,2]:record[f'{tag}_Top{k}_hit']=any(r is best for r in ordered[:k]) if best else None
                record[f'{tag}_unstable_selected']=sum(r['stability']=='unstable' for r in ordered[:2]);record[f'{tag}_recoverable_or_steady_selected']=sum(r['stability']!='unstable' for r in ordered[:2]);record[f'{tag}_known_selection_slots']=len(ordered[:2])
            iterations.append(record)
            for direction,G in [('M5',['phi4mm','minicpm']),('M6',['qwen','minicpm'])]:
                oracle_x={m:int(m in G+['distilbert','resnet50','paddleocr','whisper-small','whisper_turbo']) for m in catalog}
                relevant=cell in ['C09','C12']
                exact=next((r for r in candidate_rows if r['cell']==cell and r['iteration']==iteration and r['deployment']==oracle_x),None)
                generated=any(c['mutation']['deployment']==oracle_x for c in tr['candidates'])
                toward=[r for r in candidate_rows if r['cell']==cell and r['iteration']==iteration and r['operation'] in ['ADD','SCALE_UP'] and sum(bool(r['deployment'].get(m)) for m in G)>sum(bool(current.get(m)) for m in G) and all(r['deployment'].get(m)==current.get(m) for m in catalog if m not in G)]
                oracle.append({'cell':cell,'iteration':iteration,'oracle_direction':direction,'relevant_loose_I3':relevant,'oracle_deployment':oracle_x,'exact_generated':generated,'exact_old_rank':exact['old_rank'] if exact else None,'exact_new_rank':exact['new_rank'] if exact else None,'second_G_ADD_generated':bool(toward),'second_G_ranks':[{'deployment':r['deployment'],'old':r['old_rank'],'new':r['new_rank']} for r in toward],
                               'diagnosis':'CANDIDATE_GENERATION_LIMIT' if not generated and not toward else 'GENERATED_RANKING_OR_FEASIBILITY_LIMIT','no_generation_change':True})
    # Validate all replay witnesses twice against identical fixed initialization.
    for row in replay:
        w=Workload(grid[row['cell']]['lambda_rps']);second=solve_e2e_routing(row['deployment'],w,catalog,cal)
        distance=max((abs(v-second['routing'].get(pair,0.)) for pair,v in flat(row['R1_routing']).items()),default=0.) if second['routing'] else None
        assert second['converged'] and distance<=1e-9,(row['id'],second)
        validation.append({'id':row['id'],'converged':True,'all_constraints_valid':second['constraints_valid'],'repeat_max_rate_difference':distance,'convex_gap_seconds':second['optimality_gap_seconds']})
    csvsave('routing_replay.csv',replay);save('routing_replay.json',{'rows':replay})
    save('routing_solver_validation.json',{'status':'PASS','validated_states':len(validation),'checks':validation,'random_restarts':0,'weights':None,'memory_deployment_workload_unchanged':True})
    csvsave('candidate_replay.csv',candidate_rows);save('candidate_replay.json',{'rows':candidate_rows,'iterations':iterations})
    csvsave('oracle_reachability_audit.csv',oracle)
    measured=[r for r in candidate_rows if r.get('mean_improvement_ms') is not None]
    metrics={'labelled_candidates':len(measured),'unlabelled_candidates':len(candidate_rows)-len(measured),'iteration_count':len(iterations),'Top2_changed_iterations':sum(r['Top2_changed'] for r in iterations),
             'known_label_selection_bias':'candidate outcomes only for historically measured/known exact states; no counterfactual labels for unmeasured candidates','metrics':{p:{t:correlations(measured,p,t) for t in ['mean_improvement_ms','p95_improvement_ms']} for p in ['delta_z','delta_J_seconds']}}
    valid_iterations=[r for r in iterations if r['known_labels']>=2]
    metrics['Topk_denominator_iterations_with_at_least_two_known_labels']=len(valid_iterations)
    for tag in ['old','new']:
        slots=sum(r[f'{tag}_known_selection_slots'] for r in valid_iterations)
        metrics[tag]={f'Top{k}_hit_rate':sum(r[f'{tag}_Top{k}_hit'] for r in valid_iterations)/len(valid_iterations) if valid_iterations else None for k in [1,2]}
        metrics[tag].update(unstable_selected_rate=sum(r[f'{tag}_unstable_selected'] for r in valid_iterations)/slots if slots else None,recoverable_or_steady_selected_rate=sum(r[f'{tag}_recoverable_or_steady_selected'] for r in valid_iterations)/slots if slots else None)
    # Within-iteration ordering avoids treating different baselines as independent.
    for p in ['delta_z','delta_J_seconds']:
        comparisons=[]
        for record in iterations:
            rs=[r for r in measured if r['cell']==record['cell'] and r['iteration']==record['iteration']]
            comparisons.extend((a,b) for a,b in itertools.combinations(rs,2) if a['mean_improvement_ms']!=b['mean_improvement_ms'])
        metrics.setdefault('within_iteration_ordering',{})[p]={'pairs':len(comparisons),'accuracy':sum(.5 if a[p]==b[p] else float((a[p]-b[p])*(a['mean_improvement_ms']-b['mean_improvement_ms'])>0) for a,b in comparisons)/len(comparisons) if comparisons else None}
    save('candidate_ranking_metrics.json',metrics)
    # Mechanical 9B selections use only frozen state characteristics and routes.
    historical=[r for r in replay if r['origin']=='historical_runtime'];selected=[]
    def choose(tag,predicate,reverse=False):
        options=[r for r in historical if predicate(r) and r['id'] not in {s['id'] for s in selected}]
        changed=[r for r in options if r['routing_L1_rps']>1e-7];options=changed or options
        if options:
            r=sorted(options,key=lambda r:((-r['R0_z'] if reverse else r['R0_z']),r['cell'],tuple(sorted(r['deployment'].items()))))[0]
            selected.append(dict(r,selection_slot=tag))
    choose('low_pressure_1G',lambda r:grid[r['cell']]['intensity_level']=='I1' and r['N_G']==1)
    choose('moderate_pressure_1G',lambda r:grid[r['cell']]['intensity_level']=='I2' and r['N_G']==1)
    choose('pressured_RC_SG_like',lambda r:grid[r['cell']]['intensity_level']=='I3' and r['N_G']==1 and any('/RC_SG/' in owner for owner in states[r['id']]['owners']),True)
    if len(selected)<3:choose('pressured_1G_fallback',lambda r:grid[r['cell']]['intensity_level']=='I3' and r['N_G']==1,True)
    choose('historical_multi_G',lambda r:r['N_G']>=2)
    assert len(selected)<=4
    save('routing_gpu_manifest.json',{'frozen_before_new_GPU':True,'at_utc':now(),'selection_rule':freeze['routing_state_selection'],'states':selected,'settings':read(PROTOCOL)['measurement_settings'],'seeds':freeze['routing_seeds'],'max_new_runs':6*len(selected),'new_GPU_results_inspected':False})
    sanity=sorted([cell for cell,count in changes.items() if count],key=lambda cell:(-changes[cell],cell))[:2]
    save('ranking_gpu_manifest.json',{'frozen_before_new_GPU':True,'at_utc':now(),'selection_rule':freeze['ranking_sanity_selection'],'selected_cells':sanity,'changed_iteration_counts':changes,'status':'PLANNED' if sanity else 'NO_MATERIAL_RANKING_DIFFERENCE','same_routing_both_arms':'canonical','search_budget':read(PROTOCOL)['search_budget'],'max_new_states':len(sanity)*2*6,'max_new_runs':len(sanity)*2*6*3,'reuse':'none across ranking arms; independently evaluate both arms; no confirmations added'})
    verify(refs)
    print({'historical_states':len(states),'routing_replay':len(replay),'ranking_metrics':metrics,'routing_selection':[(r['cell'],r['selection_slot'],r['routing_L1_rps']) for r in selected],'ranking_sanity_cells':sanity},flush=True)

if __name__=='__main__':main()
