"""CPU-only structural replay with measured values and evidence-bound counts."""
import concurrent.futures,copy,hashlib,json,math,time
from pathlib import Path
from module12.common import baseline
from module12.solver import StructuralSolver
from module12.metrics import describe
from deployment_core.models import ModelProfile
from deployment_core.replica_calibration import ReplicaCalibration
from module13b.analyze import read,save,csvsave
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/module13_real_specialists';PLAN=ROOT/'results/module12_specialist_sensitivity/experiment_freeze.json'
def catalogs():
    original,hw,roles,cal=baseline();selection=read(OUT/'balanced_specialist_selection.json');assert selection['selection_valid']
    replace={task for task,s in selection['tasks'].items() if s['status']=='SELECTED_NEW_BALANCED'}
    expanded={}
    for name,p in original.items():
        # A shared old checkpoint retained for another task keeps every frozen
        # admitted capability. Never mask ASR from a retained LID Whisper.
        retain=roles[name]!='specialist' or bool(p.supported_tasks-replace)
        if retain:expanded[name]=p
    r={m:roles[m] for m in expanded};g={m:copy.deepcopy(cal.capacity_factors[m]) for m in expanded};mem={m:copy.deepcopy(cal.memory_table_mib[m]) for m in expanded};bounds={m:cal.evidence_bounds[m] for m in expanded};evidence={m:copy.deepcopy(cal.model_evidence[m]) for m in expanded};groups={}
    selected_ids=set()
    for task,s in selection['tasks'].items():
        if s['status']!='SELECTED_NEW_BALANCED':continue
        w=s['winner'];selected_ids.add(w['model_id']+'@'+w['revision'])
    # A selected shared new checkpoint also keeps every quality-admitted frozen
    # measured path, with that path's own s0/Q. No unmeasured task is inferred.
    for f in read(OUT/'real_pareto_frontiers.json')['tasks'].values():
        for w in f['universe']:
            name=w['model_id']+'@'+w['revision']
            if name in selected_ids and w['origin']=='NEW_MEASURED_CANDIDATE' and w['selection_eligible']:groups.setdefault(name,[]).append(w)
    for name,paths in groups.items():
        idle=max(w['idle_memory_mib'] for w in paths)
        expanded[name]=ModelProfile(name,frozenset(w['task'] for w in paths),idle,{w['task']:w['s0_ms'] for w in paths},{'measurement':'Module13B real standalone','operating_points_sha256':hashlib.sha256((OUT/'real_operating_points.csv').read_bytes()).hexdigest(),'shared_idle_policy':'max observed per-path mean, no double-counting'})
        r[name]='specialist';g[name]={0:0.,1:1.};mem[name]={0:0.,1:idle};bounds[name]=1;evidence[name]={'replica_scaling_status':'REAL_STANDALONE_ONLY','measured_replica_counts':[1],'unknown_gt1':True}
    expandedcal=ReplicaCalibration(g,mem,bounds,evidence,dict(cal.source_hashes)).validate(expanded)
    return original,roles,cal,expanded,r,expandedcal

def context(job):
    workload,budget=job;original,roles,cal,expanded,newroles,newcal=catalogs();solver=StructuralSolver();rows=[];fits={};identical_input_cache={}
    def matched(c,rs,ac):
        bound={m:1 if rs[m]=='specialist' and p.supported_tasks&{'asr','lid','text'} else ac.evidence_bounds[m] for m,p in c.items()}
        return ReplicaCalibration({m:{n:v for n,v in ac.capacity_factors[m].items() if n<=bound[m]} for m in c},{m:{n:v for n,v in ac.memory_table_mib[m].items() if n<=bound[m]} for m in c},bound,copy.deepcopy(ac.model_evidence),dict(ac.source_hashes)).validate(c)
    for scope in ['PRIMARY_EVIDENCE_BOUND','MATCHED_SEARCHED_SPECIALIST_COUNTS_0_1']:
      for regime,c,rs,ac in [('EFFICIENCY_OPTIMAL_SPECIALIST_REGIME',original,roles,cal),('BALANCED_SPECIALIST_REGIME',expanded,newroles,newcal)]:
        ac=matched(c,rs,ac) if scope.startswith('MATCHED') else ac
        fingerprint=json.dumps({'catalog':{m:{'tasks':sorted(p.supported_tasks),'s0':dict(p.service_time_ms),'idle':p.memory_mib,'role':rs[m]} for m,p in c.items()},'g':ac.capacity_factors,'memory':ac.memory_table_mib,'bounds':ac.evidence_bounds},sort_keys=True)
        if fingerprint in identical_input_cache:
            fit=copy.deepcopy(identical_input_cache[fingerprint]);fit['identical_math_input_witness_reused']=True
        else:
            fit=solver.solve(workload['rates'],c,rs,ac,budget['safe_memory_budget_mib']);identical_input_cache[fingerprint]=copy.deepcopy(fit)
        fit['evidence_scope']='REAL_MEASURED_STANDALONE_PLUS_UNCHANGED_LEGACY_REPLICA_EVIDENCE_CPU_REPLAY';fits[scope+'::'+regime]=fit
        row={'workload':workload['id'],'workload_regime':workload['regime'],'budget':budget['id'],'memory_budget_mib':budget['safe_memory_budget_mib'],'Specialist_regime':regime,'count_scope':scope,**describe(fit,workload,rs,c),'frozen_Module9_solver':fit.get('frozen_Module9_solver')}
        if row['feasible']:
            total=math.fsum(workload['rates'].values());row['Generalist_routed_fraction']=math.fsum(workload['rates'][k]*v for k,v in row['task_G_fraction'].items())/total
            row['G_served_task_sets']={m:sorted({path.split('::')[0] for path,v in row['routing'].items() if v>1e-9 and path.split('::')[1]==m}) for m in row['Generalist_identities']}
        rows.append(row)
    cache=OUT/'real_replay_witnesses';cache.mkdir(exist_ok=True);p=cache/(workload['id']+'__'+budget['id']+'.json');p.write_text(json.dumps({'rows':rows,'fits':fits},indent=2)+'\n')
    return rows

def analyze(rows):
    matchedrows=[r for r in rows if r['count_scope'].startswith('MATCHED')]
    primaryrows=[r for r in rows if r['count_scope']=='PRIMARY_EVIDENCE_BOUND']
    rows=primaryrows
    comparisons=[]
    for work in read(PLAN)['workloads']:
        for b in read(PLAN)['budgets']:
            a=next(r for r in rows if r['workload']==work['id'] and r['budget']==b['id'] and r['Specialist_regime']=='EFFICIENCY_OPTIMAL_SPECIALIST_REGIME');z=next(r for r in rows if r['workload']==work['id'] and r['budget']==b['id'] and r['Specialist_regime']=='BALANCED_SPECIALIST_REGIME')
            result={'workload':work['id'],'budget':b['id'],'comparable_feasible':a['feasible'] and z['feasible'],'R1':False,'R2':False,'R3':False,'R4':False,'R5':False,'R6':False,'new_second_Generalist':False}
            if result['comparable_feasible']:
                new_splits=sorted(set(z['split_tasks'])-set(a['split_tasks']));increased=[k for k in z['task_G_fraction'] if z['task_G_fraction'][k]>a['task_G_fraction'][k]+1e-9]
                consolidated=[m for m,tasks in z['G_served_task_sets'].items() if len(tasks)>=2 and any(t not in a['G_served_task_sets'].get(m,[]) for t in tasks)]
                result.update(R1=z['N_S']<a['N_S'],R2=z['Generalist_routed_fraction']>a['Generalist_routed_fraction']+1e-9,R3=bool(new_splits),R4=bool(consolidated),R5=z['N_G']>a['N_G'],R6=a['replicas']!=z['replicas'],new_second_Generalist=a['N_G']<2<=z['N_G'],new_split_tasks=new_splits,task_G_share_increased=increased,consolidated_G_ids=consolidated,delta_NS=z['N_S']-a['N_S'],delta_NG=z['N_G']-a['N_G'],delta_G_share=z['Generalist_routed_fraction']-a['Generalist_routed_fraction'],delta_J_seconds=z['J_star_seconds']-a['J_star_seconds'],delta_safe_memory_mib=z['safe_memory_mib']-a['safe_memory_mib'],numerical_certificate_gap_seconds=max(a['global_gap_seconds'],z['global_gap_seconds']))
            comparisons.append(result)
    selected=read(OUT/'balanced_specialist_selection.json');new=[s['winner'] for s in selected['tasks'].values() if s['status']=='SELECTED_NEW_BALANCED'];decisions=[]
    for w in new:
        name=w['model_id']+'@'+w['revision'];active=[r for r in rows if r['Specialist_regime']=='BALANCED_SPECIALIST_REGIME' and r['feasible'] and name in r['Specialist_identities']]
        decisions.append({'model_id':w['model_id'],'revision':w['revision'],'task':w['task'],'decision':'PROFILE_SELECTED_STANDALONE_WINNER_GT1_NEXT_STAGE' if active else 'NO_REPLICA_PROFILING_SUPPORTED_NOW','resident_optimum_contexts':[r['workload']+'/'+r['budget'] for r in active],'reason':'Selected candidate enters measured-count optimum; unknown higher counts need separate real profiling before any claim' if active else 'No optimum residency or measured evidence of boundary relevance','profiling_executed':False})
    counts={k:sum(r[k] for r in comparisons) for k in ['R1','R2','R3','R4','R5','R6','new_second_Generalist']}
    matchedchanges=[]
    for work in read(PLAN)['workloads']:
      for b in read(PLAN)['budgets']:
        a=next(r for r in matchedrows if r['workload']==work['id'] and r['budget']==b['id'] and r['Specialist_regime']=='EFFICIENCY_OPTIMAL_SPECIALIST_REGIME');z=next(r for r in matchedrows if r['workload']==work['id'] and r['budget']==b['id'] and r['Specialist_regime']=='BALANCED_SPECIALIST_REGIME');both=a['feasible'] and z['feasible']
        matchedchanges.append({'workload':work['id'],'budget':b['id'],'both_feasible':both,'Specialist_reduction':both and z['N_S']<a['N_S'],'G_share_increase':both and z['Generalist_routed_fraction']>a['Generalist_routed_fraction']+1e-9,'new_split':bool(set(z['split_tasks'])-set(a['split_tasks'])) if both else False,'replica_change':a.get('replicas')!=z.get('replicas') if both else False})
    robust=any(r['G_share_increase'] or r['new_split'] or r['Specialist_reduction'] for r in matchedchanges)
    save('real_transition_analysis.json',{'contexts':comparisons,'event_context_counts':counts,'comparability':'Same27 contexts, same optimizer; EFFICIENCY unchanged, BALANCED replaces only task paths with selected new winners','new_model_count_scope':'0/1 only; no unmeasured g(x>1) claim','matched_count_diagnostic':matchedchanges,'matched_count_transition_observed':robust,'replica_profiling_decision':decisions,'no_replica_profiling_executed':True,'R_definitions':{'R1':'Specialist replica count reduction','R2':'total workload weighted Generalist routing share increase','R3':'new task split','R4':'Generalist newly serving multiple tasks','R5':'Generalist replica count increase','R6':'replica allocation/identity changes'}})
    allretained=not new;structural=any(counts[k] for k in ['R1','R2','R3','R4','R5','R6'])
    # The distance is the immutable frozen search target metric. Its zero
    # baseline endpoint is proximity only, never itself a transition claim.
    # Gate proximity scale is locked before GPU measurements, anchored to the
    # Module12 first-memory bracket width. It never changes model selection.
    located=[w for w in new if w['target_distance']<=1e-12]
    proximity=read(OUT/'module13b_execution_freeze.json')['gate_proximity_policy'];near=[w for w in new if w['target_distance']<=proximity['near_log_distance_max']]
    gate='BASELINE_RETAINED' if allretained else 'REAL_BALANCED_REGIME_SUPPORTED' if near and structural and robust else 'REAL_BALANCED_REGIME_PARTIAL'
    prediction='PARTIAL_MATCH' if allretained else 'PREDICTION_MATCH' if near and structural and robust else 'PARTIAL_MATCH' if structural or not near else 'PREDICTION_MISMATCH'
    save('sensitivity_prediction_validation.json',{'status':prediction,'selected_real_points':new,'literal_target_set_points':[w['model_id'] for w in located],'composition_or_routing_change_observed':structural,'explanation':'All frozen-rule task winners are old baselines; no selected new operating point tests the counterfactual transition. The baseline endpoint has distance0 but is not an observed transition.' if allretained else 'Evaluate actual joint latency/memory point; global whole-pool scaling brackets do not guarantee a single-checkpoint transition. Unknown replica counts remain excluded. Exact gamma thresholds are not expected to coincide.','prediction_falsified':False if allretained else None,'Module12':'COUNTERFACTUAL_SEARCH_TARGET','Module13B_measurements':'REAL_LOCAL_Q_S0_M','replay':'CPU_STRUCTURAL_NOT_GPU_DEPLOYMENT','global_guidance':{'first_split_gamma':[1.2125,1.25],'stronger_gamma':[1.9375,2.0],'first_memory_mu':[1.,1.05]}})
    save('module13b_gate.json',{'gate':gate,'complete':True,'proximity_policy':proximity,'near_selected_points':[w['model_id']+'@'+w['revision'] for w in near],'literal_target_set_points':[w['model_id']+'@'+w['revision'] for w in located],'matched_count_transition_observed':robust,'selected_new_task_count':len(new),'baseline_retained_tasks':[task for task,s in selected['tasks'].items() if s['status']=='BASELINE_RETAINED'],'structural_event_counts':counts,'selected_models':{task:s['winner']['model_id'] for task,s in selected['tasks'].items()},'replica_profiling_next':decisions,'not_final_deployment_evidence':True,'no_counterfactual_values_used_for_new_candidates':True})
    return gate,counts

def main():
    c,r,cal,new,nr,ncal=catalogs()
    def serialize(c,roles,cal):return {'models':{m:{'supported_tasks':sorted(p.supported_tasks),'idle_memory_mib':p.memory_mib,'service_time_ms':dict(p.service_time_ms),'role':roles[m]} for m,p in c.items()},'capacity_factors':cal.capacity_factors,'memory_table_mib':cal.memory_table_mib,'evidence_bounds':cal.evidence_bounds}
    save('real_replay_catalogs.json',{'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME':serialize(c,r,cal),'BALANCED_SPECIALIST_REGIME':serialize(new,nr,ncal),'safe_factor':read(OUT/'module13b_execution_freeze.json')['safe_factor'],'policy':read(OUT/'module13b_execution_freeze.json')['regime_catalog_policy']})
    plan=read(PLAN);jobs=[(w,b) for w in plan['workloads'] for b in plan['budgets']];assert len(jobs)==27
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(context,job) for job in jobs]
        for f in concurrent.futures.as_completed(futures):
            result=f.result();rows.extend(result);print('REPLAY',result[0]['workload'],result[0]['budget'],len(rows),flush=True)
    rows.sort(key=lambda r:(r['workload'],r['budget'],r['count_scope'],r['Specialist_regime']));csvsave('real_structural_replay.csv',[r for r in rows if r['count_scope']=='PRIMARY_EVIDENCE_BOUND']);csvsave('real_structural_replay_matched_counts.csv',[r for r in rows if r['count_scope'].startswith('MATCHED')]);print('GATE',analyze(rows),flush=True)
if __name__=='__main__':main()
