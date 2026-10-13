"""CPU-only one-hop audit using unchanged existing MILP/LP/generation/ranking.
Never constructs or calls RuntimeEvaluator, a search loop, or a GPU worker.
"""
import csv,collections,copy,math
from dataclasses import fields
from .common import *
from deployment_core.models import ModelProfile,Hardware
from deployment_core.workload import Workload
from deployment_core.milp_seeder import MilpDeploymentSeeder,validate_seed
from deployment_core.replica_calibration import ReplicaCalibration,ReplicaPlanningPolicy,load_replica_calibration
from deployment_core.replicas import replica_statistics
from deployment_search.routing import solve_canonical_routing
from deployment_search.feedback import bottleneck_feedback
from deployment_search.neighborhood import generate_replica_neighborhood,missing_coverage
from deployment_search.result import CandidateEvaluation,jsonable
from deployment_search.scoring import rank_candidates

def freeze_plan():
 design={'module':'8B','frozen_before_planning_outcomes':True,'rho_max':.8,'top_b':2,'remove_pressure_threshold':.2,
         'iterations':'one-hop from each independent MILP seed; no transitions or search',
         'seed_objective':'existing minimum discrete memory, then minimum replica count',
         'ranking':'existing exact empirical delta_z primary, deterministic model-name ties; predictor=None',
         'inputs':{'workloads':ref(QUALITY/'proposed_demand_normalized_workloads.json'),'quality_policy':ref(OUT/'quality_policy.json'),
                   'quality_admission':ref(OUT/'quality_admission_matrix.json'),'candidate_pool':ref(POOL_OUT/'candidate_pool_v2_1.json'),
                   'real_service_demand':ref(POOL_OUT/'service_demand_v2_1.json'),
                   'empirical_capacity':ref(ROOT/'results/replica_capacity_calibration/replica_capacity_factors.json'),
                   'replica_bounds':ref(ROOT/'results/replica_capacity_calibration/evidence_replica_bounds.json')},
         'diversity_gate_definition':{
           'status_values':['DIVERSIFIED','NOT_DIVERSIFIED'],
           'rule':'DIVERSIFIED only if the frozen suite exhibits distinct dominant bottleneck-task identities, distinct nonempty Top-B model-family sets, and both Generalist and Specialist addition/replication patterns across workloads. Otherwise NOT_DIVERSIFIED.',
           'interpretation':'Any repeated regimes and majority-concentrated Top-B sets are reported explicitly; DIVERSIFIED does not require all six seeds/regimes to be unique.',
           'no_outcome_tuning':'lambda vectors and target shares are immutable; no iterative planning-based workload edits'},
         'generalist_witness_method':'For each G, reuse MILP with other Generalists unavailable and the same admitted Specialist edges; add one G if no-VQA seed omits it, then verify canonical LP and memory. Diagnostic witnesses are not ranked candidates or real runtime evidence.',
         'runtime_evaluator_allowed':False,'gpu_search_allowed':False,'enumeration_allowed':False}
 write(OUT/'planning_protocol.json',design,freeze=True)

def restrict(calibration,names):
 return ReplicaCalibration({m:calibration.capacity_factors[m] for m in names},
                           {m:calibration.memory_table_mib[m] for m in names},
                           {m:calibration.evidence_bounds[m] for m in names},
                           {m:calibration.model_evidence[m] for m in names},calibration.source_hashes)
def csvwrite(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def active(x):return ', '.join(f'{m}={n}' for m,n in sorted(x.items()) if n)
def full(x,all_names):return {m:x.get(m,0) for m in all_names}
def category(c,seed,roles):
 m=c.mutation.model
 if c.mutation.operation=='SCALE_DOWN':return 'scale_down'
 return ('new_generalist' if roles[m]=='generalist' else 'new_specialist') if seed.get(m,0)==0 else ('generalist_replica' if roles[m]=='generalist' else 'specialist_replica')
def canonical(x,w,catalog,bounds,cal):return solve_canonical_routing(x,w,catalog,.8,replica_bounds=bounds,capacity_mode='empirical',calibration=cal)

def main():
 completeness=read(POOL_OUT/'completeness.json')
 validation=read(POOL_OUT/'vqa_service_demand_validation.json')
 if completeness['completeness_ratio']!=1 or validation['status']!='PASS' or not validation['part_D_allowed']:raise RuntimeError('No Part D with pending new measurements')
 freeze_plan()
 proposal_path=QUALITY/'proposed_demand_normalized_workloads.json';proposal_hash=sha(proposal_path)
 admission=read(OUT/'quality_admission_matrix.json');admitted={(r['task'],r['model']) for r in admission['rows'] if r['quality_admissible']}
 pool=read(POOL_OUT/'candidate_pool_v2_1.json');service=read(POOL_OUT/'service_demand_v2_1.json')['models']
 roles={m['id']:m['role'] for m in pool['models'] if m['id'] in pool['formal_model_ids']}
 all_names=sorted(roles)
 raw_catalog={m:ModelProfile(m,frozenset(v['capabilities_verified']),v['measured_idle_vram_mib'],v['service_time_ms'],
                            {'completed_catalog':ref(POOL_OUT/'service_demand_v2_1.json'),'memory':ref(ROOT/'results/candidate_pool_v2/measured_vram.json')}) for m,v in service.items()}
 catalog={}
 quality_disabled=[]
 for m,v in raw_catalog.items():
  tasks=frozenset(k for k in v.supported_tasks if (k,m) in admitted)
  if not tasks:quality_disabled.append(m);continue
  catalog[m]=ModelProfile(m,tasks,v.memory_mib,{k:v.service_time_ms[k] for k in tasks},dict(v.sources,quality_policy=ref(OUT/'quality_policy.json')))
 # Load original empirical calibration over all 12 model identities, then use an
 # exact subset view for the existing solver. No factor or bound is recalibrated.
 raw_cal=load_replica_calibration(raw_catalog)
 cal=restrict(raw_cal,catalog).validate(catalog)
 policy=ReplicaPlanningPolicy(catalog,'empirical','empirical',cal)
 hardware=Hardware(49140,'Frozen RTX 4090',ref(ROOT/'results/candidate_pool_v2/pool_resource_summary.json'))
 bounds,provenance=policy.bounds(hardware)
 expected=read(ROOT/'results/replica_capacity_calibration/evidence_replica_bounds.json')
 # Preserve original g(x), M(x), evidence bounds; quality only restricts edges.
 assert bounds=={m:expected['replica_bounds'][m] for m in catalog}
 write(OUT/'planning_catalog.json',{'models':{m:{'name':v.name,'supported_tasks':sorted(v.supported_tasks),'memory_mib':v.memory_mib,'service_time_ms':dict(v.service_time_ms),'sources':v.sources} for m,v in catalog.items()},'formal_model_ids':all_names,
                                  'quality_disabled_model_variables':quality_disabled,'note':'Models remain in v2.1 inventory; model with no admitted routing edges has fixed zero count in all output vectors',
                                  'bounds':bounds,'bound_provenance':provenance,'capacity_factors':cal.capacity_factors,'memory_table_mib':cal.memory_table_mib},freeze=True)
 summaries=[];competition=[];details=[];solvercounts={'milp_calls':0,'canonical_lp_calls':0,'runtime_evaluations':0,'enumeration_calls':0}
 for proposal in read(proposal_path)['proposals']:
  name=proposal['name'];w=Workload(proposal['lambda_rps'])
  seed=MilpDeploymentSeeder().solve(catalog,w,hardware,.8,replica_bounds=bounds,capacity_mode='empirical',memory_mode='empirical',calibration=cal)
  solvercounts['milp_calls']+=seed.milp_call_count
  if not seed.feasible:raise RuntimeError('Frozen workload has no admitted seed: '+name)
  errors=validate_seed(seed,catalog,w,hardware,.8,bounds,planning_policy=policy)
  assert not errors,errors
  current=dict(seed.deployment);route=canonical(current,w,catalog,bounds,cal);solvercounts['canonical_lp_calls']+=1
  assert route is not None
  feedback=bottleneck_feedback(route,current)
  mutations=generate_replica_neighborhood(current,catalog,w,feedback,bounds,.2)
  assert len(mutations)<=2*len(catalog)
  candidates=[];feasible=[]
  for mut in mutations:
   c=CandidateEvaluation(mut)
   c.memory_mib=policy.deployment_memory(mut.deployment)
   if c.memory_mib>hardware.gpu_memory_mib:
    c.status='REJECT_INFEASIBLE';c.reason='memory'
   elif missing_coverage(mut.deployment,catalog,w):
    c.status='REJECT_INFEASIBLE';c.reason='coverage'
   else:
    c.canonical=canonical(mut.deployment,w,catalog,bounds,cal);solvercounts['canonical_lp_calls']+=1
    if c.canonical is None:c.status='REJECT_INFEASIBLE';c.reason='canonical_capacity_lp'
    else:
     c.delta_z=route.max_pressure-c.canonical.max_pressure
     c.status='LP_FEASIBLE'
     increment=policy.memory(mut.model,mut.deployment[mut.model])-policy.memory(mut.model,current[mut.model])
     if mut.operation=='SCALE_UP':c.relief_per_memory=c.delta_z/(increment/hardware.gpu_memory_mib) if increment>0 else None
     else:c.memory_freed_mib=-increment
     feasible.append(c)
   candidates.append(c)
  ranking=rank_candidates(feasible)
  for rank,c in enumerate(ranking,1):c.rank=rank;c.selected_top_b=rank<=2;c.status='SELECTED_TOP_B' if c.selected_top_b else 'PRUNED_TOP_B'
  selected=ranking[:2]
  counts=replica_statistics(full(current,all_names),roles)
  bottleneck_tasks={}
  for (task,m),value in route.task_contributions.items():
   if m in feedback.bottleneck_models and value>0:bottleneck_tasks[task]=bottleneck_tasks.get(task,0)+value
  dominant=[k for k,v in bottleneck_tasks.items() if math.isclose(v,max(bottleneck_tasks.values()),rel_tol=1e-7,abs_tol=1e-10)]
  coverage_models={k:[m for m,v in catalog.items() if k in v.supported_tasks] for k in w.active_tasks}
  mandatory_qwen=any(ms==['qwen'] for ms in coverage_models.values())
  # All VQA-capable models must be admitted under the user's declared policy.
  if w.arrival_rate['visual_qa']>0:assert set(coverage_models['visual_qa'])=={'qwen','phi4mm','minicpm'}
  summary={'workload':name,'seed':active(current),'seed_full_vector':json.dumps(full(current,all_names),sort_keys=True),
           'N_G':counts['N_G'],'N_S':counts['N_S'],'seed_memory_mib':seed.objective_value,
           'seed_max_pressure':route.max_pressure,'bottleneck_models':','.join(feedback.bottleneck_models),
           'bottleneck_tasks':','.join(feedback.pressure_tasks),'dominant_bottleneck_tasks':','.join(dominant),
           'Top_1':active(selected[0].mutation.deployment) if selected else '',
           'Top_1_mutated_model':selected[0].mutation.model if selected else '',
           'Top_1_category':category(selected[0],current,roles) if selected else '',
           'Top_1_delta_z':selected[0].delta_z if selected else None,
           'Top_2':active(selected[1].mutation.deployment) if len(selected)>1 else '',
           'Top_2_mutated_model':selected[1].mutation.model if len(selected)>1 else '',
           'Top_2_category':category(selected[1],current,roles) if len(selected)>1 else '',
           'Top_2_delta_z':selected[1].delta_z if len(selected)>1 else None,
           'qwen_mandatory':mandatory_qwen,'runtime_evaluations':0,'generated_candidates':len(mutations),
           'quality_disabled_model_variables':','.join(quality_disabled)}
  summaries.append(summary)
  witnesses={}
  for g in ('qwen','phi4mm','minicpm'):
   names=[m for m in catalog if roles[m]=='specialist' or m==g]
   sub={m:catalog[m] for m in names};subcal=restrict(cal,names).validate(sub);subbounds={m:bounds[m] for m in names}
   subpolicy=ReplicaPlanningPolicy(sub,'empirical','empirical',subcal)
   fixedseed=MilpDeploymentSeeder().solve(sub,w,hardware,.8,replica_bounds=subbounds,capacity_mode='empirical',memory_mode='empirical',calibration=subcal)
   solvercounts['milp_calls']+=fixedseed.milp_call_count
   fixed=None;witness=None
   if fixedseed.feasible:
    fixed=dict(fixedseed.deployment)
    if fixed[g]==0:fixed[g]=1
    if subpolicy.deployment_memory(fixed)<=hardware.gpu_memory_mib:
     witness=canonical(fixed,w,sub,subbounds,subcal);solvercounts['canonical_lp_calls']+=1
   applicable_quality=[r['task'] for r in admission['rows'] if r['model']==g and r['blocked_by_quality_threshold'] and r['task'] in w.active_tasks]
   absent_capability=[k for k in w.active_tasks if k not in raw_catalog[g].supported_tasks]
   g_candidates=[c for c in candidates if c.mutation.model==g]
   gc={'workload':name,'model':g,'feasible_with_admitted_specialists':witness is not None,
       'VQA_quality_admissible':('visual_qa',g) in admitted,'quality_policy_for_VQA':'SEMANTIC_VALIDITY_ONLY',
       'present_in_seed':current[g]>0,'generated_as_candidate':bool(g_candidates),
       'generated_scale_up':any(c.mutation.operation=='SCALE_UP' for c in g_candidates),
       'Top_B':any(c.selected_top_b for c in g_candidates),
       'blocked_by_memory':any(c.reason=='memory' for c in g_candidates),
       'blocked_by_quality_as_VQA_choice':('visual_qa',g) not in admitted,
       'blocked_by_capability_as_VQA_choice':'visual_qa' not in raw_catalog[g].supported_tasks,
       'quality_blocked_active_tasks':','.join(applicable_quality),'unsupported_or_pending_active_tasks':','.join(absent_capability),
       'candidate_rejection_reasons':','.join(sorted({c.reason for c in g_candidates if c.reason})),
       'single_generalist_witness':active(fixed) if witness else '',
       'single_generalist_witness_memory_mib':subpolicy.deployment_memory(fixed) if witness else None,
       'single_generalist_witness_pressure':witness.max_pressure if witness else None,
       'qwen_absent_witness_legal':witness is not None and (g!='qwen')}
   competition.append(gc)
   witnesses[g]={'summary':gc,'deployment':full(fixed,all_names) if witness else None,'canonical':jsonable(witness) if witness else None}
  # Direct zero-Generalist capability/memory/capacity feasibility witness.
  spec_names=[m for m in catalog if roles[m]=='specialist']
  spec_catalog={m:catalog[m] for m in spec_names};spec_cal=restrict(cal,spec_names).validate(spec_catalog)
  spec_policy=ReplicaPlanningPolicy(spec_catalog,'empirical','empirical',spec_cal)
  spec_bounds={m:bounds[m] for m in spec_names}
  specseed=MilpDeploymentSeeder().solve(spec_catalog,w,hardware,.8,replica_bounds=spec_bounds,capacity_mode='empirical',memory_mode='empirical',calibration=spec_cal)
  solvercounts['milp_calls']+=specseed.milp_call_count
  summary['zero_generalist_feasible']=specseed.feasible
  # CSV rows all gain the same field before write.
  detail={'workload':name,'lambda_rps':dict(w.arrival_rate),'seed':jsonable(seed),'canonical':jsonable(route),'bottleneck':jsonable(feedback),
          'candidate_evidence':[{**jsonable(c),'category':category(c,current,roles)} for c in candidates],
          'ranked_models':[c.mutation.model for c in ranking],'Top_B_models':[c.mutation.model for c in selected],
          'Top_B_categories':[category(c,current,roles) for c in selected],
          'generalist_witnesses':witnesses,'zero_generalist_seed':jsonable(specseed),'summary':summary}
  details.append(detail)
 assert sha(proposal_path)==proposal_hash
 csvwrite('six_workload_planning_summary.csv',summaries);csvwrite('generalist_competition.csv',competition)
 write(OUT/'planning_details.json',{'workloads':details,'solver_counts':solvercounts,'workload_source':ref(proposal_path),
                                 'lambda_vectors_modified':False,'quality_constraints_method':'filter unadmitted routing edges before unchanged existing solvers; empty-edge model fixed to zero',
                                 'inference_scope':'planning witnesses/pressure rankings only, no real performance measurements'},freeze=True)
 seeds={r['seed'] for r in summaries}
 top_sets={tuple(sorted(d['Top_B_models'])) for d in details}
 bottleneck_sets={tuple(sorted(r['bottleneck_models'].split(','))) for r in summaries}
 dominant_sets={tuple(sorted(r['dominant_bottleneck_tasks'].split(','))) for r in summaries}
 categories=set(c for d in details for c in d['Top_B_categories'])
 g_pattern=bool(categories&{'new_generalist','generalist_replica'})
 s_pattern=bool(categories&{'new_specialist','specialist_replica'})
 diversified=len(dominant_sets)>1 and len(top_sets)>1 and g_pattern and s_pattern
 histogram=collections.Counter(tuple(sorted(d['Top_B_models'])) for d in details)
 diversity={'status':'DIVERSIFIED' if diversified else 'NOT_DIVERSIFIED','definition':read(OUT/'planning_protocol.json')['diversity_gate_definition'],
            'seed_diversity':{'distinct_count':len(seeds),'signatures':sorted(seeds)},
            'Top_B_diversity':{'distinct_model_family_set_count':len(top_sets),'sets':sorted([list(x) for x in top_sets]),
                               'histogram':[{'models':list(k),'workload_count':v} for k,v in histogram.items()]},
            'bottleneck_diversity':{'distinct_model_set_count':len(bottleneck_sets),'model_sets':sorted([list(x) for x in bottleneck_sets]),
                                   'distinct_dominant_task_set_count':len(dominant_sets),'dominant_task_sets':sorted([list(x) for x in dominant_sets])},
            'Generalist_Specialist_competition_patterns':{'Generalist_mutation_in_Top_B':g_pattern,'Specialist_mutation_in_Top_B':s_pattern,'candidate_categories':sorted(categories)},
            'workloads':summaries,'lambda_vectors_modified':False,'source_hash_unchanged':proposal_hash==sha(proposal_path),
            'scope':'Distinct planning regimes in the frozen suite; does not imply six unique deployments or runtime-optimal choices'}
 write(OUT/'workload_diversity.json',diversity,freeze=True)
 print(json.dumps({'status':diversity['status'],'seed_count':len(seeds),'Top_B_sets':diversity['Top_B_diversity']['sets'],'summaries':summaries,'solver_counts':solvercounts},indent=2))
if __name__=='__main__':main()
