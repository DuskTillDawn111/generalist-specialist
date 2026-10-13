"""CPU-only recheck of Module21 solutions; no fitting or new optimization."""
from .common import *
import copy
from module21.optimizer import PoissonUtility
from module21.trajectory import choose

def main():
 assert not (OUT/'frozen_experiment_manifest.json').exists(), 'Freeze cannot be replaced'
 src=ROOT/'results/module21_positive_utility'
 inherited=read(src/'frozen_manifest.json');cpu=read(src/'cpu_fullpool_optimization.json')
 assert inherited['status']=='FROZEN_BEFORE_GPU' and cpu['Stage_A_passed']
 definition=read(inherited['utility_definition']['path']);references=read(inherited['utility_references']['path'])
 assert definition['weights']=={'Q':0.,'T':.25,'L':.30,'P':.30,'M':.15}
 catalog=read(inherited['catalog']['path']);plan=copy.deepcopy(inherited['plan'])
 assert set(plan['models'])==set(MODELS)
 refs=[inherited[k] for k in ['CPU','input_protocol','utility_definition','utility_references','quality_policy','catalog','sample_pool']]
 refs+=inherited['implementation']+inherited['reused_implementation']
 for r in refs:assert ref(r['path'])['sha256']==r['sha256'],r['path']
 protected=[]
 for p in sorted((ROOT/'results').rglob('*')):
  if p.is_file() and OUT not in p.parents:protected.append(ref(p))
 for folder in ['module20','module21','module14f']:
  protected.extend(ref(p) for p in sorted((ROOT/folder).glob('*.py')))
 save('protected_before.json',protected)
 predictor=PoissonUtility(catalog,references['latency_reference_ms'],definition['weights'],dt=.00025,bins=65536)
 checks=[];slots=[]
 for stage,rates in RATES.items():
  st=cpu['stages'][stage]
  assert st['rates']==rates
  a,b=st['winner'],st['control']
  if stage=='MID':
   assert MMS in a['deployment'] and MMS not in b['deployment']
   assert b['winner']['routing']['lid']=={G:3.}
  else:
   assert a['deployment']==b['deployment']
   assert all(a['winner']['routing'][t]==b['winner']['routing'][t] for t in rates if t!='lid')
   assert b['winner']['routing']['lid']=={MMS:12.}
   assert abs(a['shares'][2]-.08456165089208065)<1e-14
  for arm,fit in [('proposed',a),('control',b)]:
   x,p=fit['deployment'],fit['winner'];routing=p['routing']
   assert all(m in MODELS and str(n) in catalog['memory_table_mib'][m] and str(n) in catalog['capacity_factors'][m] for m,n in x.items())
   assert all(m in x and t in catalog['models'][m]['supported_tasks'] for t,row in routing.items() for m,q in row.items() if q>0)
   assert all(abs(sum(routing[t].values())-q)<1e-6 for t,q in rates.items())
   assert set(routing)==set(rates)
   mem=catalog['safe_factor']*sum(catalog['memory_table_mib'][m][str(n)] for m,n in x.items())
   assert mem<=BUDGET and abs(mem-p['safe_memory_mib'])<1e-6
   recalculated=predictor.evaluate(x,rates,fit['shares'],True)
   assert recalculated['routing']==routing
   assert recalculated['eligible']==p['eligible']
   assert abs(recalculated['U']-p['U'])<1e-10
   checks.append({'stage':stage,'arm':arm,'deployment':x,'routing':routing,'shares':fit['shares'],
                  'CPU_nominal_feasible':p['eligible'],'capacity_diagnostic_only':not p['eligible'],
                  'CPU_rho':p['rho'],'safe_memory_mib':mem,'budget_mib':BUDGET,'CPU_U':p['U'],
                  'recomputed_CPU_U':recalculated['U'],'admitted_paths_only':True,'quality_policy_sha256':inherited['quality_policy']['sha256'],
                  'actual_GPU_feasibility':'Pending measured memory and worker/system stability, never inferred from CPU rho',
                  'count_constraints_pass':True})
  for rep in [1,2,3]:
   for arm in (['proposed','control'] if rep!=2 else ['control','proposed']):
    inherited_slot=next(s for s in inherited['slots'] if s['stage']==stage and s['repetition']==rep and s['arm']==arm)
    fit=a if arm=='proposed' else b
    assert inherited_slot['deployment']==fit['deployment'] and inherited_slot['routing']==fit['winner']['routing']
    slot=copy.deepcopy(inherited_slot);slot['run_id']=f'{stage.lower()}_{arm}_rep{rep}'
    slot['source_slot_run_id']=inherited_slot['run_id'];slot['module']='22'
    slot['CPU_nominal_feasible']=fit['winner']['eligible'];slot['capacity_diagnostic_only']=not fit['winner']['eligible']
    slot['CPU_per_task']=fit['winner']['per_task']
    for label,treference in slot['trajectories'].items():
     assert ref(treference['path'])['sha256']==treference['sha256']
     trace=read(treference['path']);assert trace['rates']==rates and trace['duration']==(10. if label=='warmup' else 120.)
     assert all(0<=e['offset_seconds']<trace['duration'] for e in trace['events'])
    events=read(slot['trajectories']['measurement']['path'])['events']
    slot['expected_realized_routed_counts_from_frozen_trace']={t:{m:sum(e['task']==t and choose(e,slot['routing'],slot['deployment']).rsplit('#',1)[0]==m for e in events) for m in slot['deployment']} for t in rates}
    slots.append(slot)
 recheck={'at_utc':now(),'Stage_A_passed':True,'CPU_only':True,'no_new_optimization':True,'source_CPU':inherited['CPU'],
          'checks':checks,'HIGH_G_LID_share':cpu['stages']['HIGH']['winner']['shares'][2],
          'HIGH_only_LID_route_differs':True,'HIGH_resident_models_identical':True,
          'MID_predicted_delta_U':cpu['stages']['MID']['predicted_delta_U'],
          'HIGH_frozen_fine_predicted_delta_U':cpu['stages']['HIGH']['predicted_delta_U'],
          'HIGH_user_simulation_reference_delta_U':.04095,
          'nominal_controls_with_capacity_deficit':[],
          'warning':'Standalone service mismatch from Module21D remains; CPU feasibility and predicted U are hypotheses, not GPU observations.'}
 for rep in [1,2,3]:
  a=next(s for s in slots if s['stage']=='HIGH' and s['repetition']==rep and s['arm']=='proposed')
  b=next(s for s in slots if s['stage']=='HIGH' and s['repetition']==rep and s['arm']=='control')
  assert a['trajectories']==b['trajectories']
  for e in read(a['trajectories']['measurement']['path'])['events']:
   if e['task']!='lid':assert choose(e,a['routing'],a['deployment'])==choose(e,b['routing'],b['deployment'])
  realized=a['expected_realized_routed_counts_from_frozen_trace']['lid']
  assert realized[G]>0 and realized[MMS]>0
 save('cpu_constraint_recheck.json',recheck)
 protocol={'repetitions':3,'max_new_GPU_runs':12,'warmup_seconds':10.,'measurement_seconds':120.,
           'drain_timeout_seconds':60.,'idle_seconds':2.,'counterbalance':['P/C','C/P','P/C'],
           'trace_rule':'Reuse the still-unexecuted Module21 frozen MID/HIGH Poisson task/sample/offset/uniform trajectories without resampling or screening',
           'stability_thresholds':read(src/'input_protocol_freeze.json')['GPU_protocol']['stability_thresholds'],
           'MID_positive_rule':'Three comparable finite deltas all >0; Proposed steady and both arms eligible. Report nominal paired 95% CI; no additional CI acceptance gate.',
           'HIGH_positive_rule':'Three comparable finite deltas all >0 with mean>0.01; both arms steady and eligible; real nonzero G/S completed LID split in every Proposed repetition; identical residency and non-LID routes. Report nominal paired 95% CI; no additional CI acceptance gate.',
           'no_benefit_rule':'MID: three comparable deltas all<=0. HIGH: three comparable deltas all<=0.01. Otherwise, or if missing/ineligible comparisons, INCONCLUSIVE. No negative population claim from n=3.',
           'stop_rule':'First hard unstable worker/system or incomplete cohort stops remaining pairs of that stage; retain every outcome and report missing pairs. MID utility failure alone never stops HIGH. Memory/resource, request-generator, isolation, runtime or release failure stops campaign for safety. No retries, no window extension.',
           'quality_rule':'Original frozen hard model/path admission and output semantics; no new quality estimates or altered thresholds.',
           'unstable_U':'null; completed-subset diagnostic components cannot form a comparable delta or utility win',
           'no_full_three_phase_claim':True,'no_LOW_repair':True,'no_Module23':True}
 reused=refs+[ref(ROOT/'module20/optimizer.py'),ref(ROOT/'module21/trajectory.py')]
 save('frozen_experiment_manifest.json',{'at_utc':now(),'status':'FROZEN_BEFORE_GPU','module':'22','source_Module21':ref(src/'frozen_manifest.json'),
      'source_Module21D_decision':ref(ROOT/'results/module21_low_diagnosis/next_experiment_decision.json'),
      'CPU':inherited['CPU'],'cpu_recheck':ref(OUT/'cpu_constraint_recheck.json'),
      **{k:inherited[k] for k in ['catalog','utility_definition','utility_references','quality_policy','sample_pool']},
      'plan':plan,'model_pool':MODELS,'budget_mib':BUDGET,'rates':RATES,'slots':slots,'GPU_protocol':protocol,
      'implementation':[ref(p) for p in sorted((ROOT/'module22').glob('*.py'))],'reused_implementation':reused,
      'protected_before':ref(OUT/'protected_before.json'),'no_historical_modules_modified':True})
 print(json.dumps(recheck,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
