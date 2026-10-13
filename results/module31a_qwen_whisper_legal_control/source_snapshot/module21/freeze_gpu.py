"""Called only after Stage A passes; freeze paired native workload and exact routes."""
from .common import *
from .trajectory import arrivals
import copy

def main():
 c=read(OUT/'cpu_fullpool_optimization.json');assert c['Stage_A_passed'],'NO GPU when Stage A fails';assert not (OUT/'frozen_manifest.json').exists()
 f=read(OUT/'input_protocol_freeze.json');protocol=copy.deepcopy(f['GPU_protocol']);protocol['benefit_rule']='All 3 finite delta_U>0.01, mean>0.01 and same positive direction. Paired nominal CI reported as uncertainty, no additional CI pass threshold. Unstable controls U=null and infeasible, capacity evidence separate.';protocol['stop']='Repeated instability in either arm (>=2 runs) stops later stages; fatal runtime/memory/release error stops immediately. After each completed stage, stop if any of the 3 paired utility gains is unavailable or <=0.01. No retries or input changes.';plan=copy.deepcopy(read(ROOT/'results/module16_fullpool_gpu/frozen_gpu_manifest.json')['plan']);d=read(f['catalog']['path']);plan['models']=d['models'];plan['warmup_seconds']=10.;plan['measurement_seconds']=120.;plan['drain_timeout_seconds']=60.;plan['shutdown_timeout_seconds']=15.;plan['telemetry_interval_seconds']=.5;plan['environment_overrides'].pop('CUDA_VISIBLE_DEVICES',None)
 for record in [f['catalog'],f['utility_definition'],f['utility_references'],f['quality_policy']]:assert ref(record['path'])['sha256']==record['sha256']
 from joint_deploy.measurement.load_generator import SamplePool
 trajectories=[];slots=[]
 for stage,rates in RATES.items():
  st=c['stages'][stage]
  for rep,seed in enumerate(f['GPU_protocol']['seeds'],1):
   refs={}
   for label,duration in [('warmup',10.),('measurement',120.)]:
    pool=SamplePool(plan['sample_pool']['path'],list(rates),seed);events=arrivals(rates,duration,seed,label)
    for event in events:event['item']=pool.next(event['task'])
    name=f'trajectories/{stage.lower()}_rep{rep}_{label}.json';save(name,{'rates':rates,'duration':duration,'seed':seed,'label':label,'events':events});refs[label]=ref(OUT/name)
   trajectories.append({'stage':stage,'repetition':rep,'seed':seed,'references':refs})
   arms=['proposed','control'] if rep!=2 else ['control','proposed']
   for arm in arms:
    fit=st['winner'] if arm=='proposed' else st['control'];p=fit['winner'];x=fit['deployment']
    assert all(m in plan['registry'] for m in x)
    assert p['eligible'] and p['safe_memory_mib']<=BUDGET
    slots.append({'run_id':f'{stage.lower()}_{arm}_rep{rep}','stage':stage,'arm':arm,'repetition':rep,'seed':seed,'deployment':x,'routing':p['routing'],'lambda_rps':rates,'safe_memory_budget_mib':BUDGET,'CPU_safe_memory_mib':p['safe_memory_mib'],'CPU_rho':p['rho'],'CPU_predicted_U':p['U'],'CPU_components':{k:v for k,v in p.items() if k.endswith('_norm') or k.endswith('_contribution')},'measurement_seconds':120.,'trajectories':refs,'arrival_process':'independent_per_task_Poisson','paired_same_trajectory':True})
 reused=[ref(ROOT/p) for p in ['joint_deploy/measurement/deployment_runtime.py','joint_deploy/measurement/schema.py','joint_deploy/measurement/metrics.py','joint_deploy/measurement/run_measurement.py','module11b/measure.py','module16/gpu.py','replica_scaling/telemetry.py','performance_model/local_calibration/stability_v2.py']]
 reused.extend(ref(plan['registry'][m]['script']) for m in MODELS)
 save('frozen_manifest.json',{'at_utc':now(),'status':'FROZEN_BEFORE_GPU','input_protocol':ref(OUT/'input_protocol_freeze.json'),'CPU':ref(OUT/'cpu_fullpool_optimization.json'),'utility_definition':f['utility_definition'],'utility_references':f['utility_references'],'quality_policy':f['quality_policy'],'catalog':f['catalog'],'sample_pool':plan['sample_pool'],'plan':plan,'trajectories':trajectories,'slots':slots,'max_formal_runs':18,'implementation':[ref(p) for p in sorted((ROOT/'module21').glob('*.py'))],'reused_implementation':reused,'GPU_protocol':protocol,'pre_GPU_acceptance_clarification':ref(OUT/'acceptance_clarification.json'),'no_scientific_inputs_changed':True})
 print('FROZEN_GPU',len(slots),'runs',flush=True)
if __name__=='__main__':main()
