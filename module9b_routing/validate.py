import json,math,time
from pathlib import Path
from copy import deepcopy
from deployment_core.interference.scheduling_pressure import SchedulingPressureEstimator
from deployment_core.interference.bounded_routing import solve_fixed_deployment_routing
from module9a_scheduling_tax.common import ROOT,load,sha,now,verify

OUT=ROOT/'results/module9b_routing_iteration'
MODEL=ROOT/'results/module9a_scheduling_tax'
def save(name,value):
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def check_result(result,initial,estimator,workload):
 assert result['iterations']<=3
 assert result['final_score'][0]<=result['initial_score'][0]+1e-9
 assert result['memory_safe_mib']<=result['memory_budget_mib']+1e-6
 for k,arrival in workload.items():
  assert math.isclose(sum(result['routing_rps'].get(k,{}).values()),arrival,rel_tol=1e-7,abs_tol=1e-10)
  assert all(rate>=0 and (m,k) in estimator.by_pair and result['deployment'].get(m,0)>0 for m,rate in result['routing_rps'].get(k,{}).items())
 for step in result['trace']:
  if step['accepted']:assert step['candidate_score'][0]<=step['previous_score'][0]+1e-9
 assert not result['coefficients_refitted'] and not result['actual_runtime_or_E2E_certificate']

def main():
 assert not (OUT/'algorithm_freeze.json').exists(),'Do not overwrite the 9B experiment freeze'
 assert load(MODEL/'model_selection.json')['9A_complete']
 estimator=SchedulingPressureEstimator.load(MODEL)
 selected_hash=sha(MODEL/'fitted_model.json');features_hash=sha(MODEL/'path_features.json')
 catalog=load(ROOT/'results/module9a_interference/catalog.json')
 memory={p['model']:p['safe_memory_mib_by_replicas'] for p in catalog['paths']}
 cells=load(ROOT/'results/module8f1b_intensity_grid_freeze/campaign_cells.json')['cells']
 cases=[]
 for cell in cells:
  for method in ['static_v2','RC_SG']:
   source=ROOT/'results/module8f2_resource_static'/cell['cell_id']/method/'result.json';old=load(source)
   assert old['status']=='COMPLETE'
   x=old['deployment'];routing={k:{} for k in cell['lambda_rps']}
   for key,rate in old['routing'].items():
    k,m=key.split('::')
    if rate:
     assert (m,k) in estimator.by_pair and x.get(m,0)>0
     routing[k][m]=rate
   cases.append({'id':cell['cell_id']+'__'+method,'cell':cell['cell_id'],'method':method,'deployment':x,'workload':cell['lambda_rps'],'initial_routing':routing,'memory_budget_mib':cell['memory_budget_mib'],'source':str(source),'source_sha256':sha(source)})
 freeze={'at_utc':now(),'scope':'9B_FIXED_DEPLOYMENT_BOUNDED_ROUTING_CPU_ONLY','selected_9A_model':estimator.variant,'selected_model_sha256':selected_hash,'features_sha256':features_hash,'algorithm':'freeze current effective service coefficients; lexicographic LP minimizes peak effective utilization then weighted service time; damp proposal, re-evaluate routing-conditioned costs, accept only surrogate improvement; retain best witness','parameters':{'rho_max':.8,'max_iterations':3,'damping':.5,'runtime_budget_seconds':6},'stop_conditions':['no improvement','three iterations','runtime budget','LP timeout/infeasible surrogate','numerical cycle'],'memory_g_quality_workload_deployment_unchanged':True,'no_global_optimality_or_fixed_point_convergence_claim':True,'no_actual_capacity_or_E2E_claim':True,'9A_features_or_coefficients_refitted':False,'candidate_ranking_Module9C':False,'GPU_runs':0,'sources':[{'path':str(p),'sha256':sha(p)} for p in [Path(__file__),ROOT/'deployment_core/interference/bounded_routing.py',ROOT/'deployment_core/interference/scheduling_pressure.py']],'cases':cases}
 save('algorithm_freeze.json',freeze)
 results=[]
 for case in cases:
  initial=deepcopy(case['initial_routing']);x=deepcopy(case['deployment']);workload=deepcopy(case['workload'])
  result=solve_fixed_deployment_routing(estimator,x,workload,initial,memory,case['memory_budget_mib'])
  check_result(result,initial,estimator,workload)
  assert initial==case['initial_routing'] and x==case['deployment'] and workload==case['workload']
  result['case_id']=case['id'];save(case['id']+'.json',result);results.append(result)
 # Nonlinear evaluator control-flow test: the archived ablation coefficients
 # exercise re-evaluation without changing the selected production model.
 coupled=SchedulingPressureEstimator.load(MODEL,'I2')
 x={'qwen':1,'phi4mm':1,'whisper-small':1};workload={'asr':.01,'visual_qa':.01,'lid':.01}
 initial={'asr':{'qwen':.005,'whisper-small':.005},'visual_qa':{'qwen':.005,'phi4mm':.005},'lid':{'qwen':.005,'whisper-small':.005}}
 test=solve_fixed_deployment_routing(coupled,x,workload,initial,memory,49140)
 check_result(test,initial,coupled,workload);save('coupled_evaluator_control_flow_test.json',test)
 zero=solve_fixed_deployment_routing(estimator,{'qwen':1},{'asr':0.},{'asr':{}},memory,49140)
 assert zero['routing_rps']=={'asr':{}} and zero['final_score']==[0.,0.]
 for route,budget in [({'asr':{'qwen':.01}},1),({'asr':{'qwen':.02}},49140),({'asr':{'distilbert':.01}},49140)]:
  try:solve_fixed_deployment_routing(estimator,{'qwen':1},{'asr':.01},route,memory,budget)
  except ValueError:pass
  else:raise AssertionError('Memory/conservation/unsupported guard failure')
 # A surrogate infeasible workload keeps the input witness and its status,
 # rather than proving the physical deployment infeasible or dropping it.
 heavy=solve_fixed_deployment_routing(estimator,{'qwen':1},{'asr':100.},{'asr':{'qwen':100.}},memory,49140)
 assert not heavy['feasible_by_surrogate'] and heavy['stop_reason']=='SURROGATE_LP_INFEASIBLE'
 assert heavy['routing_rps']=={'asr':{'qwen':100.}}
 assert selected_hash==sha(MODEL/'fitted_model.json') and features_hash==sha(MODEL/'path_features.json')
 verify(freeze['sources'])
 changed=sum(r['routing_rps']!=c['initial_routing'] for r,c in zip(results,cases))
 improved=sum(r['final_score'][0]<r['initial_score'][0]-1e-9 for r in results)
 summary={'status':'PASS','scope':freeze['scope'],'selected_model':estimator.variant,'fixed_cases':len(results),'routing_changed_cases':changed,'peak_surrogate_utilization_improved_cases':improved,'surrogate_feasible_cases':sum(r['feasible_by_surrogate'] for r in results),'max_iterations_observed':max(r['iterations'] for r in results),'max_runtime_seconds':max(r['runtime_seconds'] for r in results),'all_arrival_rates_conserved':True,'admitted_support_and_replica_bounds_preserved':True,'safe_memory_budgets_preserved':True,'initial_routing_deployment_workload_not_mutated':True,'best_re_evaluated_surrogate_never_worsened':True,'nonlinear_control_flow_zero_load_and_invalid_input_tests_pass':True,'surrogate_infeasibility_not_physical_pruning':True,'selected_9A_assets_unchanged':True,'production_I0_has_no_calibrated_interference_benefit_claim':estimator.variant=='I0','new_GPU_runs':0,'worker_runtime_MPS_changes':False,'Module9C_started':False,'no_global_optimality_or_actual_E2E_claim':True}
 save('validation_results.json',summary)
 (OUT/'module9b_summary.md').write_text(f'''# Module 9B 固定 deployment routing 原型\n\n9A 选择 {estimator.variant}。9B 不重新拟合、增加或选择 predictor features；production 使用该固定模型。\n\n实现 bounded routing：冻结当前 service coefficients，LP 最小化 peak normalized effective utilization，再以 weighted mean service time 打破 tie；以 damping 0.5 更新候选，重新调用纯 evaluator，只有 surrogate 改善才保留。最多 3 iterations、6 秒预算；无改善、LP timeout/infeasible surrogate 或 numerical cycle 即停止，保留最佳 witness。没有全局最优或固定点收敛保证。\n\n验证了 {len(results)} 个 8F-2 development states（12 个冻结 cells × v2/RC-SG 各自的固定 deployment）。只更新 routing；workload、memory budget、quality admission、replica bounds、empirical g 和 deployment 都保持原值。{changed} 个 routing 发生变化，{improved} 个 peak surrogate utilization 降低。所有 conservation/support/memory/immutability 与 best-witness 检查通过。\n\n另测已存在 I2 ablation coefficient 下的 nonlinear evaluator control flow、zero workload、memory/unsupported/conservation guards，以及 surrogate infeasible 的 witness 保留；未改变 production selection。\n\nI0 是 no-interference baseline，因此这些 CPU routing 结果不构成 task-conditioned interference modeling 成功或实际 serving/E2E 改善证据。其余模型的迁移预测未获得 9A 的统计支持。9A 的 3+ models additive/busy-probability extrapolation 也未被 GPU 验证。\n\n新增 GPU runs = 0；未启用 MPS，未修改 worker/runtime、v2、RC-SG 或冻结结果。没有进入 9C candidate ranking。\n''')
 save('artifact_hashes.json',{p.name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
 print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
