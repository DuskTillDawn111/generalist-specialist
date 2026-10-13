from .common import *

def main():
 f=read(OUT/'frozen_gpu_manifest_v2.json');history=read(OUT/'history_integrity_after.json');checks=dict(history_SHA_and_mtime_preserved=history['passed'],quality_preserved=read(OUT/'quality_and_profile_preservation.json')['passed'],independent_CPU_verification=read(OUT/'CPU_oracle_correctness.json')['passed'])
 changed=[]
 for r in f['sources']+f['native_sources']+f['inputs']:
  if ref(r['path'])['sha256']!=r['sha256']:changed.append(r['path'])
 checks['GPU_frozen_execution_inputs_sources_preserved']=not changed
 cs=read(OUT/'candidate_space_definition.json');A=set(cs['members']['A']);B=set(cs['members']['B']);C=set(cs['members']['C']);checks['strict_nested_pools']=A<B<C
 checks['frozen_pool_files_preserved']=all(ref(r['path'])['sha256']==r['sha256'] for r in cs['pools'].values())
 checks['original_utility_sources_preserved']=all(ref(read(OUT/'frozen_problem_manifest.json')[k]['path'])['sha256']==read(OUT/'frozen_problem_manifest.json')[k]['sha256'] for k in ['utility_definition','utility_reference','quality_policy'])
 decisions=read(OUT/'optimizer_decisions.json');checks['all_native_V3_results_preserved']=len([r for r in decisions if r['algorithm']=='V3_NATIVE_CONTINUOUS_PRESERVED'])==9
 obs=read(OUT/'gpu_observations.json') if (OUT/'gpu_observations.json').exists() else [];checks['historical_secondary_analysis_source_frozen']=ref(OUT/'m34/reuse33_independent.py')['sha256']==read(OUT/'historical_secondary_analysis_plan.json')['source_sha256']
 checks['statistical_analysis_source_frozen']=ref(OUT/'m34/analyze_gpu.py')['sha256']==read(OUT/'statistical_analysis_plan.json')['analysis_source_sha256']
 checks['ineligible_no_legal_U']=all(o['U'] is None for o in obs if not o['eligible'])
 checks['run_ids_unique']=len({o['run_id'] for o in obs})==len(obs)
 checks['no_full_pool_GPU_oracle_claim']=not read(OUT/'module34_gate.json')['GPU']['full_pool_oracle']
 checks['required_outputs_exist']=all((OUT/p).exists() for p in ['frozen_problem_manifest.json','candidate_space_definition.json','feasible_deployments.csv','cpu_oracle_results.csv','optimizer_decisions.csv','predicted_regret.csv','optimizer_runtime_benchmark.csv','gpu_oracle_runs.csv','gpu_empirical_regret.csv','ranking_error_analysis.md','module34_gate.json','module34_summary.md','figures/deployment_selection.png','figures/oracle_regret.png','figures/search_time_space.png','figures/gpu_optimizer_heuristic.png','figures/prediction_error.png'])
 cpu_fm=read(OUT/'frozen_problem_manifest.json');checks['CPU_decision_sources_preserved']=all(ref(r['path'])['sha256']==r['sha256'] for r in cpu_fm['source_hashes'] if Path(r['path']).name!='common.py')
 before=(OUT/'preflight_source_common_v1.py.txt').read_text();after=(OUT/'m34/common.py').read_text();checks['original_common_snapshot_matches_CPU_freeze']=ref(OUT/'preflight_source_common_v1.py.txt')['sha256']==next(r['sha256'] for r in cpu_fm['source_hashes'] if Path(r['path']).name=='common.py')
 checks['common_only_documented_serialization_alias']=after.startswith(before) and after[len(before):].strip()=='# Serialization alias expected by unchanged native monitoring adapter.\njson_native=native'
 save('final_audit.json',dict(passed=all(checks.values()),checks=checks,changed_frozen_paths=changed,observed_windows=len(obs),frozen_windows=len(f['slots']),scope='Integrity and evidence reporting audit, not universal scientific-success gate.'))
 # Record all scientific artifacts; omit mutable work caches/logs/locks.
 paths=[]
 for p in OUT.rglob('*'):
  if p.is_file() and 'work' not in p.relative_to(OUT).parts and p.name not in ['scientific_artifact_sha256.json','campaign.lock'] and p.suffix!='.log':paths.append(ref(p))
 save('scientific_artifact_sha256.json',dict(files=paths,created_unix_ns=time.time_ns(),history=history))
 print('FINAL_AUDIT',all(checks.values()),checks,flush=True)
 if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
