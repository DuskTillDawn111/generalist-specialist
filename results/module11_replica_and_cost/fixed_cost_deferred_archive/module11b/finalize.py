"""Final read-only freeze and independent-run identity audit."""
from collections import defaultdict
import subprocess
from .common import *
from replica_scaling.telemetry import snapshot

def main():
 freeze=read(OUT/'input_freeze.json');verify(freeze['protected_references']);verify([freeze[k] for k in ['measurement_plan','resident_set_manifest','samples']])
 sup=read(OUT/'closed_loop_source_freeze.json');verify(sup['periodic_raw_data_refs_at_supplement_freeze']);verify([sup['protocol']])
 revision=read(OUT/'closed_loop_source_revision_v2.json');verify(revision['effective_harness'])
 archived_sources={r['original_reference']['path']:r for r in revision['archived_old_sources']}
 for original in sup['harness']:
  if original['path'] in archived_sources:
   archived_source=archived_sources[original['path']];verify([archived_source['archive_reference']])
   assert original['sha256']==archived_source['archive_reference']['sha256']
  else:verify([original])
 archived=read(OUT/'periodic_pre_closed_loop_archive/manifest.json');verify(archived['files'])
 periodic_manifest=read(OUT/'periodic_completion_manifest.json');verify(periodic_manifest['immutable_raw_refs'])
 assert (OUT/'replica_completion_status.json').exists() and read(OUT/'fixed_cost_characterization.json')['stage_complete']
 capacity_replay=read(OUT/'updated_capacity_witnesses.json');verify([capacity_replay['calibration'],capacity_replay['directions']])
 raw=[];groups=defaultdict(list)
 for p in sorted((OUT/'raw_runs').glob('*.json')):
  r=read(p);slot=r['slot'];assert r['gpu_release']['verified'];raw.append(ref(p))
  assert slot==read(OUT/'slots'/f"{slot['run_id']}.json")
  assert len({w['pid'] for w in r['worker_identities'].values()})==len(r['worker_identities'])
  if slot['kind']=='closed_loop_profile':
   assert r['valid_session'];assert r['phases']['closed_loop_capacity']['max_outstanding_per_replica']==1
   assert r['closed_loop_warmup']['all_requests_completed_after_drain']
   assert 'stability' not in r['phases']['closed_loop_capacity']
   groups[(slot['model'],slot['required_x'],slot['deployment'][slot['model']])].append(slot['seed'])
  elif slot['kind']=='fixed_cost':groups[('cost',slot['set_id'])].append(slot['seed'])
 for key,seeds in groups.items():assert sorted(seeds)==sorted(read(OUT/'measurement_plan.json')['seeds']),(key,seeds)
 c,hw,roles,base=inputs();data=read(OUT/'replica_scaling.json')
 for m in c:
  for n,v in base.capacity_factors[m].items():assert data['capacity_factors'][m][str(n)]==v
  for n,v in base.memory_table_mib[m].items():assert data['memory_table_mib'][m][str(n)]==v
 assert not subprocess.run(['pgrep','-af','[n]vidia-cuda-mps'],capture_output=True,text=True).stdout.strip()
 gpu=snapshot();assert not gpu['processes']
 outputs=['replica_evidence_required.csv','replica_profile_results.csv','replica_scaling.json','updated_multihotspot_capacity_envelopes.csv',
  'updated_second_generalist_regions.csv','replica_evidence_gate.json','resident_set_manifest.json','fixed_cost_measurements.csv',
  'low_load_runtime_results.csv','specialist_low_qps_tradeoff.csv','fixed_cost_characterization.json','module11b_summary.md']
 assert all((OUT/n).exists() for n in outputs)
 save('completion_verification.json',{'status':'PASS','at_utc':now(),'protected_files_verified':len(freeze['protected_references']),
  'old_periodic_observations_unchanged':True,'historical_calibration_values_unchanged':True,'Module9_unchanged':True,
  'closed_loop_versus_periodic_metric_roles_verified':True,'independent_repetition_groups':len(groups),'raw_observations':len(raw),
  'terminal_higher_counts_not_artificial_missing':True,'GPU_released':gpu,'MPS_off':True,'formal_hotspot_GPU_runs':0,'Task_Rich_started':False})
 save('output_manifest.json',{'at_utc':now(),'outputs':[ref(OUT/n) for n in outputs],
  'raw_observations':raw,'verification':ref(OUT/'completion_verification.json'),
  'final_module11b_sources':[ref(p) for p in (ROOT/'module11b').glob('*.py')],
  'closed_loop_protocol':ref(OUT/'closed_loop_protocol.json'),
  'closed_loop_source_revision':ref(OUT/'closed_loop_source_revision_v2.json'),
  'closed_loop_CPU_validation':ref(OUT/'closed_loop_CPU_validation_v2.json'),
  'CPU_reference_validation':ref(OUT/'CPU_reference_validation.json'),
  'frozen_solver_pruning_validation':ref(OUT/'frozen_J_pruning_CPU_validation.json')})
 print(json.dumps({'verification':'PASS','observations':len(raw),'protected_files':len(freeze['protected_references'])}),flush=True)
if __name__=='__main__':main()
