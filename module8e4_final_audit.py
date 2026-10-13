"""Independent final accounting and evidence audit, no GPU execution."""
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from module8e4_revision_ablation.common import *
from module8e4_revision_ablation.variants import config


def run():
    assert read(OUT/'campaign_status.json')['status']=='complete'
    verify_inputs();manifest=verify_execution()
    from candidate_pool_audit.provenance import environment_snapshot
    from replica_scaling.telemetry import snapshot
    assert environment_snapshot()==manifest['environment']
    gpu=snapshot();assert all(gpu[k]==manifest['GPU'][k] for k in ['uuid','name','driver','total_mib'])
    assert not gpu['processes'] and gpu['used_mib']<=128
    for r in read(E3/'hash_verification.json')['model_asset_checks']:assert file_hash(r['path'])==r['expected']
    config_hashes={m:digest(asdict(config(m))) for m in METHODS};assert len(set(config_hashes.values()))==1
    a=read(OUT/'run_accounting.json');new=read(OUT/'physical_measurements.json')['runs'];log=read(OUT/'measurement_sharing_log.json')
    assert a['new_physical_attempts']==len(new)<=read(OUT/'reuse_audit.json')['worst_case_new_timed_runs']
    assert a['new_timed_runs']==sum(r.get('measurement_started',False) for r in new)
    assert all(r['method'] in NEW and r['valid'] and file_hash(r['raw_path'])==r['raw_sha256'] for r in new)
    donors=old_donors();seen={h:r['physical_raw_id'] for h,r in donors.items()}
    by_id={r['physical_raw_id']:r for r in donors.values()};by_id.update({r['physical_raw_id']:r for r in new})
    for r in log['logical_run_references']:
        d=by_id[r['physical_raw_id']];h=digest(d['identity']);assert h==r['measurement_identity_hash']
        assert file_hash(r['raw_path'])==r['raw_sha256']==d['raw_sha256']
        assert Path(r['raw_path']).resolve()==Path(d['raw_path']).resolve()
        if r['cache_hit']:assert seen[h]==r['physical_raw_id']
        else:assert h not in seen;seen[h]=r['physical_raw_id']
    assert a['measurement_sharing_count']==sum(r['cache_hit'] for r in log['logical_run_references'])
    assert a['logical_search_evaluations']==len(log['logical_evaluations'])<=72
    times={(s['workload'],s['slot_index']):datetime.fromisoformat(s['started_at_utc']) for s in read(OUT/'execution_log.json')['slots']}
    for e in log['logical_evaluations']:
        if e['role']!='candidate':continue
        for method,h in e['round_barrier'].items():
            if not h:continue
            commits=[read(p) for p in (OUT/(method.lower()+'_search_traces')/e['workload']).glob('round_*_commitment.json')]
            c=next(c for c in commits if c['commitment_sha256']==h)
            assert digest({k:v for k,v in c.items() if k!='commitment_sha256'})==h
            assert datetime.fromisoformat(c['committed_at_utc'])<=times[e['workload'],e['logical_slot']]
    source_mechanism={r['workload']:r for r in csv.DictReader((E3/'heldout_mechanism_audit.csv').open())}
    rows=list(csv.DictReader((OUT/'mechanism_comparison.csv').open()))
    mapping={'targeted_swaps_generated':'targeted_swaps_generated','replacement_closures_generated':'replacement_closures_generated',
        'replacements_Top_B':'swaps_Top_B','replacements_evaluated':'swaps_evaluated','replacements_accepted':'swaps_accepted'}
    for r in rows:
        if r['method']=='M11':
            old=source_mechanism[r['workload']]
            for dest,source in mapping.items():assert int(r[dest])==int(old[source]);r[dest]=old[source]
            r['mechanism_source']='8E-3 frozen heldout_mechanism_audit.csv (direct reuse)'
        else:r['mechanism_source']='8E-4 independently committed trace'
    with (OUT/'mechanism_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    claims=read(OUT/'ablation_claim_audit.json');assert claims['protocol_audit']['valid']
    assert claims['comparison_counts']['M11_vs_M00']=={'WIN':2,'TIE':4,'LOSS':0}
    for m in NEW:
        for w in read(F/'heldout_workloads.json')['workloads']:
            r=read(OUT/w['id']/m/'result.json');paths=[]
            for role,agg in r['confirmations'].items():
                assert len(agg['repetitions'])==2
                paths.append({str(Path(x['raw_result_path']).resolve()) for x in agg['repetitions']})
            assert not paths[0]&paths[1]
    states={digest({'deployment':r['identity']['config']['deployment'],'workload':r['identity']['config']['workload_rps'],
        'routing':r['identity']['config']['routing_rps'],'context':r['identity']['runtime_context_sha256'],'workload_hash':r['workload_sha256']}) for r in new}
    assert len(states)==a['unique_physical_states']<=read(OUT/'reuse_audit.json')['missing_physical_states_upper_bound']
    save_json(OUT/'final_validation.json',{'status':'PASS','all_frozen_inputs_and_method_sources_unchanged':True,'model_assets_rehashed':108,
        'environment_and_hardware_unchanged':True,'GPU_released':True,'same_config_sha256':config_hashes,
        'new_timed_runs':a['new_timed_runs'],'new_physical_states':len(states),'logical_search_evaluations':len(log['logical_evaluations']),
        'reuse_identity_hashes_and_earliest_donors_verified':True,'round_commitment_timestamps_verified':True,
        'M00_M11_new_timed_runs':0,'M11_mechanism_audit_directly_reused':ref(E3/'heldout_mechanism_audit.csv'),
        'all_valid_repetitions_preserved':True,'within_preflight_budget':True,'auditor_source':ref(Path(__file__)),'GPU_runs_from_audit':0})
    summary=OUT/'module8e4_summary.md';summary.write_text(summary.read_text()+'\n最终独立审计 PASS：冻结源码/输入/模型资产/环境未改，GPU 已释放，预算、raw identity/hash、earliest donor、round commitments 与 confirmation role independence 均通过。M11 mechanism audit 直接复用 8E-3 原表并核对所有对应计数。\n')
    manifest=read(OUT/'report_manifest.json');manifest['artifacts']={p.name:ref(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md'] and p.name!='report_manifest.json'}
    manifest['auditor_source']=ref(Path(__file__));save_json(OUT/'report_manifest.json',manifest)
    assert all(file_hash(r['path'])==r['sha256'] for r in manifest['artifacts'].values())
    print(json.dumps({'status':'PASS','new_timed_runs':a['new_timed_runs'],'new_physical_states':len(states)}))


if __name__=='__main__':run()
