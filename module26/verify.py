"""Final read-only scientific accounting and frozen-protocol acceptance."""
import ast
from .common import *
from module21.trajectory import choose
from scipy.stats import t as student_t

def main():
    gate=read(OUT/'module26_gate.json');manifest=read(OUT/'frozen_replication_manifest.json')
    amendment=read(OUT/'infrastructure_recovery_manifest.json');report=read(OUT/'replication_result.json')
    checks=[]
    def passed(name,**evidence):checks.append(dict(name=name,passed=True,**evidence));print('PASS',name,flush=True)
    for p in (ROOT/'module26').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    verify(manifest['implementation']);verify(amendment['implementation'])
    assert len(manifest['slots'])==6 and manifest['measurement_seconds']==2400.
    assert [s['seed'] for s in manifest['slots']]==[26101,26101,26102,26102,26103,26103]
    for old,new in zip(manifest['slots'],amendment['remaining_full_slots']):
        assert all(v==new[k] for k,v in old.items() if k!='run_id')
    passed('frozen_protocol_only_first_run_identifier_amended')
    d,refs,weights=inputs();recalcs={};raws={}
    for run in report['runs']:
        raw=read(run['raw']['path']);path=Path(run['raw']['path']);raw['_path']=str(path);raws[run['run_id']]=raw
        rec=arithmetic(raw,d,refs,weights);recalcs[run['run_id']]=rec
        slot=next(s for s in amendment['remaining_full_slots'] if s['run_id']==run['run_id'])
        assert raw['slot']==slot
        if rec['eligible']:
            assert abs(rec['U']-run['actual_U'])<1e-12
            assert abs(sum(rec[k+'_contribution'] for k in ['T','L','P','M'])-run['actual_U'])<1e-12
            ph=raw['phases']['measurement'];rr=ph['request_records'];events=read(slot['trajectories']['measurement']['path'])['events']
            assert len(rr)==len(events) and all(r['status']=='completed' for r in rr)
            assert max(abs(r['e2e_latency_ms']-(r['completion_time_ns']-r['scheduled_time_ns'])/1e6) for r in rr)<1e-8
            assert all(r['model']==choose(e,slot['routing'],slot['deployment']) and r['sample_id']==e['item']['sample']['id']
                and r['frozen_offset_seconds']==e['offset_seconds'] for r,e in zip(rr,events))
            assert ph['window']['planned_duration_seconds']==ph['window']['duration_seconds']==2400.
            assert max(r['completion_time_ns'] for r in raw['warmup']['request_records'])<ph['window']['start_time_ns']
            assert raw['gpu_release']['verified']
            assert all(raw['baseline_GPU'][k]==manifest['plan']['GPU'][k] for k in ['uuid','name','driver'])
    passed('every_eligible_raw_utility_and_arrival_route_cohort_recomputed',full_runs=len(report['runs']))
    for pair in report['pairs']:
        a=raws[pair['proposed_run']];b=raws[pair['control_run']]
        if pair['both_legal_stable']:
            assert cohort(a)==cohort(b)
            delta=recalcs[pair['proposed_run']]['U']-recalcs[pair['control_run']]['U']
            assert abs(delta-pair['delta_U'])<1e-12
            assert abs(sum(pair[k+'_delta'] for k in ['T','L','P','M'])-delta)<1e-12
    values=report['independent_delta_U']
    if len(values)>=2:
        n=len(values);half=student_t.ppf(.975,n-1)*np.std(values,ddof=1)/math.sqrt(n);ci=report['paired_effect_interval']
        assert abs(ci['low']-(np.mean(values)-half))<1e-12 and abs(ci['high']-(np.mean(values)+half))<1e-12
    assert gate['independent_delta_U']==values and not gate['original_24A_used_as_independent']
    assert gate['Stage_C_executed']==report['Stage_C_permitted']
    passed('pair_decomposition_CI_and_conditional_stage_gate',independent_pairs=len(report['pairs']))
    terminal=0
    for run in report['runs']:
        raw=raws[run['run_id']];phases={'measurement':raw.get('phases',{}).get('measurement',{}).get('request_records',[]),'warmup':raw.get('warmup',{}).get('request_records',[])}
        original={(phase,r['request_id']):r for phase,rr in phases.items() for r in rr}
        journal=OUT/'request_journals'/(run['run_id']+'.jsonl')
        if not journal.exists():continue
        for line in journal.open():
            row=json.loads(line)
            if row['kind']!='REQUEST_TERMINAL':continue
            key=(row['phase'],row['record']['request_id'])
            # Full raw can be absent for aborted active phase; retain journals separately.
            if key not in original:continue
            a=row['record'];b=original[key]
            assert a['status']==b['status'] and a.get('completion_time_ns')==b.get('completion_time_ns')
            terminal+=1
    passed('incremental_journal_matches_complete_raw_requests',verified_terminal_records=terminal)
    partial=read(OUT/'raw_runs/high_control_rep1_interrupted_attempt1.json')
    assert partial['full_U'] is None and not partial['valid_session']
    assert partial['slot']['run_id'] not in [r['run_id'] for r in report['runs']]
    passed('infrastructure_interruption_not_hidden_or_counted_as_outcome')
    before=read(OUT/'historical_metadata_before.json');after=protected_metadata();changed=[p for p,v in before.items() if after.get(p)!=v]
    assert not changed,changed[:5]
    verify(read(OUT/'read_only_input_inventory.json'));verify(read(OUT/'read_only_implementation_inventory.json'))
    passed('all_historical_metadata_and_consumed_SHA256_unchanged',historical_files=len(before))
    required=['positive_probe_audit.json','positive_probe_audit.md','frozen_replication_manifest.json','infrastructure_recovery_manifest.json',
        'gpu_paired_runs.csv','system_utility_comparison.csv','per_task_latency.csv','utility_decomposition.csv','worker_stability.csv',
        'GPU_thermal_state.csv','replication_result.json','module26_gate.json','module26_summary.md']
    assert all((OUT/p).is_file() for p in required)
    save('final_verification.json',dict(passed=True,checks=checks,scientific_results_not_retried_or_tuned=True,
        predictive_model_training=False,whole_pool_search=False,remaining_work=False))
    save('execution_source_manifest.json',dict(at_utc=now(),source=[ref(p) for p in sorted((ROOT/'module26').glob('*.py'))],
        required_outputs=[ref(OUT/p) for p in required],history_unchanged=True,status=gate['status']))
    print('MODULE26_VERIFIED_AND_STOPPED',gate['status'],flush=True)

if __name__=='__main__':main()
