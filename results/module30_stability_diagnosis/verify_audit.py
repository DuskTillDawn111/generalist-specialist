"""Independent CPU verification of final Module30 numerical artifacts."""
import sys
sys.dont_write_bytecode=True
import csv,json,math,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from reproduce_audit import verify_inputs
P=Path(__file__).resolve().parent
def load(n):return json.loads((P/n).read_text())
def save(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    required=['summary.md','window_reconciliation.json','stability_bins.csv','stability_reproduction.json',
        'diversion_timeline.csv','dynamic_guard_breakdown.json','competing_explanations.md','quality_interference_caveats.md',
        'next_experiment_preregistration_draft.md','input_integrity_before_after.json','module30_gate.json']
    assert all((P/n).is_file() for n in required)
    checks=[dict(name='required_artifacts_exist',passed=True)]
    bins=list(csv.DictReader((P/'stability_bins.csv').open()));norm=list(csv.DictReader((P/'normalized_events_and_latencies.csv').open()))
    st=load('stability_reproduction.json')
    for run in ['S_only','Dynamic']:
        b=sorted([r for r in bins if r['run']==run and r['scope']=='ALL'],key=lambda x:int(x['left_s']))
        rr=[r for r in norm if r['run']==run]
        assert len(b)==20 and sum(int(x['sent']) for x in b)==7607
        assert sum(int(x['successful_responses']) for x in b)==(7575 if run=='S_only' else 7588)
        # Little's-law area identity, independent of original event integration.
        area=sum(max(0.,min(float(r['completion_s']),600)-float(r['submitted_s'])) for r in rr)
        integrated=sum(float(x['time_mean_submitted_pending'])*30 for x in b)
        assert abs(area-integrated)<1e-7
        tail=b[-5:];xs=[float(x['right_s']) for x in tail];ys=[float(x['time_mean_submitted_pending']) for x in tail]
        mx=sum(xs)/5;my=sum(ys)/5
        slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
        assert abs(slope-st['runs'][run]['formal_tail_only']['slope'])<1e-12
        assert all(st['runs'][run]['conjunctive_tail_failure_conditions'].values())
        assert st['runs'][run]['result']['stability_state_v2']=='unstable'
        assert [int(x['pending_at_right']) for x in tail][-2]==0
        residual=[float(r['tail_OLS_residual_requests']) for r in tail]
        assert abs(sum(residual))<1e-10
        assert abs(sum((x-mx)*r for x,r in zip(xs,residual)))<1e-8
        checks.append(dict(name=run+'_formal_area_OLS_residuals_and_original_tail_failure',passed=True,area_request_seconds=area,
            slope_rps=slope,formal_end_pending=int(b[-1]['pending_at_right'])))
    tail=list(csv.DictReader((P/'tail_growth_decomposition.csv').open()))
    for run in ['S_only','Dynamic']:
        all_slope=st['runs'][run]['formal_tail_only']['slope']
        for scope in ['WORKER','TASK','WORKER_TASK']:
            assert abs(sum(float(r['slope_contribution_rps']) for r in tail if r['run']==run and r['scope']==scope)-all_slope)<1e-10
    checks.append(dict(name='additive_worker_task_slope_partitions',passed=True))
    dvs=list(csv.DictReader((P/'diversion_timeline.csv').open()));assert len(dvs)==67 and len({r['request_id'] for r in dvs})==67
    original=Path('/root/autodl-tmp/generalist-specialist/results/module29_offline_opportunity/stage2_20261009T153923Z')
    trace={r['request_id']:r for r in csv.DictReader((original/'routing_decision_trace.csv').open()) if r['phase']=='measurement'}
    assert len(trace)==7005
    for r in dvs:
        q=trace[r['request_id']];assert q['selected_worker']=='qwen#0' and q['reason']=='G_LOWER_SYSTEM_MARGINAL_COST'
        assert float(q['decision_t_s'])==float(r['decision_s']) and float(q['gap_ms'])==float(r['gap_ms'])
        assert q['guards']==r['guards'] or json.loads(q['guards'])==json.loads(r['guards'])
        assert float(r['G_completed_s'])>=float(r['decision_s']) and float(r['G_service_ms'])>0
    actual=[r for r in trace.values() if r['selected_worker']=='qwen#0'];assert len(actual)==67
    guard=load('dynamic_guard_breakdown.json');assert guard['denominator']==7005 and sum(guard['actual_primary_reasons'].values())==7005
    assert guard['cost_eligible_by_actual_hysteresis']==814 and sum(guard['primary_reasons_given_cost_eligible'].values())==814
    assert guard['nonexclusive_guard_failures_given_cost_eligible']['token_backpressure']==747
    for g in guard['guards'].values():assert g['PASS']+g['FAIL']+g['UNKNOWN']==7005
    checks.append(dict(name='67_actual_diversions_match_original_trace_and_priority_denominators',passed=True,actions=67))
    q=load('G_quality_cluster_sensitivity.json');assert q['request_correct']==61 and q['request_n']==67 and q['unique_source_samples']==27
    quality=list(csv.DictReader((P/'quality_by_task_worker.csv').open()))
    row=next(r for r in quality if r['run']=='Dynamic' and r['worker']=='qwen#0' and r['task']=='lid')
    assert row['closed_set_flag_n']=='0' and row['closed_set_validity']=='UNKNOWN'
    overlap=list(csv.DictReader((P/'execution_overlap_pairs.csv').open()));oldoverlap=list(csv.DictReader((original/'S_G_execution_overlaps.csv').open()))
    assert len(overlap)==len(oldoverlap)==403
    newset={(r['S_request_id'],r['G_request_id']) for r in overlap}
    oldset={(r['S_request_id'],r['G_target_request_id']) for r in oldoverlap};assert newset==oldset
    checks.append(dict(name='quality_missing_flags_cluster_caveat_and_403_overlap_pairs',passed=True))
    for name in ['pending_trajectories.svg','diversion_timeline.svg']:ET.parse(P/name)
    checks.append(dict(name='standalone_SVG_graphs_well_formed',passed=True))
    assert load('module30_gate.json')['recommendation_gate']=='GO'
    assert not load('module30_gate.json')['GPU_execution_authorized_by_this_gate']
    assert load('module30_gate.json')['formal_utility_comparison_at_original_load']=='NO_GO'
    n=verify_inputs(True);checks.append(dict(name='all_input_sets_sizes_SHA256_unchanged',passed=True,files=n))
    save('final_verification.json',dict(passed=True,checks=checks,original_Module28_Module29_modified=False,
        GPU_inferences_started=0,optimizer_modified=False,legal_U=None,delta_U=None))
    files=sorted(str(x.resolve()) for x in P.iterdir() if x.is_file())
    save('artifact_paths.json',dict(output_directory=str(P),created_file_paths=sorted(set(files+[str(P/'artifact_paths.json')]))))
    print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
