"""Read existing evidence into a dated inventory; no fitting or GPU operation."""
from collections import Counter,defaultdict
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/predictor-gpu-20261027'


def build():
    refs={}
    def ref(path):
        p=Path(path).resolve();value=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        refs[str(p)]=value;return value
    def load(path):
        initial=ref(path);value=json.loads(Path(path).read_text())
        if initial!=ref(path):raise ValueError('Evidence changed during inventory read')
        return value
    micro=[]
    for folder in ['native-training-v3','semantic-training-v5','background-training-v7']:
        plan=load(BASE/folder/'plan.json');raw=load(BASE/folder/'raw.json');jobs={j['id']:j for j in plan['jobs']}
        executed=[jobs[b['job_id']] for b in raw['batches']];inputs={p for j in executed for p in j['inputs'].values()}
        measured=defaultdict(list)
        for row in plan['inputs']:
            if row['pool_id']in inputs:measured[row['task']].append(row['input_size'])
        rows=raw['request_records'];samples=raw['board_samples'];lengths=defaultdict(set)
        events=[e for r in rows for e in r.get('execution_events',[])]
        for row in rows:
            for event in row.get('execution_events',[]):
                if event.get('semantic_phase')=='prefill' and event.get('event')=='start':
                    lengths[row['model']].add(event['input_sequence_length'])
        micro.append(dict(name=folder,plan=ref(BASE/folder/'plan.json'),native_raw=ref(BASE/folder/'raw.json'),
            status=raw['status'],batch_count=len(executed),request_count=len(rows),completed=sum(r['status']=='completed'for r in rows),
            context_counts=dict(sorted(Counter(j['context']for j in executed).items())),executed_split_counts=dict(Counter(j['split']for j in executed)),
            resident_aliases=sorted(raw['ready']),measured_input_size_values={k:sorted(v)for k,v in measured.items()},
            input_size_scope='Declared public input_size; not token length for image/audio/OCR',
            observed_prefill_sequence_lengths={k:sorted(v)for k,v in lengths.items()},
            scheduled_offsets_ms=sorted({v for j in executed for v in j['start_offsets_ms'].values()}),
            SSG_offset_patterns_by_sorted_aliases=sorted({tuple(j['start_offsets_ms'][a]for a in sorted(j['start_offsets_ms']))for j in executed if j['context']=='SSG'}),
            actual_worker_start_rows=sum('observed_worker_start_time_ns'in r for r in rows),
            phase_event_rows=sum(bool(r.get('execution_events'))for r in rows),
            semantic_prefill_events=sum(e.get('semantic_phase')=='prefill'and e.get('event')=='start'for e in events),
            semantic_decode_events=sum(e.get('semantic_phase')=='decode'and e.get('event')=='start'for e in events),
            admission_event_rows=sum(any('admission'in k for k in r)for r in rows),samples=len(samples),
            owned_sampled_peak_mib=max(s['owned_used_mib']for s in samples),
            board_watts_range=[min(s['watts']for s in samples),max(s['watts']for s in samples)],
            max_sample_gap_ms=max(b['time_ns']-a['time_ns']for a,b in zip(samples,samples[1:]))/1e6,
            foreign_pid_samples=sum(bool(s['foreign_pids'])for s in samples),
            all_resident_owned_samples=sum(set(s['owned_aliases'])==set(raw['ready'])for s in samples),
            errors=raw['errors'],telemetry_errors=raw['telemetry_errors']))
    context=load(BASE/'background-training-v7/context.json');probe=load(BASE/'hardware-probe/receipt.json')
    counter_path=BASE/'hardware-probe/tiny-counters.csv';ref(counter_path)
    kernel=load(BASE/'semantic-trace-v5/exports/summary.json')
    report=load(BASE/'flow-validation-h600-selected-v16/report/report.json');gate=load(BASE/'flow-validation-h600-selected-v16/gate.json')
    capacity=load(BASE/'flow-validation-h600-selected-v16/capacity-report.json');dfs=load(BASE/'cpu-search/h600-selected-v16/summary.json')
    by_id={c['candidate_id']:c for c in capacity['candidates']};validation=[]
    for row in report['candidates']:
        q=by_id[row['candidate_id']];s=q['original_stability']
        validation.append(dict(candidate_id=row['candidate_id'],requested=row['requested'],known=row['known'],
            actual_six_terms_complete=row['observed_metrics']['six_term_comparison_complete'],observed_U=row['observed_metrics']['full_U'],
            service_mare=row['timing_errors']['service']['mean_absolute_relative_error'],e2e_mare=row['timing_errors']['e2e']['mean_absolute_relative_error'],
            queue_mae=row['queue_error']['mae'],original_stability_passed=s['original_stability_passed'],
            stability_state=s['stability']['stability_state_v2'],stability_reason=s['stability']['stability_reason_v2'],
            original_rho_threshold_satisfied=q['original_rho_threshold_satisfied'],statistical_profile_applicable=q['statistical_profile_applicable'],
            statistical_planning_admission_passed=q['statistical_original_rho_planning_admission_passed'],
            original_measured_resolver_admission=q['original_measured_profile_resolver_admission']))
    process=[]
    for pid in [85446,87518,87937]:
        p=Path('/proc')/str(pid)
        if not p.exists():continue
        stat=(p/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        process.append(dict(pid=pid,state=fields[0],start_ticks=int(fields[19]),
            argv=[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x],cwd=str((p/'cwd').resolve())))
    absent={f:not(BASE/f).exists()for f in ['h600-admission-pilot-v1/completed.json','admission-training-continuation-v1/completed.json',
        'admission-validation-continuation-v1/completed.json','four-group-continuation-v1/completed.json']}
    for path in [__file__,*[ROOT/p for p in ['gsdeploy/resources.py','gsdeploy/forecasting/simulator.py',
            'tools/predictor_validation_20261027/h600_four_group_diagnostic_v1.py',
            'tools/predictor_validation_20261027/h600_four_group_statistics_v1.py',
            'tools/predictor_validation_20261027/h600_four_group_run_v1.py',
            'docs/usher_methods_and_project_adaptation_20261024.md','docs/roomie_partition_spillover_design_20261024.md',
            'docs/h600_execution_mechanism_readiness_20261027.md']],*[BASE/p for p in ['admission-next-stage-roles-v1.json',
            'admission-training-v1/campaign.json','admission-validation-v1/protocol.json','admission-training-continuation-v1/protocol.json',
            'admission-validation-continuation-v1/protocol.json','four-group-continuation-v1/protocol.json','four-group-diagnostic-v1/plan.json']]]:
        ref(path)
    value=dict(schema='gsdeploy.user_request_evidence_inventory.v1',snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Read-only inventory against four-stage request; no new GPU, fitting, selection, threshold, or retry',microprofiles=micro,
        hardware=dict(nsys_returncode=probe['commands'][0]['returncode'],ncu_returncode=probe['commands'][1]['returncode'],
            counter_permission_denied='ERR_NVGPUCTRPERM'in counter_path.read_text(),context_counters=context['counters'],execution=context['execution'],
            compute_memory_complementarity_physically_validated=False,memory_capacity_is_not_DRAM_bandwidth=True),
        kernel_preflight=dict(request_count=len(kernel['requests']),all_anchors_matched=kernel['all_anchors_matched'],resource_demand=kernel['resource_demand'],
            maximum_observed_semantic_kernel_tail_ms=max(s['max_kernel_tail_past_call_return_ms']for r in kernel['requests']for s in r['semantic_rows']),scope=kernel['claim']),
        independent_shared=dict(status=gate['status'],failed_checks=gate['failed_checks'],unknown_checks=gate['unknown_checks'],
            delta_sign_accuracy=report['delta_sign_accuracy'],rank_accuracy=report['rank_accuracy'],candidate_coverage=report['candidate_coverage'],candidates=validation,
            known_input_single_block=True,unseen_input_generalization_validated=False,final_gain_eligible=False),
        mechanisms=[dict(name=n,status=s,evidence=e)for n,s,e in [
            ('Roomie-inspired phase/context sequence prediction','IMPLEMENTED_EMPIRICAL_NOT_PHYSICAL_RESOURCE_MODEL','Real START/PHASE/prefill/decode and launch-attributed kernel preflight; no validated occupancy/SM/DRAM/L2 model'),
            ('compute/memory complementary placement','UNKNOWN_PHYSICAL_CLASSIFICATION','Counters ERR_NVGPUCTRPERM; fixed residency; no complementary placement intervention/attribution'),
            ('whole-request admission','IMPLEMENTED_PENDING_INDEPENDENT_GPU_DIAGNOSTIC','Host permit and controller audit exist; shared/max1/max2 pilot and T8/V4 pending in snapshot; phase1 shared raws have no permit-wait fields'),
            ('MPS active-thread cap','CONTRACT_ONLY_NATIVE_LAUNCH_BLOCKED','Unverified controlled platform adapter blocked by worker_environment; no daemon/client/cap-effect receipt'),
            ('exclusive SM/green contexts','ADAPTER_NOT_IMPLEMENTED_HARDWARE_SUPPORT_UNKNOWN','Declaration validation only; no resource-context binding or disjointness proof'),
            ('joint or split batch across requests','EXECUTOR_NOT_IMPLEMENTED','Fixed batch1; simulator raises JOINT_BATCH_OR_MERGED_WEIGHT_EXECUTOR_REQUIRED; internal microbatch is different'),
            ('same-model shared weights','POTENTIAL_EXACT_IDENTITY_NOT_EXECUTED','text aliases share text spec but independent PIDs/model owners; no shared tensor owner/co-batched kernel receipt'),
            ('heterogeneous maximum common weight submatrix/cache reuse','SHARED_BLOCKS_UNKNOWN_EXECUTOR_NOT_IMPLEMENTED','No exact block extraction/merged operator/cache-hit experiment; L2 counters unavailable; no reuse discount')]],
        same_domain_DFS=dict(status=dfs['status'],candidate_count=dfs['candidate_count'],requests_per_candidate=dfs['request_count'],
            same_optimum_and_all_ties=dfs['same_optimum_and_all_ties'],all_ties=dfs['all_tie_ids'],pruning_count=dfs['pruning_count'],
            forecast_calls_saved=dfs['forecast_calls_saved'],physical_optimality_certified=False,original_gate_constrained_optimality_certified=False,
            admission_same_domain_DFS_completed=False,unsearched=dfs['unsearched']),
        pending_finite_stages=dict(process_snapshot=process,completion_file_absent=absent,
            T8=dict(seeds=[628101,628102],runs=8,requests=9600),V4=dict(seed=628201,runs=4,requests=4800),F8=dict(seed=628301,runs=8,requests=9600,independent_blocks=1),
            final_qualification='Descriptive diagnostic only; no qualified positive final gain',
            strong_baseline='Locked spill_high minus strongest observed balanced/skewed in admission2; not globally optimized no-spill deployment',
            missing=['Actual admission/permit-wait evidence','All8 T8 completions and separate Q/E/M/capacity fit',
                'V4 independent slowdown/full queue/P95/DeltaU/rank/native stability','F8 all actual six terms and within-admission strong baseline contrast',
                'Admission same-domain fresh DFS if claiming optimization','Complementarity/batch/submatrix/cache/MPS physical evidence absent; finite T/V/F does not implement them']),
        limitations=['Sampled owned peak is not continuous peak or bound','Whole-board power is not per-kernel power',
            'Python semantic call return is not GPU completion','One input seed does not prove unseen-input generalization',
            'Requests within one block are not independent replicates','Collection completion is not predictor PASS or deployable gain'],
        source_receipts=list(refs.values()),GPU_started=False,new_experiment_launched=False)
    return value


if __name__=='__main__':
    value=build();path=BASE/'request-evidence-inventory-v1.json'
    with path.open('x')as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(path=str(path),sources=len(value['source_receipts']),snapshot_utc=value['snapshot_utc'],GPU_started=False)))
