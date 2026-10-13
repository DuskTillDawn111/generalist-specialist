"""Resume remaining frozen slots after a CPU-only reporting exception; no GPU retry."""
from run_followup import *
from report_followup_repaired import write_reports

def main():
    f=json.loads((OUT/'frozen_candidate_manifest.json').read_text())
    anchor=json.loads((OUT/'gpu_run_manifest.json').read_text())
    for r in f['implementation']+f['reused_implementation']+[f['catalog'],f['utility_definition'],f['utility_reference'],f['quality_policy'],anchor['workloads'],anchor['protocol']]:
        assert ref(r['path'])['sha256']==r['sha256'],r['path']
    ledger=json.loads((OUT/'attempt_ledger.json').read_text())
    assert len(ledger)==1 and ledger[0]['status']=='COMPLETED'
    raws=[json.loads((OUT/'raw_runs'/(row['run_id']+'.json')).read_text()) for row in ledger]
    assert all(raw['valid_session'] and raw['gpu_release']['verified'] for raw in raws)
    save('report_repair_amendment.json',dict(at_unix_ns=time.time_ns(),scope='CPU_REPORT_FIELD_NAME_ONLY',
        reason='Repeated status keyword in per-request CSV dict caused post-run TypeError',
        change='request status renamed request_status, run status remains status; no numerical formulas changed',
        original_report=ref(OUT/'report_followup.py'),repaired_report=ref(OUT/'report_followup_repaired.py'),
        original_first_raw=ref(OUT/'raw_runs/R1_HIGH_s_only.json'),freeze_files_changed=False,
        GPU_outcome_retries=0,new_workload_or_router_settings=False,remaining_frozen_slots=[s['run_id'] for s in f['exploration_slots'][1:]],
        original_failure_log_preserved=True))
    write_reports(raws,f,final=False)
    original_gpu.DurableMonitor=SafeMonitor
    fatal=None
    for slot in f['exploration_slots'][1:]:
        assert len(ledger)<6 and not (OUT/'raw_runs'/(slot['run_id']+'.json')).exists()
        ledger.append(dict(run_id=slot['run_id'],scenario=slot['scenario'],arm=slot['arm'],order=slot['run_order'],status='STARTING',start_unix_ns=time.time_ns()))
        save('attempt_ledger.json',ledger);print('START',slot['run_id'],flush=True)
        try:
            raw=original_gpu.run(slot,f);raws.append(raw);v=original_report.evaluate(raw,f)
            ledger[-1].update(status=raw['status'],end_unix_ns=time.time_ns(),eligible=v['eligible'],U=v['U'],worker_states=v['worker_states']);save('attempt_ledger.json',ledger)
            print('FINISH',slot['run_id'],'eligible',v['eligible'],'U',v['U'],'worker_states',v['worker_states'],'errors',raw['errors'],flush=True)
            safe=(raw['valid_session'] and v['full_cohort'] and v['capability'] and v['measured_counts'] and v['semantic_valid'] and v['memory_ok'] and v['recovery_confirmed'] and not raw['errors'])
            if not safe:fatal=dict(run_id=slot['run_id'],reason='INFRASTRUCTURE_SAFETY_OR_ORIGINAL_HARD_CONSTRAINT',errors=raw['errors'],evaluation=v)
        except (Exception,KeyboardInterrupt) as exc:
            fatal=dict(run_id=slot['run_id'],reason=repr(exc),traceback=traceback.format_exc())
            ledger[-1].update(status='EXCEPTION',error=fatal,end_unix_ns=time.time_ns());save('attempt_ledger.json',ledger);print('EXCEPTION',fatal,flush=True)
        write_reports(raws,f,fatal,final=False)
        if fatal:break
    gate=write_reports(raws,f,fatal,final=True)
    save('campaign_completed.json',dict(at_unix_ns=time.time_ns(),completed_and_stopped=True,
        executed=[r['slot']['run_id'] for r in raws],attempts=len(ledger),formal_seconds=sum(600 for r in raws if r.get('phases',{}).get('measurement')),
        fatal=fatal,gates=gate,no_automatic_next_module=True,no_independent_long_window=True,
        CPU_report_repair=ref(OUT/'report_repair_amendment.json')))
    print('FINAL_GATES',gate,flush=True)

if __name__=='__main__':main()
