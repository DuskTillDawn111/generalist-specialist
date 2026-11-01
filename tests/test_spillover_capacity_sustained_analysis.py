"""CPU checks for role separation, exact support exposure and frozen gates."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location('sustained_analysis',Path(__file__).resolve().parents[1]/'tools/spillover_capacity_20261022/analyze_sustained.py')
a=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(a)


def fixture(state='C',block=1,with_support=True):
    aliases={'text#0':'text','text#1':'text','generalist_2#0':'generalist_2','ocr#0':'ocr'}
    active=['generalist_2#0'] if state=='B' else ['text#0','text#1','generalist_2#0']
    pool=[f'p{i}' for i in range(32)];order=pool*3
    planned={alias:[dict(request_id=f'measurement:{alias}:{i}',role='primary',model=alias,task='text',pool_id=p,
        sequence=i,arrival_index=i,repetition=i//32) for i,p in enumerate(order)] for alias in active}
    supports={alias:[dict(request_id=f'support:{alias}:{i}',role='support',model=alias,task='text',pool_id=pool[i%32],
        support_sequence=i,sequence=96+i,arrival_index=96+i,repetition=i//32) for i in range(1024)] for alias in active}
    job=dict(job_id=f'block{block}_{state}',state=state,block=block,seed=102+block,active_aliases=active,
        input_pool_ids=order,planned_requests=planned,support_requests=supports)
    job['job_identity_sha256']=a.legacy.job_digest(job)
    packet=dict(aliases=aliases,resident_deployment={'text':2,'generalist_2':1,'ocr':1},protocol='sustained_joint_primary_support_v1')
    begin=40_000_000_000;finish=begin+2_880_000_000
    records=[];completed={};s_counts={}
    for alias in active:
        duration=30_000_000 if alias=='generalist_2#0' else 10_000_000
        templates=list(planned[alias]);s_count=192 if duration==10_000_000 and with_support else 0
        templates+=supports[alias][:s_count];s_counts[alias]=s_count;completed[alias]=begin+96*duration
        for i,p in enumerate(templates):
            start=begin+i*duration;end=start+duration
            response=dict(request_id=p['request_id'],pool_id=p['pool_id'],task_type='text',submit_ns=start,
                start_ns=start,end_ns=end,service_ms=duration/1e6,prediction=p['pool_id'],success=True,generated_tokens=2)
            records.append(dict(p,phase='measurement',status='completed',scheduled_time_ns=start,enqueued_time_ns=start,
                actual_submit_time_ns=start,actual_submit_unix_ns=start,submit_write_completed_time_ns=start,
                completion_time_ns=end,completion_unix_ns=end,worker_response=response))
    raw=dict(schema='gsdeploy.sustained_joint_capacity.raw.v1',protocol=packet['protocol'],status='completed',job=job,
        job_identity_sha256=job['job_identity_sha256'],packet_sha256='packet-sha',replica_to_family=aliases,
        plan={'deployment':packet['resident_deployment']},request_records=records,
        measurement_window=dict(start_time_ns=begin,end_time_ns=finish+2,primary_completion_time_ns=finish,
            support_admission_stop_time_ns=finish+1,alias_completion_time_ns=completed,
            primary_completed_by_alias={alias:96 for alias in active},support_released_by_alias=s_counts,
            released_by_alias={alias:96+s_counts[alias] for alias in active}),
        memory_evidence=dict(resource_valid=True),energy_evidence=dict(coverage_valid=True,sampling_errors=[]),
        shutdown={alias:dict(exit_code=0,forced=False,bye_received=True) for alias in aliases},errors=[])
    references={'text':{p:p for p in pool}}
    return packet,job,raw,references


def analyze(f):
    packet,job,raw,refs=f
    return a.analyze_job(packet,job,raw,'packet-sha',refs,
        lambda reference,response:dict(error=int(reference!=response['prediction'])),lambda task,score:(score['error'],1),
        dict(verified=True,phase_native_success_complete={'priming':True,'warmup':True}))


class SustainedAnalysis(unittest.TestCase):
    def test_support_extends_joint_activity_but_not_primary_samples(self):
        report=analyze(fixture())
        self.assertEqual(report['status'],'COMPLETE_DESCRIPTIVE_DIAGNOSTIC')
        self.assertEqual(report['primary']['service_available_requests'],288)
        self.assertEqual(report['support']['service_available_requests'],384)
        self.assertEqual(len(report['per_input']),96)
        self.assertTrue(all(r['planned_requests']==3 for r in report['per_input']))
        self.assertEqual(report['coverage']['aliases']['generalist_2#0']['positive_primary_count'],96)
        self.assertEqual(report['coverage']['aliases']['generalist_2#0']['fraction']['min'],1)
        self.assertEqual(report['measurement_timing']['primary_window_seconds'],2.88)
        self.assertAlmostEqual(report['measurement_timing']['support_drain_seconds'],2e-9)
        self.assertIsNone(report['U']);self.assertFalse(report['ranking_performed'])

    def test_early_finish_without_support_is_insufficient(self):
        report=analyze(fixture(with_support=False))
        self.assertEqual(report['status'],'INSUFFICIENT_DIAGNOSTIC_EVIDENCE')
        self.assertFalse(report['coverage']['passed'])
        self.assertEqual(report['coverage']['aliases']['generalist_2#0']['positive_primary_count'],32)
        self.assertTrue(all(r['covered_primary_service_ms']['count']==1 for r in report['per_input'] if r['alias']=='generalist_2#0'))

    def test_missing_primary_not_replaced_by_support(self):
        f=fixture();f[2]['request_records']=[r for r in f[2]['request_records'] if r['request_id']!='measurement:text#0:0']
        report=analyze(f)
        self.assertEqual(report['primary']['planned_requests'],288)
        self.assertEqual(report['primary']['service_unknown_requests'],1)
        self.assertEqual(report['support']['service_available_requests'],384)
        self.assertFalse(report['diagnostic_checks']['all_primary_and_released_support_native_scored'])
        self.assertFalse(report['coverage']['passed'])

    def test_native_failed_primary_and_failed_support_are_preserved(self):
        f=fixture();primary=f[2]['request_records'][0];primary['worker_response']['success']=False
        support=next(r for r in f[2]['request_records'] if r['role']=='support');support['status']='incomplete';support['worker_response']=None
        report=analyze(f)
        self.assertEqual(report['primary']['quality_unknown_requests'],1)
        self.assertEqual(report['support']['service_unknown_requests'],1)
        self.assertEqual(report['status'],'INSUFFICIENT_DIAGNOSTIC_EVIDENCE')

    def test_bad_template_duplicate_or_nonprefix_support_rejected(self):
        for case in ['role','duplicate','support_gap','support_template']:
            f=fixture();raw=f[2]
            if case=='role':raw['request_records'][0]['role']='support'
            elif case=='duplicate':raw['request_records'].append(copy.deepcopy(raw['request_records'][0]))
            elif case=='support_gap':raw['request_records']=[r for r in raw['request_records'] if r['request_id']!='support:text#0:0']
            else:
                f[1]['support_requests']['text#0'][0]['support_sequence']=2
                f[1]['job_identity_sha256']=a.legacy.job_digest(f[1]);raw['job_identity_sha256']=f[1]['job_identity_sha256']
            with self.subTest(case=case),self.assertRaises(ValueError):analyze(f)

    def test_admission_after_stop_and_one_inflight_break_are_flagged(self):
        f=fixture();row=next(r for r in f[2]['request_records'] if r['role']=='support');row['scheduled_time_ns']=f[2]['measurement_window']['support_admission_stop_time_ns']+1
        report=analyze(f)
        self.assertFalse(report['diagnostic_checks']['serial_and_window_contract'])
        self.assertTrue(any(r['reason']=='Support admitted after stop' for r in report['contract_flags']))

    def test_resource_energy_native_and_shutdown_gates_cannot_be_averaged(self):
        for case in ['resource','energy','shutdown','runtime']:
            f=fixture()
            if case=='resource':f[2]['memory_evidence']['resource_valid']=False
            elif case=='energy':f[2]['energy_evidence']['coverage_valid']=False
            elif case=='shutdown':f[2]['shutdown']['ocr#0']['forced']=True
            if case=='runtime':
                with self.assertRaises(ValueError):a.analyze_job(f[0],f[1],f[2],'packet-sha',f[3],None,None,dict(verified=False))
            else:
                report=analyze(f);self.assertEqual(report['status'],'INSUFFICIENT_DIAGNOSTIC_EVIDENCE')

    def test_covered_median_screen_uses_eligible_repetitions_not_fast_tail(self):
        first=analyze(fixture());second=analyze(fixture(block=2))
        self.assertTrue(a.repeatability_gate(first,second)['passed'])
        second['per_input'][0]['covered_primary_service_ms']['p50']=1000
        second['per_input'][0]['service_ms']['p50']=first['per_input'][0]['service_ms']['p50']
        result=a.repeatability_gate(first,second)
        self.assertFalse(result['passed']);self.assertGreater(next(r for r in result['per_input'] if r['alias']==second['per_input'][0]['alias'] and r['pool_id']==second['per_input'][0]['pool_id'])['second_over_first'],2)
        second['per_input'][0]['covered_primary_service_ms']['count']=1
        result=a.repeatability_gate(first,second)
        self.assertIsNone(next(r for r in result['per_input'] if r['alias']==second['per_input'][0]['alias'] and r['pool_id']==second['per_input'][0]['pool_id'])['second_over_first'])

    def test_threshold_inclusive_two_of_three_and_every_primary_positive(self):
        report=analyze(fixture());rows=report['request_diagnostics']
        target=[r for r in rows if r['role']=='primary' and r['model']=='generalist_2#0' and r['pool_id']=='p0']
        for row,value in zip(target,[.90,.90,.01]):row['activity']['all_active_aliases_busy_fraction']=value
        self.assertTrue(a.coverage_gate(rows,report['active_aliases'],'C')['passed'])
        target[-1]['activity']['all_active_aliases_busy_fraction']=0
        self.assertFalse(a.coverage_gate(rows,report['active_aliases'],'C')['passed'])
        target[-1]['activity']['all_active_aliases_busy_fraction']=.01;target[0]['activity']['all_active_aliases_busy_fraction']=.89999
        self.assertFalse(a.coverage_gate(rows,report['active_aliases'],'C')['passed'])

    def test_b_singleton_is_not_joint_evidence_and_pairs_ignore_support(self):
        baseline=analyze(fixture(state='B'));joint=analyze(fixture())
        self.assertEqual(baseline['coverage']['status'],'NOT_APPLICABLE_SINGLE_ACTIVE_ALIAS')
        pairs=a.repetition_pairs(baseline,joint)
        self.assertEqual(len(pairs),96);self.assertTrue(all(r['joint_over_baseline']==1 for r in pairs))
        self.assertTrue(all(r['prediction_same'] for r in pairs))

    def test_repeatability_requires_complete_input_roster_and_inclusive_bounds(self):
        first=analyze(fixture());second=analyze(fixture(block=2))
        for r in second['per_input']:r['covered_primary_service_ms']['p50']*=2
        self.assertTrue(a.repeatability_gate(first,second)['passed'])
        second['per_input'].pop();self.assertFalse(a.repeatability_gate(first,second)['passed'])


class TerminalReceiptProtocol(unittest.TestCase):
    def test_running_queue_is_waiting_without_reading_live_raw(self):
        import json,tempfile
        packet=dict(jobs=[dict(job_id='job1')])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'queue.json';live=Path(tmp)/'run.live.json'
            live.write_text('deliberately invalid raw')
            path.write_text(json.dumps(dict(schema='gsdeploy.sustained_capacity_queue.v1',status='RUNNING',packet_sha256='sha',
                jobs=[dict(job_id='job1',status='running',raw={'path':str(live),'sha256':'bad'})])))
            waiting,ready=a.registered_paths(packet,[path],'sha')
            self.assertFalse(ready);self.assertEqual(waiting[0]['status'],'running')

    def test_final_exact_filename_and_published_sha_are_required(self):
        import json,tempfile
        packet=dict(jobs=[dict(job_id='job1')])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'queue.json';raw=Path(tmp)/'run.live.json';raw.write_text('{}\n')
            progress=dict(schema='gsdeploy.sustained_capacity_queue.v1',status='ALL_FOUR_SUSTAINED_DIAGNOSTIC_ATTEMPTS_COMPLETE',
                packet_sha256='sha',jobs=[dict(job_id='job1',status='completed',raw={'path':str(raw),'sha256':'bad'})])
            path.write_text(json.dumps(progress));waiting,ready=a.registered_paths(packet,[path],'sha')
            self.assertFalse(ready);self.assertEqual(waiting[0]['status'],'PUBLISHED_FINAL_RECEIPT_MISSING')
            final=Path(tmp)/'run.json';raw.rename(final);progress['jobs'][0]['raw']['path']=str(final);path.write_text(json.dumps(progress))
            waiting,ready=a.registered_paths(packet,[path],'sha');self.assertFalse(waiting);self.assertEqual(len(ready),1)
            with self.assertRaises(ValueError):a.legacy.read_final(final,'bad')

    def test_wrong_queue_or_conflicting_registration_rejected(self):
        import json,tempfile
        packet=dict(jobs=[dict(job_id='job1')])
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'q1.json';second=Path(tmp)/'q2.json'
            progress=dict(schema='gsdeploy.sustained_capacity_queue.v1',status='RUNNING',packet_sha256='sha',jobs=[dict(job_id='job1',status='running')])
            first.write_text(json.dumps(progress));progress['jobs'][0]['status']='failed';second.write_text(json.dumps(progress))
            with self.assertRaises(ValueError):a.registered_paths(packet,[first,second],'sha')
            with self.assertRaises(ValueError):a.registered_paths(packet,[first],'wrong')


if __name__=='__main__':unittest.main()
