"""Independent audit rejects identity, timing, isolation and power tampering."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.queue_coupled_utility_20261023 import experiment_audit as a


class ControlledAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name); asset = root/'asset'; asset.write_text('cpu')
        pool = [dict(pool_id=f't{i:02}', task_type='text', sample=dict(context='c', question='q')) for i in range(32)]
        pool += [dict(pool_id=f'{task}{i}', task_type=task, sample=dict(input=str(asset)))
                 for task in ('image', 'asr', 'ocr') for i in range(3)]
        samples = root/'pool.jsonl'; samples.write_text(''.join(json.dumps(row)+'\n' for row in pool))
        models = {'text': dict(supported_tasks=['text']), 'generalist_2': dict(supported_tasks=['text','image','asr','ocr']),
                  'ocr': dict(supported_tasks=['ocr'])}
        aliases = {f'{family}#{i}': dict(family=family, supported_tasks=models[family]['supported_tasks'])
                   for family, count in a.PRIMARY.items() for i in range(count)}
        base = root/'base.json'; base.write_text(json.dumps(dict(config=dict(workload={t: .01 for t in ('text','image','asr','ocr')},
            catalog=dict(models=models, safe_factor=1.1), gpu_memory_mib=1000.))))
        source = root/'source.py'; source.write_text('immutable = True\n')
        schedule = a._schedule([r['pool_id'] for r in pool if r['task_type']=='text'])
        packet = dict(schema='gsdeploy.coupled_controlled_calibration_packet.v1', deployment=a.PRIMARY,
            sources=[a.sha(source)], bindings=[a.sha(base),a.sha(samples)], base_packet=a.sha(base), input_pool=a.sha(samples),
            policy_contract_for_preflight_only=dict(aliases=aliases), batches=schedule, batch_schedule_sha256=a.canonical(schedule),
            seeds=[301,302], planned_batches=320, planned_text_requests=576, no_retries=True)
        requests=[]; batches=[]; offset=10**15; clock=1_000_000_000
        for case in schedule:
            ids=[]
            for alias in case['aliases']:
                i=len(requests); rid=f'controlled:{i}'; ids.append(rid)
                begin, end, completion = clock+1_000_000, clock+2_000_000, clock+2_200_000
                requests.append(dict(request_id=rid, model=alias, phase='controlled', task='text', pool_id=case['pool_id'],
                    arrival_index=i, status='completed', scheduled_time_ns=clock, enqueued_time_ns=clock+100_000,
                    actual_submit_time_ns=clock+200_000, completion_time_ns=completion, completion_unix_ns=completion+offset,
                    calibration_seed=case['seed'], calibration_context=case['context'], calibration_batch_id=case['id'],
                    worker_response=dict(request_id=rid, pool_id=case['pool_id'], task_type='text', arrival_index=i,
                                         success=True, service_ms=1., start_ns=begin+offset-200_000, end_ns=end+offset),
                    actual_service_interval=dict(start_time_ns=begin,end_time_ns=end,start_unix_ns=begin+offset,end_unix_ns=end+offset)))
            batches.append(dict(case, request_ids=ids,start_time_ns=clock,end_time_ns=clock+10_000_000,all_completed=True))
            clock += 10_000_000
        priming=[]
        for alias,spec in aliases.items():
            for task in spec['supported_tasks']:
                ids=sorted(row['pool_id'] for row in pool if row['task_type']==task)[:3]
                priming += [dict(model=alias,task=task,pool_id=pid,status='completed',completion_time_ns=500_000_000) for pid in ids]
        raw=dict(schema='gsdeploy.coupled_controlled_calibration.v1',source_packet_digest=a.canonical(packet),
            source_receipts=packet['sources'],input_pool=packet['input_pool'],no_retries=True,backend='native',status='completed',
            errors=[],calibration_usable=True,U=None,original_stability_evaluated=False,seeds=[301,302],
            calibration_phase_seeds_are_arrival_seeds=False,deployment=a.PRIMARY,policy_contract=packet['policy_contract_for_preflight_only'],
            aliases=aliases,replica_to_family={alias:spec['family'] for alias,spec in aliases.items()},
            worker_registry={alias:dict(env={key:'1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}) for alias in aliases},
            preflight=dict(ready=True),shutdown={alias:dict(forced=False,exit_code=0,bye_received=True) for alias in aliases},
            batches=batches,request_records=requests,priming_request_records=priming,
            memory_evidence=dict(measured=True,resource_valid=True,safe_memory_mib=110.,samples=[dict(owned_used_mib=100.,foreign_pids=[])]),
            measurement_window=dict(start_time_ns=1_000_000_000,end_time_ns=clock,duration_seconds=(clock-1_000_000_000)/1e9),
            energy_evidence=dict(kind='measured',scope='whole_gpu_board',coverage_valid=True,sampling_errors=[],
                run_joules=100.*(clock-1_000_000_000)/1e9,samples=[dict(time_ns=i*100_000_000,watts=100.) for i in range(61)]))
        self.packet=packet;self.raw=raw;self.packet_path=root/'packet.json';self.raw_path=root/'raw.json';self.source=source

    def run_audit(self,raw=None,packet=None):
        self.packet_path.write_text(json.dumps(packet or self.packet))
        self.raw_path.write_text(json.dumps(raw or self.raw))
        return a.audit_controlled(self.packet_path,self.raw_path)

    def test_complete_synthetic_evidence_passes_all_protocol_checks(self):
        result=self.run_audit()
        self.assertTrue(result['verified'],result['errors'])
        self.assertGreater(len(result['checks']),5000)
        self.assertEqual(result['diagnostics']['request_count'],576)
        self.assertFalse(result['answers_or_quality_references_extracted'])

    def test_source_changed_and_packet_digest_changed_are_rejected(self):
        self.source.write_text('immutable = False\n')
        self.assertIn('ALL_SOURCE_INPUT_BINDING_HASHES',self.run_audit()['errors'])
        packet=copy.deepcopy(self.packet);packet['planned_text_requests']=575
        self.assertIn('SOURCE_PACKET_DIGEST',self.run_audit(packet=packet)['errors'])

    def test_schedule_identity_and_batch_drain_overlap_are_rejected(self):
        raw=copy.deepcopy(self.raw);raw['batches'][1]['seed']=302
        self.assertIn('BATCH_IDENTITY_1',self.run_audit(raw)['errors'])
        raw=copy.deepcopy(self.raw);raw['batches'][1]['start_time_ns']=raw['batches'][0]['end_time_ns']-1
        self.assertIn('BATCH_DRAIN_ISOLATION_1',self.run_audit(raw)['errors'])

    def test_worker_interval_clock_mapping_and_response_identity_are_rejected(self):
        raw=copy.deepcopy(self.raw);raw['request_records'][0]['actual_service_interval']['start_time_ns']+=1
        self.assertIn('EXACT_WORKER_SERVICE_CLOCK_CONVERSION_controlled:0',self.run_audit(raw)['errors'])
        raw=copy.deepcopy(self.raw);raw['request_records'][0]['worker_response']['pool_id']='wrong'
        self.assertIn('WORKER_RESPONSE_IDENTITY_controlled:0',self.run_audit(raw)['errors'])

    def test_power_gap_never_imputed_and_formal_utility_claim_rejected(self):
        raw=copy.deepcopy(self.raw)
        raw['energy_evidence']['samples']=[dict(time_ns=0,watts=100.),dict(time_ns=6_000_000_000,watts=100.)]
        self.assertIn('ORIGINAL_2_SECOND_GAP_ENERGY_INTEGRAL',self.run_audit(raw)['errors'])
        raw=copy.deepcopy(self.raw);raw['U']=0.
        self.assertIn('NO_FORMAL_UTILITY_OR_STABILITY_CLAIM',self.run_audit(raw)['errors'])

    def test_opaque_worker_answers_do_not_affect_measurement_audit(self):
        raw=copy.deepcopy(self.raw)
        for row in raw['request_records']:
            row['worker_response']['answers']=['PRIVATE']
            row['worker_response']['raw_output']='PRIVATE'
        result=self.run_audit(raw)
        self.assertTrue(result['verified'],result['errors'])
        self.assertNotIn('PRIVATE',json.dumps(result))


if __name__=='__main__':unittest.main()
