"""CPU receipt fixtures verify reference derivation and rejection paths."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy import normalization as norm
from gsdeploy.config import SystemConfig
from gsdeploy.energy import energy_key
from gsdeploy.reference_freeze import freeze_references
from gsdeploy.trace import make_trace, trace_digest

ROOT=Path(__file__).resolve().parents[1]


class ReferenceFreezeTests(unittest.TestCase):
    def setup_receipts(self, base):
        d=SystemConfig.load(ROOT/'configs/paper_fifo_reference_v2_cpu_fixture.json').as_dict()
        d['catalog'].update(performance_mode='independent',memory_mode='standalone_proxy',
                            measurement_context={'gpu_uuid':'CPU fabricated GPU identity'})
        s=d['catalog'][norm.KEY]
        s.update(status='pending',measurement_context=d['catalog']['measurement_context'],
                 input_identity=d['catalog']['quality']['identity'],deadline_multiplier=2,
                 reference_deployment={'s':1})
        s['tasks']['text'].update(reference_path='s',provenance={},deadline_ms=None)
        s['tasks']['text']['provenance']['quality']='CPU fabricated development anchors'
        d['latency_reference_ms']['text']=None
        c=SystemConfig(**d)
        for name,seed,watts in [('task',100,10),('ep1',101,10),('ep2',102,20)]:
            trace=make_trace(c.workload,1.,seed,c.arrival_process)
            records=[]
            for i,(offset,t) in enumerate(trace):
                at=1_000_000_000+round(offset*1e9)
                records.append(dict(request_id=f'measurement:{i}',task=t,model='s#0',
                    status='completed',scheduled_time_ns=at,actual_submit_time_ns=at,
                    completion_time_ns=at+2_000_000))
            raw=dict(backend='native',status='completed',errors=[],plan=dict(deployment={'s':1},routing={'text':{'s':2.}}),
                measurement_window=dict(duration_seconds=1.,start_time_ns=1_000_000_000,end_time_ns=2_000_000_000,
                    drain_end_time_ns=2_000_000_000,offered_requests=len(trace),trace_sha256=trace_digest(trace)),
                transport_config=dict(random_seed=seed,arrival_process=c.arrival_process,workload_rps=c.workload,
                    warmup_seconds=.01,drain_timeout_seconds=2.),request_records=records,
                priming_policy='one_per_routed_task',memory_evidence=dict(measured=True,resource_valid=True),
                preflight=dict(ready=True,gpu_checked=True,gpu={'uuid':s['measurement_context']['gpu_uuid']},
                    sample_pool_sha256=s['input_identity']['input_pool_sha256'],admission_evidence={'verified':True}),
                energy_evidence=dict(kind='measured',scope='whole_gpu_board',startup_begin_time_ns=0,
                    samples=[dict(time_ns=i*1_000_000_000,watts=watts) for i in range(3)],
                    profile_key=energy_key({'s':1},{'text':{'s':2.}},c.workload,1.,seed,c.arrival_process,catalog=c.catalog)))
            (base/(name+'.json')).write_text(json.dumps(raw))
        manifest=dict(schema='gsdeploy.normalization_measurements.v1',split='normalization_development',
                      confirmation_seeds=[999],task_runs={'text':['task.json']},episode_runs=['ep1.json','ep2.json'])
        return c,manifest

    def test_reference_values_are_derived_from_raw_intervals_and_repeats(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); c,m=self.setup_receipts(base)
            # This test only checks denominator derivation; native stability/identity
            # assessment is exercised separately by evaluation tests.
            evaluated=dict(stability={'stability_state_v2':'steady'},
                           utility=dict(constraints_verified=True,quality_identity_verified=True))
            with patch('gsdeploy.evaluation.evaluate_run',return_value=evaluated):
                s=freeze_references(c,m,base)
            self.assertEqual(s['status'],'frozen')
            self.assertEqual(s['tasks']['text']['deadline_ms'],4.)
            self.assertEqual(s['episode_reference_joules'],30.)
            self.assertEqual(len(s['reference_receipts']),4)
            self.assertTrue(all(len(r['sha256'])==64 for r in s['reference_receipts'][1:]))
            changed={**c.catalog,norm.KEY:s,'measurement_context':{'gpu_uuid':'another GPU'}}
            with self.assertRaisesRegex(ValueError,'GPU/checkpoints'):
                norm.validate(changed,c.workload,{'text':4.},1.,100.)

    def test_confirmation_leakage_and_uncovered_drain_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);c,m=self.setup_receipts(base)
            m['confirmation_seeds']=[101]
            with self.assertRaisesRegex(ValueError,'confirmation'):
                freeze_references(c,m,base)
            m['confirmation_seeds']=[999]
            path=base/'ep1.json';raw=json.loads(path.read_text())
            raw['energy_evidence']['samples'].pop()
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'telemetry'):
                freeze_references(c,m,base)

    def test_mock_receipts_do_not_freeze_physical_references(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);c,m=self.setup_receipts(base)
            p=base/'task.json';raw=json.loads(p.read_text());raw['backend']='mock';p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'native'):
                freeze_references(c,m,base)

    def test_same_model_policy_accepts_different_uuids_and_retains_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);c,m=self.setup_receipts(base)
            m['gpu_compatibility']={'policy':'same_model','name':'CPU fixture model'}
            for i,name in enumerate(('task','ep1','ep2')):
                p=base/(name+'.json');raw=json.loads(p.read_text())
                raw['preflight']['gpu']={'uuid':f'fixture-device-{i}',
                    'name':'CPU fixture model','memory_mib':100.}
                p.write_text(json.dumps(raw))
            evaluated=dict(stability={'stability_state_v2':'steady'},
                           utility=dict(constraints_verified=True,quality_identity_verified=True))
            with patch('gsdeploy.evaluation.evaluate_run',return_value=evaluated):
                s=freeze_references(c,m,base)
            self.assertEqual(s['gpu_compatibility'],m['gpu_compatibility'])
            self.assertEqual({r['gpu']['uuid'] for r in s['reference_receipts'] if 'gpu' in r},
                             {'fixture-device-0','fixture-device-1','fixture-device-2'})
            p=base/'task.json';raw=json.loads(p.read_text())
            raw['preflight']['gpu']['name']='different model';p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'matching GPU'):
                freeze_references(c,m,base)
            raw['preflight']['gpu'].update(name='CPU fixture model',memory_mib=99.)
            p.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'matching GPU'):
                freeze_references(c,m,base)

    def test_serial_reference_uses_full_pinned_pool_and_observed_ready(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);c,m=self.setup_receipts(base)
            path=base/'task.json';raw=json.loads(path.read_text())
            records=raw['request_records']
            pool=[dict(task_type='text',pool_id=f'p{i}',sample={'task':'text'}) for i in range(len(records))]
            pool_path=base/'inputs.jsonl'
            pool_path.write_text(''.join(json.dumps(r)+'\n' for r in pool))
            sha=hashlib.sha256(pool_path.read_bytes()).hexdigest()
            c.catalog['quality']['identity']['input_pool_sha256']=sha
            c.catalog[norm.KEY]['input_identity']=dict(c.catalog['quality']['identity'])
            for name in ('task','ep1','ep2'):
                p=base/(name+'.json');r=json.loads(p.read_text())
                r['preflight']['sample_pool_sha256']=sha
                if name=='task':
                    r['transport_config']['arrival_process']='isolated_serial'
                    r['isolated_reference_plan']=dict(schema='gsdeploy.isolated_reference_plan.v1',
                        task='text',input_pool=dict(path=str(pool_path),sha256=sha),
                        pool_ids=[v['pool_id'] for v in pool])
                    for i,row in enumerate(r['request_records']):
                        row['pool_id']=f'p{i}'
                        row['worker_response']={'end_perf_ns':row['completion_time_ns']-1}
                        row['previous_handler_next_request_ready']=dict(clock='host_perf_counter_ns',
                            boundary='previous_handler_next_request_ready',
                            perf_ns=row['completion_time_ns']+1,received_time_ns=row['completion_time_ns']+2)
                    r['measurement_window']['trace_sha256']=trace_digest([
                        ((row['scheduled_time_ns']-r['measurement_window']['start_time_ns'])/1e9,'text')
                        for row in r['request_records']])
                tr=r['transport_config'];w=r['measurement_window']
                r['energy_evidence']['profile_key']=energy_key(r['plan']['deployment'],r['plan']['routing'],
                    tr['workload_rps'],w['duration_seconds'],tr['random_seed'],tr['arrival_process'],catalog=c.catalog)
                p.write_text(json.dumps(r))
            evaluated=dict(stability={'stability_state_v2':'steady'},
                           utility=dict(constraints_verified=True,quality_identity_verified=True))
            with patch('gsdeploy.evaluation.evaluate_run',return_value=evaluated):
                result=freeze_references(c,m,base)
            self.assertEqual(result['tasks']['text']['deadline_ms'],4.)
            original=path.read_text();r=json.loads(original)
            r['request_records'][0]['previous_handler_next_request_ready']['perf_ns']=None
            path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'READY'):
                freeze_references(c,m,base)
            r=json.loads(original);r['request_records'].pop();path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'entire frozen task'):
                freeze_references(c,m,base)


if __name__=='__main__': unittest.main()
