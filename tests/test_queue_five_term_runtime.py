"""Prospective objective binding, causal mock lifecycle, and baseline route mask."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.queue_five_term_20261023 import native_runner, protocol, evaluation, policy


def sources():
    return {str(p.resolve()): protocol.receipt(p)['sha256']
            for p in Path(protocol.__file__).parent.glob('*.py')}


class Provider:
    def service(self, row, active, aliases, scenario):
        return dict(known=True, context_supported=True, service_ms=100. if aliases[row['alias']]['family']=='text' else 10.)
    def residual(self, row, age, active, aliases, scenario):
        return dict(known=True, tail_supported=True, remaining_ms=10.)
    def power(self, active, aliases, scenario):
        return dict(known=True, watts=100.)


class FiveTermTests(unittest.TestCase):
    def template(self, models=None, services=None, *, workload=None, refs=None):
        models = models or {'text': {'supported_tasks':['text']}, 'generalist_2': {'supported_tasks':['text']}}
        services = services or {'text': {'text': {'t':100.}}, 'generalist_2': {'text': {'t':10.}}}
        workload = workload or {'text':1.85}
        deployment={'text':2, 'generalist_2':1}
        if 'ocr' in models:
            deployment['ocr']=1
        return protocol.original.make_policy_contract(
            deployment, services,
            {m:{t:.1 for t in v['supported_tasks']} for m,v in models.items()}, dict(policy.FIXED_WEIGHTS),
            refs or {'text':100.}, strategy='queue_latency', models=models, text_rate=workload['text'],
            background_routing={t:{'ocr' if t=='ocr' and 'ocr' in models else 'generalist_2':r} for t,r in workload.items() if t!='text'})

    def test_fixed_objective_rejects_old_weights_and_tail_does_not_affect_cost(self):
        aliases={'s':dict(family='text',supported_tasks=['text']), 'g':dict(family='generalist_2',supported_tasks=['text'])}
        def make():
            p=policy.CoupledUtilityPolicy(aliases, Provider(), {'text':{'text':.1},'generalist_2':{'text':.2}},
                                          dict(policy.FIXED_WEIGHTS), {'text':100.}, {'text':1.})
            p.start_phase(0,1_000_000_000);p.set_arrival('r',0)
            return p
        snapshot={a:dict(outstanding=[],last_completed_ns=None) for a in aliases}
        result=make().choose('text','t',0,snapshot)
        with patch.object(policy,'percentile',return_value=1e12):
            changed=make().choose('text','t',0,snapshot)
        self.assertEqual(result['alias'],changed['alias'])
        for left,right in zip(result['audit'],changed['audit']):
            self.assertEqual(left['score'],right['score'])
            self.assertNotEqual(left['scenarios']['mean']['terms']['P'],right['scenarios']['mean']['terms']['P'])
        with self.assertRaisesRegex(ValueError,'Fixed five-term'):
            policy.validate_weights({t:1/6 for t in 'QTLPME'})
        bad=dict(policy.FIXED_WEIGHTS,P=1e-12,L=.2-1e-12)
        with self.assertRaisesRegex(ValueError,'Fixed five-term'):
            policy.validate_weights(bad)

    def test_legacy_mask_preserves_latency_mode_and_changes_only_allowed_routes(self):
        template=self.template()
        on=protocol.make_legacy_contract(template,allow_spill=True,sources=sources())
        off=protocol.make_legacy_contract(template,allow_spill=False,sources=sources())
        self.assertEqual(on['strategy'],off['strategy']);self.assertEqual(on['mode'],'latency')
        snapshot={a:dict(outstanding=[],last_completed_ns=None) for a in on['aliases']}
        a=protocol.make_dispatcher(on);b=protocol.make_dispatcher(off)
        self.assertEqual(a.legacy.policy.mode,b.legacy.policy.mode)
        self.assertEqual(a.legacy.policy.weights['Q'],0.)
        self.assertEqual(a.choose('text','t',0,snapshot)['alias'],'generalist_2#0')
        self.assertTrue(b.choose('text','t',0,snapshot)['alias'].startswith('text#'))
        wrong=copy.deepcopy(off);wrong['baseline_switch']['mode']='quality_aware'
        with self.assertRaisesRegex(ValueError,'route-switch'):
            protocol.validate_contract(wrong)

    def test_cpu_mock_lifecycle_and_replay_use_new_dispatcher_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);asset=root/'asset';asset.write_text('cpu')
            samples=root/'runtime-inputs.jsonl'
            records=[dict(pool_id='t',task_type='text',sample=dict(context='c',question='q',label='ok'))]
            records += [dict(pool_id=t,task_type=t,sample=dict(input=str(asset),label='ok')) for t in ['image','asr','ocr']]
            samples.write_text(''.join(json.dumps(r)+'\n' for r in records))
            models={'text':dict(supported_tasks=['text'],service_time_ms={'text':1.}),
                    'generalist_2':dict(supported_tasks=['text','image','asr','ocr'],service_time_ms={t:1. for t in ['text','image','asr','ocr']}),
                    'ocr':dict(supported_tasks=['ocr'],service_time_ms={'ocr':1.})}
            workload={'text':40.,'image':10.,'asr':10.,'ocr':10.}
            refs={t:100. for t in workload}
            services={m:{t:{'t' if t=='text' else t:1.} for t in spec['supported_tasks']} for m,spec in models.items()}
            contract=protocol.make_legacy_contract(self.template(models,services,workload=workload,refs=refs),allow_spill=False,sources=sources())
            config=dict(catalog=dict(models=models,safe_factor=1.1,memory_table_mib={m:{'1':10.,'2':20.} for m in models},
                                     quality=dict(identity=dict(input_pool_sha256=protocol.receipt(samples)['sha256']))),
                        workload=workload,weights=dict(policy.FIXED_WEIGHTS),latency_reference_ms=refs,gpu_memory_mib=1000.)
            system=SimpleNamespace(**config,as_dict=lambda:copy.deepcopy(config))
            wrong=copy.deepcopy(system);wrong.weights={t:1/6 for t in 'QTLPME'}
            with self.assertRaisesRegex(ValueError,'Fixed five-term'):
                native_runner.run(wrong,protocol.nominal_plan(contract),samples,contract)
            with patch('subprocess.run',side_effect=AssertionError('No GPU query in mock')):
                raw=native_runner.run(system,protocol.nominal_plan(contract),samples,contract,backend='mock',
                                      duration=.15,warmup=.1,seed=113,arrival_process='deterministic_periodic',
                                      startup_timeout=10.,drain_timeout=10.)
            self.assertEqual(raw['status'],'completed',raw['errors'])
            self.assertTrue(raw['request_records']);self.assertEqual(raw['config']['weights'],policy.FIXED_WEIGHTS)
            self.assertTrue(evaluation.audit_phase(system,raw)['verified'])
            self.assertTrue(evaluation.audit_phase(system,raw,'warmup')['verified'])
            self.assertFalse(raw['energy_evidence']['coverage_valid']);self.assertIsNone(raw['evaluation']['utility']['U'])
            self.assertTrue(all(r['model'].startswith('text#') for r in raw['request_records'] if r['task']=='text'))
            template=self.template(models,services,workload=workload,refs=refs)
            resource=Path('artifacts/queue-coupled-study-20261023/controlled-resource-fit-v2/resources.json')
            contracts=[protocol.make_legacy_contract(template,allow_spill=True,sources=sources()),
                       protocol.make_contract(template,resource,allow_spill=True,sources=sources())]
            for contract in contracts:
                with patch('subprocess.run',side_effect=AssertionError('No GPU query in mock')):
                    checked=native_runner.run(system,protocol.nominal_plan(contract),samples,contract,backend='mock',
                                              duration=.15,warmup=.1,seed=113,arrival_process='deterministic_periodic',
                                              startup_timeout=10.,drain_timeout=10.)
                self.assertEqual(checked['status'],'completed',checked['errors'])
                self.assertTrue(evaluation.audit_phase(system,checked)['verified'])
                self.assertTrue(evaluation.audit_phase(system,checked,'warmup')['verified'])
                self.assertIsNone(checked['evaluation']['utility']['U'])

    def test_native_binding_persists_new_objective_before_system_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'base.json'
            old={t:1/6 for t in 'QTLPME'}
            base.write_text(json.dumps(dict(config=dict(catalog={},weights=old,drain_seconds=300),inputs=str(root),jobs=[dict(identity={})])))
            identity=dict(horizon_seconds=600,warmup_seconds=30)
            job=dict(id='job',phase='development',seed=401,identity=identity,nominal_plan={'deployment':{}},
                     policy_contract={'weights':dict(policy.FIXED_WEIGHTS)})
            packet=dict(base_packet=protocol.receipt(base),jobs=[job]);captured={}
            def bind(bound, binding, output):
                captured['binding_weights']=copy.deepcopy(bound['config']['weights'])
                return bound['config'],root/'quality.json',{}
            def run(system,*args,**kwargs):
                captured['runtime_weights']=copy.deepcopy(system.weights)
                return dict(status='completed',evaluation={'utility':{'U':None}})
            static=SimpleNamespace(bind_native_job=bind);configuration=SimpleNamespace(SystemConfig=lambda **v:SimpleNamespace(**v))
            with patch.object(protocol,'validate_packet',return_value={'valid':True}), \
                 patch.object(native_runner,'frozen_runtime',return_value=SimpleNamespace(__package__='frozen.runtime')), \
                 patch.object(native_runner.importlib,'import_module',side_effect=[static,configuration]), \
                 patch.object(native_runner,'run',side_effect=run):
                raw=native_runner.execute_job(packet,'job',root/'execution',execute=True)
            self.assertEqual(raw['status'],'completed',raw.get('errors'))
            self.assertEqual(captured,dict(binding_weights=policy.FIXED_WEIGHTS,runtime_weights=policy.FIXED_WEIGHTS))
            self.assertEqual(json.loads(base.read_text())['config']['weights'],old)
            self.assertEqual(raw['objective_binding']['base_weights'],old)


if __name__=='__main__':
    unittest.main()
