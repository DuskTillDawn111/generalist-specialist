"""Probe preparation must preserve the accepted execution context and gates."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.collect_activation_training import POOL, digest, save
from experiments.prepare_activation_phase_pair_probe import prepare
from gsdeploy.forecasting.native_order import SCHEMA as ORDER_SCHEMA
from tests.test_phase_crossed_design import template


class PairPreparationTests(unittest.TestCase):
    def fixture(self, base):
        solo=base/'solo';solo.mkdir();observer=base/'observer.json'
        source=solo/'solo-round1';source.mkdir()
        code=base/'worker.py';code.write_text('# frozen fixture\n');sha=hashlib.sha256(code.read_bytes()).hexdigest()
        sources=[dict(path=str(code),snapshot=str(code),sha256=sha)]
        save(source/'sources.json',sources)
        aliases={f+'#0':f for f in ('generalist_2','text','image','ocr','asr')}
        context=dict(resident_aliases=aliases,input_pool_sha256=hashlib.sha256(POOL.read_bytes()).hexdigest(),
                     source_content=[dict(path=str(code),sha256=sha)])
        receipt=lambda n:str(base/n/'raw.json')+':sha256:fixture'
        solo_receipts=[str(solo/f'solo-round{i}'/'raw.json')+':sha256:fixture' for i in (1,2)]
        save(observer,dict(observer_repeat_check_passed=True,issues=[],sources={f'semantic-{i}':receipt(f'semantic-{i}') for i in (1,2)}))
        save(solo/'description.json',dict(sources=solo_receipts,changed_output_profiles=[]))
        save(solo/'completion-template-diagnostic.json',dict(sources=solo_receipts,paths={'fixture':dict(maximum=.1)}))
        save(solo/'quality-description.json',{})
        t=template();t['execution_context_sha256']=digest(context)
        t['native_order_law']=dict(schema=ORDER_SCHEMA,status='FITTED_UNVALIDATED',execution_context_sha256=digest(context),
            descriptor='post_prime_owner_request_ordinal',physical_cause_identified=False,groups=[],fit_diagnostics=[dict(accepted=True)])
        def load(path):return dict(context=copy.deepcopy(context),source=str(path/'raw.json')+':sha256:fixture')
        return solo,observer,t,context,load

    def test_three_restarts_keep_exact_context_all_phase_pairs_and_idle_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);solo,observer,t,context,load=self.fixture(base);output=base/'probe'
            with patch('experiments.prepare_activation_phase_pair_probe.load_bundle',side_effect=load),patch(
                    'experiments.prepare_activation_phase_pair_probe.build_band_template',return_value=(t,{},[])):
                prepare(solo,observer,output,6111191)
            cohorts=[]
            for i in (1,2,3):
                capture=output/f'phase-pair-round{i}'
                self.assertEqual(json.loads((capture/'context.json').read_text()),context)
                plan=json.loads((capture/'plan.json').read_text());self.assertEqual(plan['split'],'training')
                self.assertEqual(plan['execution_context_sha256'],digest(context))
                self.assertEqual(sum(len(j['requests']) for j in plan['jobs']),153)
                cohorts.append({j['id'] for j in plan['jobs'][:74]})
                self.assertEqual(len(cohorts[-1]),74)
                self.assertTrue(all(j['intended_phase_alignment']['actual_overlap_guaranteed'] is False for j in plan['jobs'][:74]))
                for left,right in zip(plan['jobs'][74::2],plan['jobs'][75::2]):
                    self.assertEqual(len(left['requests']),1);self.assertEqual(right['requests'],[])
                self.assertFalse(plan['physical_predictor_qualified'])
            self.assertEqual(cohorts[0],cohorts[1]);self.assertEqual(cohorts[1],cohorts[2])

    def test_unstable_baseline_does_not_schedule_concurrent_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);solo,observer,t,context,load=self.fixture(base)
            r=json.loads((solo/'completion-template-diagnostic.json').read_text());r['paths']['fixture']['maximum']=.21
            save(solo/'completion-template-diagnostic.json',r)
            with patch('experiments.prepare_activation_phase_pair_probe.load_bundle',side_effect=load):
                with self.assertRaisesRegex(ValueError,'feasible completion baseline'):
                    prepare(solo,observer,base/'rejected',6111191)
            self.assertFalse((base/'rejected').exists())


if __name__=='__main__':unittest.main()
