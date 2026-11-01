"""Generalized family/task resource audit arithmetic with CPU fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.background_resources import fit_generalized_resources,TaskAwareResourceProvider,TaskConditionalResourceProvider
from tools.predictor_validation_20261027.empirical_resources import EmpiricalResourceProvider


class BackgroundResourcesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        path=Path(self.tmp.name)/'fixture';path.write_text('CPU synthetic source')
        self.sources=[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]
        self.residents={'g':'generalist_2','s':'text'}
        self.raw={'status':'completed','backend':'native','kind':'measured','execution_context_sha256':'CPU-fixture',
            'observation_start_ns':0,'observation_end_ns':10_000_000,
            'board_samples':[{'time_ns':i*1_000_000,'watts':100+i*10,'owned_used_mib':500,
                              'owned_aliases':['g','s'],'foreign_pids':[]} for i in range(11)]}
        self.traces=[{'id':'a','split':'training','observation_window_ns':{'start':0,'end':10_000_000},
            'requests':[{'request_id':'a','alias':'g','family':'generalist_2','task':'image','handler_start_ns':1_000_000,'handler_end_ns':4_000_000},
                        {'request_id':'b','alias':'g','family':'generalist_2','task':'asr','handler_start_ns':5_000_000,'handler_end_ns':8_000_000}]}]

    def fit(self):return fit_generalized_resources(self.raw,self.traces,self.residents,sources=self.sources)

    def test_same_family_different_tasks_audited_not_silently_equivalent(self):
        result=self.fit();audit=result['task_specific_resource_omission_audit']
        self.assertEqual(audit['known_tasks'],['asr','image'])
        self.assertEqual(audit['family_contexts_with_multiple_task_mixes'],1)
        mixes=[c for c in audit['active_task_contexts'].values() if c['members']]
        self.assertEqual(len(mixes),2)
        self.assertNotEqual(mixes[0]['mean_watts'],mixes[1]['mean_watts'])
        self.assertTrue(audit['unknown_task_mix_may_not_transfer'])
        provider=EmpiricalResourceProvider(result,'CPU-fixture')
        self.assertTrue(provider.power({'g':{}},{'g':{'family':'generalist_2'}},'mean')['known'])

    def test_unseen_task_mix_is_unknown_even_with_known_family_count(self):
        provider=TaskAwareResourceProvider(self.fit(),'CPU-fixture')
        aliases={'g':{'family':'generalist_2'}}
        self.assertTrue(provider.power({'g':{'task':'image'}},aliases,'mean')['known'])
        self.assertFalse(provider.power({'g':{'task':'text'}},aliases,'mean')['known'])
        self.assertFalse(provider.memory({'g':{'task':'text'}},aliases,'max')['known'])
        self.assertFalse(provider.power({'g':{}},aliases,'mean')['known'])

    def test_foreign_or_resident_missing_rejected(self):
        self.raw['board_samples'][0]['foreign_pids']=[44]
        with self.assertRaisesRegex(ValueError,'residents owned'):self.fit()

    def test_validation_cannot_enter_fit(self):
        self.traces[0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'Training-only'):self.fit()

    def test_same_alias_overlap_rejected(self):
        self.traces[0]['requests'][1]['handler_start_ns']=2_000_000
        with self.assertRaisesRegex(ValueError,'alias overlap'):self.fit()

    def test_resource_source_receipt_required(self):
        Path(self.sources[0]['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'Changed generalized'):self.fit()

    def test_task_conditional_power_uses_direct_mean_without_family_pool(self):
        result=self.fit();provider=TaskConditionalResourceProvider(result,'CPU-fixture')
        aliases={'g':{'family':'generalist_2'}}
        image=provider.power({'g':{'task':'image'}},aliases,'mean')
        asr=provider.power({'g':{'task':'asr'}},aliases,'mean')
        self.assertAlmostEqual(image['watts'],125)
        self.assertAlmostEqual(asr['watts'],165)
        self.assertNotEqual(image['watts'],asr['watts'])
        self.assertFalse(provider.power({'g':{'task':'image'}},aliases,'max')['known'])

    def test_task_conditional_memory_uses_own_trial_peaks_with_idle_floor(self):
        result=self.fit()
        for row in result['task_specific_resource_omission_audit']['active_task_contexts'].values():
            if row['members']:
                row['mean_trial_sampled_peak_mib']=400 if row['members'][0]['task']=='image' else 600
                row['max_trial_sampled_peak_mib']=700 if row['members'][0]['task']=='image' else 800
        provider=TaskConditionalResourceProvider(result,'CPU-fixture');aliases={'g':{'family':'generalist_2'}}
        self.assertEqual(provider.memory({'g':{'task':'image'}},aliases,'mean')['peak_memory_mib'],500)
        self.assertEqual(provider.memory({'g':{'task':'asr'}},aliases,'mean')['peak_memory_mib'],600)
        self.assertEqual(provider.memory({'g':{'task':'image'}},aliases,'max')['peak_memory_mib'],700)
        self.assertFalse(provider.memory({'g':{'task':'image'}},aliases,'max')['safe_factor_applied'])

    def test_task_conditional_unobserved_or_mismatched_identity_stays_unknown(self):
        provider=TaskConditionalResourceProvider(self.fit(),'CPU-fixture');aliases={'g':{'family':'generalist_2'}}
        for active in ({'g':{'task':'text'}},{'g':{}},{'g':{'task':'image','family':'text'}},{'unknown':{'task':'image'}}):
            self.assertFalse(provider.power(active,aliases,'mean')['known'])
            self.assertFalse(provider.memory(active,aliases,'mean')['known'])
        self.assertTrue(provider.power({},aliases,'mean')['known'])
        self.assertTrue(provider.memory({},aliases,'mean')['known'])

if __name__=='__main__':unittest.main()
