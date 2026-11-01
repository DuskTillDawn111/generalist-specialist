"""Prospective CPU resource reporting checks; fixtures do not prove GPU accuracy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.resources_validation import seal,evaluate
from tools.predictor_validation_20261027.empirical_resources import SCHEMA,resource_context,RESIDENTS


class ResourceValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        aliases={a:{'family':f} for a,f in RESIDENTS.items()}
        key,members=resource_context({},aliases)
        self.model={'schema':SCHEMA,'execution_context_sha256':'fixture','provenance':{'split':'training'},'contexts':{key:{'members':members,'evidence':'CPU-fixture',
             'scenarios':{'mean':{'watts':100.,'peak_memory_mib':500.},'max':{'watts':110.,'peak_memory_mib':550.}}}}}
        self.plan={'execution_context_sha256':'fixture','jobs':[{'id':'job','split':'validation'}]}
        for name,value in [('model',self.model),('plan',self.plan),('catalog',{'safe_factor':1.1})]:self.write(name,value)
        seal(self.root/'model.json',self.root/'plan.json',self.root/'catalog.json',self.root/'protocol.json')
        protocol=json.loads((self.root/'protocol.json').read_text());self.start=protocol['frozen_ns']+1_000_000
        self.raw={'execution_context_sha256':'fixture','observation_start_ns':self.start,'observation_end_ns':self.start+10_000_000,
            'batches':[{'job_id':'job'}],'board_samples':[{'time_ns':self.start+i*1_000_000,'watts':100.,'owned_used_mib':500.,'owned_aliases':list(RESIDENTS),'foreign_pids':[]} for i in range(11)]}
        self.traces=[{'id':'job','split':'validation','observation_window_ns':{'start':self.start,'end':self.start+10_000_000},'requests':[]}]

    def write(self,name,value):
        path=self.root/(name+'.json');path.write_text(json.dumps(value));return path

    def assess(self):
        path=self.write('raw',self.raw)
        with patch('tools.predictor_validation_20261027.resources_validation.map_native_collection',return_value={'traces':self.traces}):
            return evaluate(self.root/'protocol.json',path)

    def test_exact_energy_memory_and_safe_factor_once(self):
        report=self.assess();mean=report['scenarios']['mean']
        self.assertAlmostEqual(mean['whole_window']['observed_joules'],1.)
        self.assertEqual(mean['whole_window']['predicted_joules'],1.)
        self.assertEqual(mean['raw_owned_peak']['safe_factor_application_count'],1)
        self.assertEqual(mean['raw_owned_peak']['safe_predicted_mib'],550.)
        self.assertFalse(mean['raw_owned_peak']['capacity_bound'])
        self.assertIsNone(report['global_pass'])

    def test_preseal_required(self):
        self.raw['observation_start_ns']=0
        with self.assertRaisesRegex(ValueError,'sealed before'):self.assess()

    def test_changed_model_rejected(self):
        self.model['contexts']={};self.write('model',self.model)
        with self.assertRaisesRegex(ValueError,'Changed sealed'):self.assess()

    def test_validation_only_and_foreign_rejected(self):
        self.traces[0]['split']='training'
        with self.assertRaisesRegex(ValueError,'Independent validation'):self.assess()
        self.traces[0]['split']='validation';self.raw['board_samples'][0]['foreign_pids']=[4]
        with self.assertRaisesRegex(ValueError,'Complete owned'):self.assess()

    def test_unseen_context_stays_unknown(self):
        self.traces[0]['requests']=[{'alias':'text#0','family':'text','handler_start_ns':self.start,'handler_end_ns':self.start+10_000_000}]
        report=self.assess()['scenarios']['mean']
        self.assertIsNone(report['whole_window']['predicted_joules'])
        self.assertIsNone(report['raw_owned_peak']['predicted_mib'])
        self.assertTrue(report['unknown_context_ids'])

    def test_max_scenario_has_descriptive_error_only(self):
        report=self.assess()['scenarios']['max']
        self.assertAlmostEqual(report['whole_window']['absolute_relative_error'],.1)
        self.assertEqual(report['raw_owned_peak']['underestimate_mib'],0)
        self.assertFalse(report['raw_owned_peak']['capacity_bound'])

if __name__=='__main__':unittest.main()
