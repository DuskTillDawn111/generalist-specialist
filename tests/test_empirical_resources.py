"""Analytical telemetry interpolation/resource provenance checks on CPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.empirical_resources import (fit_resources,EmpiricalResourceProvider,integrate_samples,resource_context,RESIDENTS)


class ResourceTests(unittest.TestCase):
    def raw(self):
        samples=[{'time_ns':i*1_000_000,'watts':100+i*10,'owned_used_mib':500+i,
            'owned_aliases':list(RESIDENTS),'foreign_pids':[],'query_duration_ns':100} for i in range(11)]
        return {'execution_context_sha256':'CPU-fixture','board_samples':samples,'observation_start_ns':0,'observation_end_ns':10_000_000}

    def traces(self):
        return [{'id':'a','split':'training','observation_window_ns':{'start':1_000_000,'end':9_000_000},
                 'requests':[{'alias':'text#0','family':'text','handler_start_ns':2_000_000,'handler_end_ns':8_000_000}]}]

    def fit(self,raw=None,traces=None):
        raw=raw or self.raw();traces=traces or self.traces()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'CPU-fixture.json';path.write_text(json.dumps(raw))
            with patch('tools.predictor_validation_20261027.empirical_resources.map_native_collection',return_value={'traces':traces}):
                return fit_resources(path)

    def test_integration_uses_internal_sample_knots(self):
        samples=[{'time_ns':0,'watts':0},{'time_ns':5,'watts':10},{'time_ns':10,'watts':0}]
        self.assertEqual(integrate_samples(samples,0,10),50)
        self.assertEqual(integrate_samples(samples,2,8),42)

    def test_empirical_full_context_power_includes_idle_and_peak_scope(self):
        result=self.fit();aliases={a:{'family':f} for a,f in RESIDENTS.items()}
        provider=EmpiricalResourceProvider(result,'CPU-fixture')
        power=provider.power({'text#0':{}},aliases,'mean')
        self.assertAlmostEqual(power['watts'],150)
        self.assertTrue(provider.power({},aliases,'mean')['known'])
        memory=provider.memory({'text#0':{}},aliases,'max')
        self.assertEqual(memory['peak_memory_mib'],510) # max(run idle, batch sampled peak)
        self.assertFalse(memory['capacity_bound']);self.assertFalse(memory['safe_factor_applied'])
        self.assertFalse(result['telemetry']['allocator_peaks_summed'])

    def test_unseen_threeway_is_unknown_not_pairwise_sum(self):
        result=self.fit();aliases={a:{'family':f} for a,f in RESIDENTS.items()}
        provider=EmpiricalResourceProvider(result,'CPU-fixture')
        self.assertFalse(provider.power({'text#0':{},'text#1':{},'generalist_2#0':{}},aliases,'mean')['known'])
        self.assertFalse(provider.memory({'text#0':{},'text#1':{}},aliases,'mean')['known'])

    def test_validation_telemetry_not_mined(self):
        traces=self.traces();traces[0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'Training-only'):self.fit(traces=traces)

    def test_sampling_gap_over_two_seconds_unknown(self):
        raw=self.raw();raw['board_samples'][-1]['time_ns']=3_000_000_000
        with self.assertRaisesRegex(ValueError,'2-second'):self.fit(raw)

    def test_foreign_process_and_missing_idle_resident_rejected(self):
        for foreign in (True,False):
            raw=self.raw()
            if foreign:raw['board_samples'][0]['foreign_pids']=[55]
            else:raw['board_samples'][0]['owned_aliases'].pop()
            with self.assertRaisesRegex(ValueError,'All owned'):self.fit(raw)

    def test_full_family_multiplicity_and_config_binding(self):
        aliases={a:{'family':f} for a,f in RESIDENTS.items()}
        self.assertNotEqual(resource_context({'text#0':{}},aliases)[0],resource_context({'text#0':{},'text#1':{}},aliases)[0])
        with self.assertRaisesRegex(ValueError,'Matching explicit'):
            EmpiricalResourceProvider(self.fit(),'another config')

if __name__=='__main__':unittest.main()
