"""CPU analytical checks, not physical calibration evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.handler_profiles import fit_handler,handler_context,handler_traces,DEFINITION
from tests.test_macro_profiles import row,trace

class HandlerTests(unittest.TestCase):
    def fit(self,traces):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'fixture.json';source.write_text(json.dumps(traces))
            with patch('tools.predictor_validation_20261027.handler_profiles.handler_traces',return_value=(traces,{})):
                return fit_handler(source)

    def test_mapper_returns_trace_list_and_report(self):
        with patch('tools.predictor_validation_20261027.handler_profiles.map_native_collection',return_value={'traces':[{'id':'fixture'}],'mapping_report':{'source':'fixture'}}):
            traces,report=handler_traces('not accessed')
        self.assertEqual(traces,[{'id':'fixture'}]);self.assertEqual(report,{'source':'fixture'})

    def test_target_specific_coefficients_do_not_pool_inputs(self):
        traces=[trace(i,[row('s0','text',pool=pool,end=length)]) for i,(pool,length) in enumerate([
            ('short',10_000_000),('short',10_000_000),('long',20_000_000),('long',20_000_000)])]
        result=self.fit(traces)
        self.assertEqual(result['coverage']['accepted_target_groups'],2)
        self.assertEqual({r['pool_id'] for r in result['fit_diagnostics']},{'short','long'})
        self.assertTrue(all(r['condition_number']==1 for r in result['fit_diagnostics']))
        self.assertTrue(all(r['coefficient_sensitivity']['max_relative_coefficient_change']<1e-12 for r in result['fit_diagnostics']))

    def test_full_context_multiplicity_preserved(self):
        one=handler_context([row('s0','text')])
        two=handler_context([row('s0','text'),row('s1','text')])
        three=handler_context([row('s0','text'),row('s1','text'),row('g','generalist_2')])
        self.assertNotEqual(one[0],two[0]);self.assertNotEqual(two[0],three[0])
        self.assertEqual(two[1],[{'family':'text','count':2}])

    def test_validation_cannot_change_training_result(self):
        trains=[trace(i,[row('s0','text')]) for i in range(2)]
        result=self.fit(trains+[trace(3,[row('s0','text',end=100_000_000)],split='validation')])
        self.assertEqual(result['solo_baselines'][0]['mean_ms'],10)
        self.assertEqual(result['coverage']['validation_batches_excluded'],1)

    def test_residual_violations_rejected_without_threshold_increase(self):
        result=self.fit([trace(0,[row('s0','text')]),trace(1,[row('s0','text',end=100_000_000)])])
        self.assertFalse(result['known'])
        self.assertFalse(result['fit_diagnostics'][0]['accepted'])
        self.assertGreater(result['fit_diagnostics'][0]['max_relative_work_error'],.15)
        self.assertEqual(DEFINITION['maximum_relative_work_error'],.15)
        self.assertEqual(result['accepted_target_contexts'],[])

    def test_missing_solo_unknown(self):
        result=self.fit([trace(0,[row('s0','text'),row('g','generalist_2')])])
        self.assertFalse(result['known'])
        self.assertEqual(len(result['coverage']['unknown_solo_baselines']),2)

if __name__=='__main__':unittest.main()
