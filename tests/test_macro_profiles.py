"""Analytical CPU checks for the separate, explicitly assumed macro model."""
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from tools.predictor_validation_20261027.macro_profiles import _context, fit_macro, DEFINITION


def row(alias, family, start=0, end=10_000_000, pool='x'):
    return {'alias':alias,'family':family,'task':'text','pool_id':pool,'request_id':alias,
            'handler_start_ns':start,'handler_end_ns':end,
            'segments':[{'phase':'macro0','start_ns':start,'end_ns':end}]}


def trace(index, rows, split='training'):
    return {'id':str(index),'split':split,'requests':rows,'execution_context_sha256':'CPU-fixture'}


class MacroTests(unittest.TestCase):
    def test_context_preserves_multiway_and_multiplicity_but_excludes_alias_input(self):
        a,b=row('s0','text'),row('s1','text',pool='y')
        key,members=_context([(a,a['segments'][0]),(b,b['segments'][0])])
        self.assertEqual(members,[{'family':'text','phase':'macro0','count':2}])
        c=row('renamed','text',pool='different')
        self.assertEqual(key,_context([(c,c['segments'][0]),(a,a['segments'][0])])[0])
        self.assertNotEqual(key,_context([(a,a['segments'][0])])[0])

    def fit(self,traces):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'CPU-fixture.json';source.write_text(json.dumps(traces))
            with patch('tools.predictor_validation_20261027.macro_profiles.macro_traces',return_value=(traces,{'fixture':True})):
                return fit_macro(source)

    def test_singleton_baseline_and_validation_exclusion(self):
        result=self.fit([trace(1,[row('s0','text')]),trace(2,[row('s0','text')]),
                         trace(3,[row('s0','text',end=100_000_000)],split='validation')])
        self.assertEqual(result['coverage']['excluded_validation_batches'],1)
        self.assertEqual(result['solo_baselines'][0]['mean_ms'],10)
        self.assertTrue(result['fit_diagnostics'][0]['accepted'])
        context=next(iter(result['supported_contexts'].values()))
        self.assertAlmostEqual(context['progress']['text:macro0'],1)
        self.assertIsNone(context['resource_demand']['sm'])
        self.assertFalse(result['core_enabled'])

    def test_pool_normalization_precedes_explicit_sharing(self):
        traces=[trace(i,[row('s0','text',end=duration,pool=pool)]) for i,(pool,duration) in enumerate([
            ('short',10_000_000),('short',10_000_000),('long',20_000_000),('long',20_000_000)])]
        result=self.fit(traces)
        self.assertTrue(result['fit_diagnostics'][0]['accepted'])
        self.assertEqual(len(result['fit_diagnostics'][0]['cross_input_sharing_diagnostics']),2)

    def test_missing_solo_never_assumes_one(self):
        result=self.fit([trace(1,[row('s0','text'),row('g','generalist_2')])])
        self.assertFalse(result['known'])
        self.assertEqual(len(result['coverage']['unknown_solo_baselines']),2)

    def test_threeway_context_is_not_inferred_from_pairwise(self):
        traces=[trace(i,[row('s0','text')]) for i in range(2)]
        traces += [trace(i+2,[row('g','generalist_2')]) for i in range(2)]
        result=self.fit(traces)
        self.assertTrue(all(sum(m['count'] for m in c['members'])==1 for c in result['supported_contexts'].values()))

    def test_definition_is_fixed_before_holdout(self):
        self.assertEqual(DEFINITION['maximum_relative_work_error'],.15)
        self.assertEqual(DEFINITION['minimum_observations'],2)

if __name__=='__main__':unittest.main()
