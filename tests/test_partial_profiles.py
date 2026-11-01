"""Solo evidence preservation using CPU synthetic intervals only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.partial_profiles import fit_partial_solo


def row(rid,alias='s0',family='text',duration=10):
    return {'request_id':rid,'alias':alias,'family':family,'task':'text','pool_id':'x','handler_start_ns':0,'handler_end_ns':duration*1_000_000,
            'segments':[{'id':'handler','start_ns':0,'end_ns':duration*1_000_000}]}


def trace(index,rows,split='training'):
    return {'id':str(index),'split':split,'execution_context_sha256':'fixture','requests':rows}


class PartialTests(unittest.TestCase):
    def fit(self,traces):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'CPU-fixture.json';source.write_text(json.dumps(traces))
            with patch('tools.predictor_validation_20261027.partial_profiles._traces',return_value=(traces,{})):
                return fit_partial_solo(source)

    def test_solo_survives_unidentifiable_shared_observations(self):
        traces=[trace(0,[row('a')]),trace(1,[row('b')]),trace(2,[row('c'),row('d',alias='g',family='generalist_2')])]
        result=self.fit(traces)
        self.assertEqual(result['coverage']['accepted_solo_groups'],1)
        self.assertEqual(result['coverage']['shared_contexts_inferred'],0)
        self.assertEqual(result['coverage']['excluded_non_solo_handlers'],2)
        self.assertEqual(result['accepted_solo_profiles'][0]['progress'],1)

    def test_whole_handler_overlap_excludes_even_nonoverlapping_target_subphase(self):
        value=row('a',duration=10);value['segments'][0]['end_ns']=1_000_000
        other=row('g',alias='g',family='generalist_2',duration=3);other['handler_start_ns']=2_000_000
        result=self.fit([trace(0,[value,other])])
        self.assertEqual(result['coverage']['target_groups'],0)

    def test_two_independent_handlers_required(self):
        result=self.fit([trace(0,[row('a')])])
        self.assertFalse(result['known'])

    def test_duplicate_observation_never_counted_twice(self):
        with self.assertRaisesRegex(ValueError,'Repeated'):
            self.fit([trace(0,[row('a')]),trace(0,[row('a')])])

    def test_validation_not_fitted(self):
        result=self.fit([trace(0,[row('a')]),trace(1,[row('b')]),trace(2,[row('c',duration=1000)],split='validation')])
        self.assertEqual(result['accepted_solo_profiles'][0]['baseline_ms'],10)
        self.assertEqual(result['coverage']['validation_batches_excluded'],1)

    def test_reconstruction_threshold_not_relaxed(self):
        result=self.fit([trace(0,[row('a')]),trace(1,[row('b',duration=100)])])
        self.assertFalse(result['known']);self.assertGreater(result['fit_diagnostics'][0]['max_relative_work_error'],.15)

if __name__=='__main__':unittest.main()
