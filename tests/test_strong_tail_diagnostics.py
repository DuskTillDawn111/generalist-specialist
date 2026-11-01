"""Post-hoc interval accounting and per-alias queues without result reads."""
import ast
from collections import defaultdict
import math
from pathlib import Path
import statistics
import unittest

PATH=Path(__file__).resolve().parents[1]/'tools/spillover_context_20261022/diagnose_strong_tail.py'
tree=ast.parse(PATH.read_text());ns=dict(defaultdict=defaultdict,math=math,statistics=statistics)
for name in ('pct','stat','fifo','overlap_seconds'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),ns)


class StrongTailDiagnosticsTests(unittest.TestCase):
    def test_background_overlap_unions_intervals_without_double_count(self):
        self.assertEqual(ns['overlap_seconds'](1,5,[(0,2),(1.5,3),(4,6)]),3.)
        self.assertEqual(ns['overlap_seconds'](1,5,[(5,7)]),0.)
        self.assertEqual(ns['overlap_seconds'](1,5,[]),0.)

    def test_exact_cohort_service_block_swap_does_not_mix_alias_queues(self):
        rows=[dict(task='text',alias='a',sample_id='x',arrival_seconds=0.),
            dict(task='text',alias='b',sample_id='y',arrival_seconds=.1),
            dict(task='text',alias='a',sample_id='z',arrival_seconds=.2)]
        result=ns['fifo'](rows,[1000.,100.,100.]);swapped=ns['fifo'](rows,[100.,100.,100.])
        self.assertAlmostEqual(result['rows'][1]['latency_ms'],swapped['rows'][1]['latency_ms'])
        self.assertAlmostEqual(result['rows'][2]['latency_ms'],900.)
        self.assertAlmostEqual(swapped['rows'][2]['latency_ms'],100.)


if __name__=='__main__':unittest.main()
