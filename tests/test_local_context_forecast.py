"""Input-domain convex hull and unknown FIFO semantics, without result reads."""
import ast
from collections import defaultdict
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import linprog

PATH=Path(__file__).resolve().parents[1]/'tools/spillover_context_20261022/forecast_local108.py'
tree=ast.parse(PATH.read_text());ns=dict(np=np,linprog=linprog,defaultdict=defaultdict)
for name in ('in_context_hull','fifo_unknown'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),ns)


class LocalContextForecastTests(unittest.TestCase):
    def test_marginal_box_is_not_context_convex_hull(self):
        contexts=[[0.,0.],[1.,1.]]
        self.assertTrue(ns['in_context_hull']([.5,.5],contexts)[0])
        self.assertFalse(ns['in_context_hull']([.2,.8],contexts)[0])
        self.assertFalse(ns['in_context_hull']([1.1,1.1],contexts)[0])

    def test_unknown_service_poison_only_same_alias_suffix(self):
        rows=[dict(alias='a',arrival_seconds=0.,predicted_median_scale_service_ms=100.),
            dict(alias='a',arrival_seconds=.1,predicted_median_scale_service_ms=None),
            dict(alias='b',arrival_seconds=.2,predicted_median_scale_service_ms=100.),
            dict(alias='a',arrival_seconds=.3,predicted_median_scale_service_ms=100.)]
        result=ns['fifo_unknown'](rows,1.,1.)
        self.assertEqual(result[0]['finish_seconds'],.1)
        self.assertIsNone(result[1]['finish_seconds']);self.assertIsNone(result[3]['finish_seconds'])
        self.assertAlmostEqual(result[2]['finish_seconds'],.3)


if __name__=='__main__':unittest.main()
