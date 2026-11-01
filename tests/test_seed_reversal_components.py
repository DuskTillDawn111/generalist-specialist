"""FIFO component arithmetic without research-result reads."""
import ast
from collections import defaultdict
import math
from pathlib import Path
import statistics
import unittest

PATH=Path(__file__).resolve().parents[1]/'tools/spillover_context_20261022/diagnose_seed_reversal.py'
tree=ast.parse(PATH.read_text());ns=dict(defaultdict=defaultdict,math=math,statistics=statistics)
for name in ('pct','replay'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),ns)


class SeedReversalComponentsTests(unittest.TestCase):
    def test_alias_queues_and_measurement_window_count(self):
        rows=[dict(task='text',pool_id='i',alias='a',arrival_seconds=0.),
            dict(task='text',pool_id='j',alias='b',arrival_seconds=.1),
            dict(task='text',pool_id='k',alias='a',arrival_seconds=.2)]
        config=dict(horizon_seconds=1.,workload={'text':1.},latency_reference_ms={'text':1000.})
        result=ns['replay'](rows,[1000.,100.,1000.],config)
        self.assertAlmostEqual(result['details'][1]['finish_seconds'],.2)
        self.assertAlmostEqual(result['details'][2]['queue_ms'],800.)
        self.assertAlmostEqual(result['terms']['T'],1/3)

    def test_submit_release_changes_only_its_alias_fifo(self):
        rows=[dict(task='text',pool_id='i',alias='a',arrival_seconds=0.),dict(task='text',pool_id='j',alias='b',arrival_seconds=.1)]
        config=dict(horizon_seconds=1.,workload={'text':1.},latency_reference_ms={'text':1000.})
        result=ns['replay'](rows,[100.,100.],config,[.5,.1])
        self.assertAlmostEqual(result['details'][0]['latency_ms'],600.)
        self.assertAlmostEqual(result['details'][1]['latency_ms'],100.)


if __name__=='__main__':unittest.main()
