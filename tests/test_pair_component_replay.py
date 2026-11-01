"""Check queue arithmetic and clock conventions used in post-hoc diagnosis."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_component_replay_test',ROOT/'tools/spillover_context_20261022/diagnose_pair_replay.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ComponentReplayTests(unittest.TestCase):
    def row(self,index,arrival,alias):
        return dict(index=index,arrival_seconds=arrival,alias=alias,pool_id=str(index),task='text')

    def test_distinct_aliases_advance_independent_fifo_clocks(self):
        rows=[self.row(0,0,'a'),self.row(1,.25,'a'),self.row(2,.5,'b')]
        result=module.replay(rows,lambda _:1000)
        self.assertEqual([r['latency_ms'] for r in result['rows']],[1000,1750,1000])
        self.assertEqual([r['queue_ms'] for r in result['rows']],[0,750,0])
        self.assertAlmostEqual(result['per_task']['text']['p95_ms'],1675)

    def test_submit_release_delay_keeps_latency_origin_at_scheduled_arrival(self):
        rows=[self.row(0,0,'a'),self.row(1,.25,'a')]
        result=module.replay(rows,lambda _:1000,lambda r:r['arrival_seconds']+.5)
        self.assertEqual([r['latency_ms'] for r in result['rows']],[1500,2250])
        self.assertEqual([r['queue_ms'] for r in result['rows']],[0,750])

    def test_invalid_service_and_release_cannot_enter_replay(self):
        rows=[self.row(0,1,'a')]
        for value in [0,-1,float('nan'),float('inf')]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                module.replay(rows,lambda _:value)
        with self.assertRaises(ValueError):module.replay(rows,lambda _:1000,lambda _:0)


if __name__=='__main__':unittest.main()
