import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_fine_test',ROOT/'tools/spillover_context_20261022/fine_routes.py')
fine=importlib.util.module_from_spec(spec);spec.loader.exec_module(fine)


class FineRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet=json.loads((ROOT/'artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/experiment-packet.json').read_text())
        cls.config=json.loads(Path(cls.packet['system_config']).read_text())

    def test_route_and_input_exposure_matches_existing_frozen_reconstruction(self):
        old=fine.exposure.build_case(self.config,self.packet['samples'],1.85,107,1/8)
        new=fine.cohort(self.config,self.packet['samples'],1.85,107,'1/8')
        self.assertEqual(old['trace_sha256'],new['trace_sha256'])
        self.assertEqual(old['predicted_quality_Q_loss'],new['historical_Q_loss'])
        self.assertEqual([(r['scheduled_offset_seconds'],r['pool_id'],r['alias']) for r in old['scheduled_records']],
                         [(r['arrival_seconds'],r['pool_id'],r['alias']) for r in new['rows']])

    def test_tiny_route_keeps_arrivals_inputs_and_both_specialist_aliases(self):
        a=fine.cohort(self.config,self.packet['samples'],1.85,107,'0')
        b=fine.cohort(self.config,self.packet['samples'],1.85,107,'1/64')
        self.assertEqual([(r['arrival_seconds'],r['pool_id']) for r in a['rows']],
                         [(r['arrival_seconds'],r['pool_id']) for r in b['rows']])
        aliases={r['alias'] for r in b['rows'] if r['task']=='text'}
        self.assertEqual(aliases,{'text#0','text#1','generalist_2#0'})
        count=sum(r['alias']=='generalist_2#0' for r in b['rows'] if r['task']=='text')
        total=sum(r['task']=='text' for r in b['rows'])
        self.assertLess(abs(count-total/64),2)

    def test_fifo_serializes_tasks_on_same_alias_and_preserves_other_alias(self):
        rows=[dict(alias='g',task='image',arrival_seconds=0),dict(alias='g',task='text',arrival_seconds=.5),
              dict(alias='s',task='text',arrival_seconds=.5)]
        result=fine.fifo(rows,lambda row:2000 if row['task']=='image' else 1000,horizon=2)
        self.assertEqual(result['alias_finish_seconds'],{'g':3.,'s':1.5})
        self.assertEqual(result['per_task']['text']['mean_ms'],1750.)
        self.assertAlmostEqual(result['completion_fraction'],2/3)

    def test_invalid_service_and_invalid_explicit_workloads(self):
        with self.assertRaises(ValueError):
            fine.fifo([dict(alias='s',task='text',arrival_seconds=0)],lambda row:float('nan'))
        for rate,seed,share in [(0,107,'0'),(1.8,-1,'0'),(1.8,107,'-1/32')]:
            with self.assertRaises(ValueError):fine.cohort(self.config,self.packet['samples'],rate,seed,share)


if __name__=='__main__':unittest.main()
