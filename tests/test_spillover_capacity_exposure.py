"""CPU contracts: exact formal routing and conditional input quality exposure."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('capacity_exposure',ROOT/'tools/spillover_capacity_20261022/exposure.py')
exposure=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(exposure)


class FormalExposure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path=ROOT/'artifacts/static-spillover-study-20261022/text-capacity-diagnostic-v1/system-config.json'
        if not cls.config_path.exists():
            cls.config_path=ROOT/'artifacts/static-study-v6-20261021/inputs/system-config.json'
        cls.config=json.loads(cls.config_path.read_text())
        cls.samples=ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl'
        cls.api=exposure.frozen_api()

    def case(self,share,config=None):
        return exposure.build_case(config or self.config,self.samples,.8,101,share,self.api)

    def test_exact_runtime_alias_roster(self):
        result=self.case(1/8);rates=result['workload']
        trace=self.api['trace'].make_trace(rates,600,101,self.config['arrival_process'])
        aliases,routes=self.api['trace'].expand_routes(result['deployment'],result['routing'])
        router=self.api['routing'].Router(routes)
        pool=self.api['workload'].SamplePool(self.samples,list(rates),101)
        expected=[]
        for index,(offset,task) in enumerate(trace):
            item=pool.next(task);alias=router.choose(task)
            expected.append((index,offset,task,item['pool_id'],alias,aliases[alias]))
        actual=[(r['arrival_index'],r['scheduled_offset_seconds'],r['task'],r['pool_id'],r['alias'],r['family']) for r in result['scheduled_records']]
        self.assertEqual(actual,expected)

    def test_same_input_trace_across_all_routes(self):
        results=[self.case(s) for s in exposure.SHARES]
        self.assertEqual(len({r['trace_sha256'] for r in results}),1)
        rosters=[[(r['scheduled_offset_seconds'],r['task'],r['pool_id']) for r in case['scheduled_records']] for case in results]
        self.assertTrue(all(r==rosters[0] for r in rosters))

    def test_zero_share_never_routes_text_to_generalist(self):
        result=self.case(0)
        self.assertFalse(any(r['task']=='text' and r['family']=='generalist_2' for r in result['scheduled_records']))
        group=next(g for g in result['family_groups'] if g['family']=='generalist_2' and g['task']=='text')
        self.assertEqual(group['scheduled_requests'],0)
        self.assertEqual(group['unique_inputs'],0)
        self.assertEqual(len(group['missing_inputs']),32)
        self.assertIsNone(group['conditional_scheduled_loss'])

    def test_family_coverage_counts_and_intersection_are_exact(self):
        result=self.case(1/4)
        for group in result['family_groups']:
            rows=[r for r in result['scheduled_records'] if r['task']==group['task'] and r['family']==group['family']]
            counts=Counter(r['pool_id'] for r in rows)
            self.assertEqual(sum(group['per_input_counts'].values()),len(rows))
            self.assertEqual({p:n for p,n in group['per_input_counts'].items() if n},dict(counts))
        a={r['pool_id'] for r in result['scheduled_records'] if r['task']=='text' and r['family']=='text'}
        b={r['pool_id'] for r in result['scheduled_records'] if r['task']=='text' and r['family']=='generalist_2'}
        self.assertEqual(result['text_family_input_intersection'],sorted(a&b))

    def test_conditional_quality_is_not_full_pool_average(self):
        config=copy.deepcopy(self.config)
        initial=self.case(1/4)
        selected={r['pool_id'] for r in initial['scheduled_records'] if r['task']=='text' and r['family']=='generalist_2'}
        self.assertTrue(0<len(selected)<32)
        scores=config['catalog']['quality']['paths']['generalist_2']['text']['samples']
        for pool_id,score in scores.items():score['EM']=int(pool_id in selected);score.pop('correct',None)
        result=self.case(1/4,config)
        group=next(g for g in result['family_groups'] if g['family']=='generalist_2' and g['task']=='text')
        self.assertEqual(group['conditional_scheduled_loss'],0)
        self.assertGreater(group['full_pool_replay_loss'],0)
        self.assertTrue(result['frozen_predict_quality_verified'])
        self.assertAlmostEqual(result['predicted_quality_Q_loss'],result['manual_total_quality_loss'])

    def test_complete_output_no_overwrite_and_no_import_pollution(self):
        previous={k:v for k,v in sys.modules.items() if k=='gsdeploy' or k.startswith('gsdeploy.')}
        with tempfile.TemporaryDirectory() as temp:
            destination=Path(temp)/'new'
            result=exposure.produce(self.config_path,self.samples,[.8],[101],destination)
            self.assertEqual(len(result['cases']),3)
            self.assertEqual(len(list(destination.glob('*-roster.csv'))),3)
            self.assertIsNone(result['U']);self.assertEqual(result['raw_files_read'],0)
            self.assertFalse(result['GPU_started'])
            before=(destination/'report.json').read_bytes()
            with self.assertRaises(FileExistsError):exposure.produce(self.config_path,self.samples,[.8],[101],destination)
            self.assertEqual((destination/'report.json').read_bytes(),before)
            manifest=json.loads((destination/'manifest.json').read_text())
            for filename,expected in manifest['files'].items():self.assertEqual(exposure.receipt(destination/filename)['sha256'],expected)
        self.assertEqual(previous,{k:v for k,v in sys.modules.items() if k=='gsdeploy' or k.startswith('gsdeploy.')})

    def test_explicit_invalid_inputs_rejected(self):
        for value in [0,-1,float('nan'),float('inf')]:
            with self.subTest(value=value),self.assertRaises(ValueError):exposure.build_case(self.config,self.samples,value,101,0,self.api)
        for value in [-1,True,1.5]:
            with self.subTest(value=value),self.assertRaises(ValueError):exposure.build_case(self.config,self.samples,.8,value,0,self.api)
        with self.assertRaises(SystemExit):exposure.main([])


if __name__=='__main__':unittest.main()
