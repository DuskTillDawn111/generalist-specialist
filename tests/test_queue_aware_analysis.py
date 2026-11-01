import json
from pathlib import Path
import tempfile
import unittest

from tools.queue_aware_20261023.analysis import analyze


class AnalysisTests(unittest.TestCase):
    def raw(self,root,strategy,seed,cost,eligible=True):
        contract=dict(deployment={'text':2},strategy=strategy)
        utility={t+'_norm':cost for t in 'QTLPME'}
        utility.update(U=-cost if eligible else None,eligible=eligible,utility_complete=True)
        utility.update({k:True for k in ['quality_identity_verified','cohort_verified','constraints_verified','memory_verified','energy_verified']})
        raw=dict(policy_contract=contract,transport_config=dict(random_seed=seed),
            evaluation=dict(utility=utility),request_records=[],measurement_window=dict(start_time_ns=0),
            replica_to_family={'text#0':'text','text#1':'text'},config=dict(weights={t:1/6 for t in 'QTLPME'}),
            status='completed',errors=[])
        path=root/f'{strategy}-{seed}.json';path.write_text(json.dumps(raw));return path

    def test_missing_seed_never_gets_partial_mean_or_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths=[self.raw(root,'steady',1,2),self.raw(root,'steady',2,2),self.raw(root,'partial',1,.1)]
            result=analyze(paths,[1,2],root/'result')
            self.assertEqual(result['winner']['candidate']['strategy'],'steady')
            self.assertFalse(result['full_declared_table_complete'])
            self.assertIsNone(result['ranking'][1]['mean_U'])
            self.assertEqual(result['comparisons'][0]['descriptive_whole_pair_summary']['n_whole_run_pairs'],1)
            self.assertFalse(result['comparisons'][0]['all_declared_pairs_eligible'])

    def test_ineligible_outcomes_retained_and_arithmetic_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.raw(root,'bad',1,.1,False)
            result=analyze([path],[1],root/'result')
            self.assertIsNone(result['winner'])
            raw=json.loads(path.read_text());raw['evaluation']['utility'].update(eligible=True,U=-99)
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'arithmetic'):
                analyze([path],[1],root/'bad-result')


if __name__=='__main__':unittest.main()
