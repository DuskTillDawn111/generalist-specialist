"""Low-cost regression for exclusive progress after the second candidate."""
from pathlib import Path
import json
import tempfile
import unittest
from tools.predictor_validation_20261027 import observable_context_preflight_v43 as v43
from tools.predictor_validation_20261027 import observable_context_preflight_v38 as v38


class SecondCandidateProgressTests(unittest.TestCase):
    def test_second_candidate_has_distinct_durable_receipt_without_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);report=dict(status='RUNNING',cells=[])
            self.assertIs(v43.ObservedForecast,v38.ObservedForecast)
            self.assertIs(v43.save_forecast,v38.save_forecast)
            for cid in v43.CANDIDATES:
                report['cells'].append(dict(candidate_id=cid,status='UNKNOWN'))
                v43.save_candidate_progress(path,cid,report)
            paths=sorted(path.glob('progress-report-*.json'))
            self.assertEqual(len(paths),4)
            for index,cid in enumerate(v43.CANDIDATES):
                stored=json.loads((path/('progress-report-'+cid+'.json')).read_text())
                self.assertEqual([x['candidate_id'] for x in stored['cells']],list(v43.CANDIDATES[:index+1]))
            self.assertFalse((path/'report.json').exists())
            with self.assertRaises(FileExistsError):v43.save_candidate_progress(path,v43.CANDIDATES[1],report)


if __name__=='__main__':unittest.main()
