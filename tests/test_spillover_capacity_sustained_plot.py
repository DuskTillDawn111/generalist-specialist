"""Published-analysis plotting boundaries: no raw data traversal or overwrites."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('sustained_plot',ROOT/'tools/spillover_capacity_20261022/plot_sustained.py')
p=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(p)
FIX=importlib.util.spec_from_file_location('sustained_analysis_fixture',ROOT/'tests/test_spillover_capacity_sustained_analysis.py')
f=importlib.util.module_from_spec(FIX);FIX.loader.exec_module(f)


def publish(path):
    report=f.analyze(f.fixture());report['raw']={'path':'/nonexistent/never-read/run.json','sha256':'not-read'}
    detail=path/'detail.json';detail.write_text(json.dumps(report)+'\n')
    overview=dict(schema='gsdeploy.sustained_capacity_analysis.v1',status='INSUFFICIENT_DIAGNOSTIC_EVIDENCE',all_four_terminal=False,
        jobs=[dict(job_id=report['job_id'],detail_file='detail.json')],paired_primary_repetitions=f.a.repetition_pairs(f.analyze(f.fixture('B')),report),U=None)
    (path/'report.json').write_text(json.dumps(overview)+'\n')
    (path/'manifest.json').write_text(json.dumps(dict(files={name:p.sha(path/name) for name in ['detail.json','report.json']}))+'\n')
    return report


class PublishedPlot(unittest.TestCase):
    def test_only_published_analysis_is_needed_and_roles_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);publish(root);data=p.load_plot_data(root)
            self.assertEqual(data['raw_files_read'],0)
            self.assertEqual(len(data['jobs'][0]['intervals']),672)
            self.assertEqual(len(data['generalist_pairs']),96)
            self.assertEqual(sum(r['role']=='support' for r in data['jobs'][0]['intervals']),384)
            self.assertFalse(data['all_four_terminal']);self.assertIsNone(data['U'])

    def test_tampered_or_unregistered_detail_cannot_be_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);publish(root);(root/'detail.json').write_text('{}\n')
            with self.assertRaises(ValueError):p.load_plot_data(root)
            publish(root);report=json.loads((root/'report.json').read_text());report['jobs'][0]['detail_file']='../run.json'
            (root/'report.json').write_text(json.dumps(report)+'\n');manifest=json.loads((root/'manifest.json').read_text())
            manifest['files']['report.json']=p.sha(root/'report.json');(root/'manifest.json').write_text(json.dumps(manifest)+'\n')
            with self.assertRaises(ValueError):p.load_plot_data(root)

    def test_plot_artifacts_are_new_and_include_data_and_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);analysis=root/'analysis';analysis.mkdir();publish(analysis)
            output=root/'figures';p.render(p.load_plot_data(analysis),output)
            names={q.name for q in output.iterdir()}
            self.assertEqual(names,{'primary-support-timeline.png','primary-support-timeline.pdf',
                'generalist-paired-service-overlap.png','generalist-paired-service-overlap.pdf','plot-data.json','manifest.json'})
            with self.assertRaises(FileExistsError):p.render(p.load_plot_data(analysis),output)


if __name__=='__main__':unittest.main()
