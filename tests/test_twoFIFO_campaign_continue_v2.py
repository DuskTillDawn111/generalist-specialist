import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_campaign_continue_v2 as m


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class ContinueTests(unittest.TestCase):
    def setup_campaign(self,root):
        attempts=[]
        for i in range(12):
            folder=root/('attempt-'+str(i));folder.mkdir();plan=folder/'plan.json'
            write(plan,dict(attempt_id='attempt-'+str(i),execution_id='execution-'+str(i)))
            attempts.append(dict(plan=m.receipt(plan),attempt_id='attempt-'+str(i),execution_id='execution-'+str(i),
                collection=str(folder/'collection'),existing_review=str(folder/'native-review01/report.json'),existing_macros=str(folder/'macros-v2.json')))
        manifest=dict(attempts=attempts,campaign={'path':str(root/'campaign.json')});path=root/'manifest.json';write(path,manifest)
        return path,manifest

    def execution(self,item,status='CHILD_EXITED_REQUIRES_PROFILE_REVIEW'):
        return dict(profile_execution=dict(schema=m.collector.SCHEMA+'.execution_binding',plan=item['plan'],
            execution_id=item['execution_id'],attempt_id=item['attempt_id']),budget=m.collector.supervisor.BUDGET,
            status=status,child_reaped=True,live_group_pids=[],experiment_lock_held_through_group_exit=True,
            all_owned_descendants_exited_and_reaped=True,final_source_audit='PASS',watchdog=dict(status='EXITED',return_code=0))

    def test_existing_unfinished_attempt_is_readonly_waiting_never_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest=self.setup_campaign(root);collection=Path(manifest['attempts'][0]['collection']);collection.mkdir()
            marker=collection/'raw.json';marker.write_text('unfinished bytes')
            with patch.object(m,'audit',return_value=manifest),patch.object(m.collector,'execute') as execute,patch.object(m,'review') as review:
                result=m.run(path,root/'run',execute=True)
            self.assertEqual(result['status'],'WAITING_EXISTING_ATTEMPT_FINAL_RECEIPT');execute.assert_not_called();review.assert_not_called()
            self.assertEqual(marker.read_text(),'unfinished bytes')

    def test_without_execute_never_launches_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest=self.setup_campaign(root)
            with patch.object(m,'audit',return_value=manifest),patch.object(m.collector,'execute') as execute:
                result=m.run(path,root/'run')
            self.assertEqual(result['status'],'READY_EXPLICIT_EXECUTE_REQUIRED');execute.assert_not_called()

    def test_failure_or_disk_preflight_stops_without_retry(self):
        for which in ('failure','disk','source'):
            with self.subTest(which=which),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path,manifest=self.setup_campaign(root)
                if which=='failure':
                    item=manifest['attempts'][0];write(Path(item['collection'])/'execution.json',self.execution(item,'FAILED_NO_RETRY'))
                audit_effect=[manifest,ValueError('source drift')] if which=='source' else None
                disk=type('Disk',(),dict(free=0))()
                with patch.object(m,'audit',side_effect=audit_effect,return_value=manifest),patch.object(m.shutil,'disk_usage',return_value=disk),patch.object(m.collector,'execute') as execute:
                    result=m.run(path,root/'run',execute=True)
                self.assertEqual(result['status'],'STOPPED_NO_RETRY');execute.assert_not_called()

    def test_complete_attempts_are_reviewed_before_next_launch_in_fixed_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest=self.setup_campaign(root);calls=[]
            def execute(plan_path,output):
                item=next(a for a in manifest['attempts'] if a['plan']['path']==plan_path)
                calls.append(('execute',item['attempt_id']));write(Path(output)/'execution.json',self.execution(item))
            def review(raw_path,output,execution):
                attempt=Path(raw_path).parent.parent.name;calls.append(('review',attempt))
                raw=Path(raw_path);write(raw,{})
                ref=Path(output)/'report.json';write(ref,dict(status=m.COMPLETE,training_feature_extraction_eligible=True,issues=[],raw=m.receipt(raw)));return m.receipt(ref)
            def extract(review_path):
                calls.append(('macro',Path(review_path).parent.parent.name));return dict(requests=[],macros=0)
            with patch.object(m,'audit',return_value=manifest),patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),dict(free=100*2**30))()),patch.object(m.collector,'execute',side_effect=execute),patch.object(m,'review',side_effect=review),patch.object(m.macros,'extract',side_effect=extract),patch.object(m,'observed_resources',return_value=dict(startup=dict(energy_joules=1.),cases=[dict(measurement_energy_joules=1.)])):
                result=m.run(path,root/'run',execute=True)
            self.assertEqual(result['status'],'COMPLETE_ALL_TWELVE_NATIVE_ATTEMPTS_REVIEWED')
            self.assertEqual(result['execute_invocations'],12)
            self.assertEqual([kind for kind,attempt in calls],['execute','review','macro']*12)
            self.assertEqual([attempt for kind,attempt in calls if kind=='execute'],[a['attempt_id'] for a in manifest['attempts']])
            with patch.object(m,'audit',return_value=manifest):
                with self.assertRaises(FileExistsError):m.run(path,root/'run',execute=True)

    def test_not_eligible_review_blocks_every_following_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest=self.setup_campaign(root);item=manifest['attempts'][0]
            write(Path(item['collection'])/'execution.json',self.execution(item))
            report=root/'ineligible.json';write(report,dict(status='NOT_ELIGIBLE',training_feature_extraction_eligible=False,issues=['failed']))
            with patch.object(m,'audit',return_value=manifest),patch.object(m.collector,'execute') as execute,patch.object(m,'review',return_value=m.receipt(report)),patch.object(m.macros,'extract') as extract:
                result=m.run(path,root/'run',execute=True)
            self.assertEqual(result['status'],'STOPPED_NO_RETRY');execute.assert_not_called();extract.assert_not_called()

    def test_unobservable_measurement_or_startup_stops_with_facts_and_keeps_remaining_attempts_unstarted(self):
        for unknown in ('startup','measurement'):
            with self.subTest(unknown=unknown),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path,manifest=self.setup_campaign(root);item=manifest['attempts'][0]
                collection=Path(item['collection']);write(collection/'execution.json',self.execution(item))
                write(collection/'raw.json',{})
                report=root/'eligible.json';write(report,dict(status=m.COMPLETE,
                    training_feature_extraction_eligible=True,issues=[],raw=m.receipt(collection/'raw.json')))
                facts=dict(startup=dict(energy_joules=None if unknown=='startup' else 1.),
                    cases=[dict(measurement_energy_joules=None if unknown=='measurement' else 1.)])
                with patch.object(m,'audit',return_value=manifest),patch.object(m.collector,'execute') as execute,patch.object(m,'review',return_value=m.receipt(report)),patch.object(m.macros,'extract') as extract,patch.object(m,'observed_resources',return_value=facts):
                    result=m.run(path,root/'run',execute=True)
                self.assertEqual(result['status'],'STOPPED_NO_RETRY')
                self.assertIn('power-gap gate',result['error'])
                execute.assert_not_called();extract.assert_not_called()
                saved=root/'run'/item['attempt_id']/'observed-resources.json'
                self.assertEqual(json.loads(saved.read_text()),facts)
                self.assertTrue(all(not Path(a['collection']).exists() for a in manifest['attempts'][1:]))


if __name__=='__main__':unittest.main()
