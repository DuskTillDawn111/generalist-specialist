"""Entry gates and declaration-tamper checks; native rebuilding is separately audited."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import unified_kernel_dag_intake_reauthentication_v1 as m
from tools.predictor_validation_20261027 import kernel_dag_repeated_instance_alignment_v1 as adapter
from tools.predictor_validation_20261027.native_collect import receipt


class IntakeReauthenticationTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root)
        def write(name,value):
            path=root/name;path.write_text(json.dumps(value));return receipt(path)
        native=write('strict.json',dict(training_only=True,validation_started=False))
        device=write('device.json',dict(hardware='synthetic-only'))
        occupancy=write('occupancy.json',dict(static='synthetic-only'))
        catalog=dict(training_only=True,validation_labels_read=False,profiles=[dict(nodes=[dict(id='k')],
            edges=[dict(before='k',after='end',kind='native-edge')])],sources=[native,device,occupancy])
        capture=dict(split='training',validation_labels_read=False,kernels=[dict(node_id='k',start_ns=1,end_ns=3)])
        paths=dict(observations=[dict(node_id='k',path=[dict(left_ns=1,right_ns=3)])])
        artifacts={name:write(name,value) for name,value in [('catalog.json',catalog),('capture.json',capture),('paths.json',paths)]}
        report=dict(schema=m.native_intake.SCHEMA,validation_labels_read=False,
            status='COMPLETE_NATIVE_MAPPING_PATH_DIAGNOSTICS_NO_FIT',review=native,
            sources=[native,device,occupancy],artifacts=artifacts,native_mapping_authenticated=True,
            common_native_session_clock=True,rejected_kernel_count=0,full_session_inventory=dict(outside_measured_overlap_wall_ns=0))
        entry=dict(intake_report=write('report.json',report),device_attributes=device,occupancy_audit=occupancy)
        return entry,native,report,dict(catalog=catalog,capture=capture,paths=paths)

    def rebuild(self,report,truth):
        def execute(review,device,output,*,occupancy_audit):
            output=Path(output);output.mkdir()
            rebuilt=copy.deepcopy(report);rebuilt['artifacts']={}
            for name,key in [('catalog.json','catalog'),('capture.json','capture'),('paths.json','paths')]:
                target=output/name;target.write_text(json.dumps(truth[key]));rebuilt['artifacts'][name]=receipt(target)
            (output/'report.json').write_text(json.dumps(rebuilt))
            return rebuilt
        return execute

    def test_bad_roles_or_missing_input_pins_fail_before_native_sql_work(self):
        for mode in ('report-role','native-review','device-pin','occupancy-pin','catalog-role','capture-role'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                entry,native,report,truth=self.fixture(root)
                if mode=='report-role':report.pop('validation_labels_read')
                if mode=='native-review':report['review']={'path':'foreign','sha256':'0'*64}
                if mode=='device-pin':entry.pop('device_attributes')
                if mode=='occupancy-pin':entry.pop('occupancy_audit')
                if mode in ('catalog-role','capture-role'):
                    name='catalog.json' if mode=='catalog-role' else 'capture.json'
                    target=Path(report['artifacts'][name]['path']);value=json.loads(target.read_text())
                    value['validation_labels_read']=True;target.write_text(json.dumps(value));report['artifacts'][name]=receipt(target)
                Path(entry['intake_report']['path']).write_text(json.dumps(report));entry['intake_report']=receipt(entry['intake_report']['path'])
                with patch.object(m.native_intake,'intake',side_effect=AssertionError('native labels touched before role/pin gates')) as native_work:
                    with self.assertRaises(ValueError):m.authenticate(entry,native)
                    native_work.assert_not_called()

    def test_output_directory_changes_are_ignored_and_complete_inputs_contents_are_equal(self):
        with tempfile.TemporaryDirectory() as root:
            entry,native,report,truth=self.fixture(root)
            with patch.object(m.native_intake,'intake',side_effect=self.rebuild(report,truth)) as native_work:
                result=m.authenticate(entry,native)
            native_work.assert_called_once()
            self.assertEqual(result['catalog'],truth['catalog']);self.assertEqual(result['capture'],truth['capture'])
            args,kwargs=native_work.call_args
            self.assertEqual(args[:2],(native['path'],entry['device_attributes']['path']))
            self.assertEqual(kwargs,dict(occupancy_audit=entry['occupancy_audit']['path']))
            self.assertFalse(Path(args[2]).exists())

    def test_rehashed_dag_timing_paths_or_input_sources_do_not_override_native_reconstruction(self):
        for mode in ('edge','node','timing','paths','sources'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                entry,native,report,truth=self.fixture(root)
                if mode=='sources':
                    extra=Path(root)/'invented-source';extra.write_text('extra');report['sources'].append(receipt(extra))
                else:
                    name='catalog.json' if mode in ('edge','node') else 'capture.json' if mode=='timing' else 'paths.json'
                    target=Path(report['artifacts'][name]['path']);value=json.loads(target.read_text())
                    if mode=='edge':value['profiles'][0]['edges'][0]['kind']='caller-rehashed-fake-dependency'
                    if mode=='node':value['profiles'][0]['nodes'][0]['id']='caller-rehashed-fake-node'
                    if mode=='timing':value['kernels'][0]['end_ns']=4
                    if mode=='paths':value['observations'][0]['path'][0]['right_ns']=4
                    target.write_text(json.dumps(value));report['artifacts'][name]=receipt(target)
                target=Path(entry['intake_report']['path']);target.write_text(json.dumps(report));entry['intake_report']=receipt(target)
                # The source-of-truth rebuild keeps the original exact source set.
                original=copy.deepcopy(report);original['sources']=truth['catalog']['sources']
                with patch.object(m.native_intake,'intake',side_effect=self.rebuild(original,truth)):
                    with self.assertRaises(ValueError):m.authenticate(entry,native)

    def test_repeated_loader_rebuilds_native_intake_before_using_published_profiles(self):
        review=dict(schema=adapter.PRODUCER_SCHEMA+'.review',status='PASS_CPU_MATRIX_NATIVE_BINDING_ONLY',
            training_only=True,validation_labels_read=False,campaign={'path':'campaign'},capture_id='capture',
            native_review={'path':'native','sha256':'a'*64})
        entry=dict(producer_review={'path':'review'},intake_report={'path':'forged'})
        with patch.object(adapter,'read',return_value=review) as read,\
             patch.object(adapter.producer,'authenticate_review',return_value=review),\
             patch.object(adapter,'reauthenticate_intake',side_effect=ValueError('fresh SQLite DAG differs')) as rebuild:
            with self.assertRaisesRegex(ValueError,'fresh SQLite DAG differs'):adapter.load_member(entry)
            rebuild.assert_called_once_with(entry,review['native_review']);self.assertEqual(read.call_count,1)


if __name__=='__main__':unittest.main()
