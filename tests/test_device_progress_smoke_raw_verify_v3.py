"""CPU only: exact v5/v2 identities cannot borrow the old v4 contract."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tests import test_device_progress_smoke_raw_verify_v1 as base
from tools.predictor_validation_20261027 import device_progress_smoke_raw_verify_v2 as old
from tools.predictor_validation_20261027 import device_progress_smoke_raw_verify_v3 as v


def packet(directory,mode='natural',split='smoke'):
    raw,plan,context=base.RawVerifierTests().packet(directory,mode=mode,split=split)
    context.update(protocol='paired_profile_v5_'+mode,supervisor='profile_training_supervisor_v2_outer_lease_only',execution_sources=[])
    for name in {**v.FROZEN,**v.PIPELINE_FROZEN}:
        original=Path(v.__file__).with_name(name).resolve();frozen=Path(directory)/('frozen-'+name);frozen.write_bytes(original.read_bytes())
        context['execution_sources'].append(dict(original_path=str(original),frozen=v.receipt(frozen)))
    context['sources']=[member['frozen'] for member in context['execution_sources']]
    plan.update(schema='gsdeploy.paired_profile_campaign.v5',execution_context_sha256=v.context_identity(context))
    raw['execution_context_sha256']=plan['execution_context_sha256'];write_roles(directory,raw,plan,context)
    return raw,plan,context


def write_roles(directory,raw,plan,context):
    for name,value in [('plan',plan),('context',context)]:
        path=Path(directory)/(name+'.json');path.write_text(json.dumps(value));raw[name]=v.receipt(path)


class V3RoleTests(unittest.TestCase):
    def test_exact_v5_roles_for_both_modes_and_nonvalidation_splits(self):
        for mode,split in [('natural','smoke'),('kernel','smoke'),('natural','training'),('kernel','training')]:
            with tempfile.TemporaryDirectory() as directory:
                raw,_,_=packet(directory,mode,split);role=v.qualify_roles(raw)
                self.assertEqual(role['plan_schema'],'gsdeploy.paired_profile_campaign.v5')
                self.assertEqual(len(role['exact_pipeline_sources']),10)
    def test_v3_rejects_v4_and_v2_rejects_v5(self):
        with tempfile.TemporaryDirectory() as directory:
            raw,_,_=base.RawVerifierTests().packet(directory)
            with self.assertRaisesRegex(ValueError,'paired-v5'):v.qualify_roles(raw)
            raw,_,_=packet(directory)
            with self.assertRaisesRegex(ValueError,'paired-v4'):old.qualify_roles(raw)
    def test_protocol_supervisor_and_mode_cannot_borrow_old_identity(self):
        for key,value in [('protocol','paired_profile_v4_natural'),('supervisor','profile_training_supervisor_v1_outer_lease_only'),('paired_profile_mode','kernel')]:
            with tempfile.TemporaryDirectory() as directory:
                raw,plan,context=packet(directory);context[key]=value;write_roles(directory,raw,plan,context)
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'v5 context'):v.qualify_roles(raw)
    def test_missing_or_replaced_pipeline_source_is_rejected(self):
        for name in v.PIPELINE_FROZEN:
            for mutation in ('missing','wrong_SHA','old_path','unreceipted_bytes'):
                with tempfile.TemporaryDirectory() as directory:
                    raw,plan,context=packet(directory);member=next(x for x in context['execution_sources'] if x['original_path'].endswith('/'+name))
                    if mutation=='missing':context['execution_sources'].remove(member)
                    elif mutation=='wrong_SHA':member['frozen']['sha256']='0'*64
                    elif mutation=='old_path':member['original_path']=member['original_path'].replace('_v5.py','_v4.py').replace('_v2.py','_v1.py')
                    else:Path(member['frozen']['path']).write_text('changed')
                    context['sources']=[x['frozen'] for x in context['execution_sources']];write_roles(directory,raw,plan,context)
                    with self.subTest(name=name,mutation=mutation),self.assertRaises(ValueError):v.qualify_roles(raw)
    def test_duplicate_source_and_context_identity_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw,plan,context=packet(directory);context['execution_sources'].append(copy.deepcopy(context['execution_sources'][0]));context['sources']=[x['frozen'] for x in context['execution_sources']];write_roles(directory,raw,plan,context)
            with self.assertRaisesRegex(ValueError,'Duplicate'):v.qualify_roles(raw)
        with tempfile.TemporaryDirectory() as directory:
            raw,plan,context=packet(directory);context['extra']='changed';write_roles(directory,raw,plan,context)
            with self.assertRaisesRegex(ValueError,'context identity'):v.qualify_roles(raw)
    def test_validation_rejected_before_response_body(self):
        class Body(dict):
            def get(self,*args):raise AssertionError('Response body touched')
        with tempfile.TemporaryDirectory() as directory:
            raw,plan,context=packet(directory);plan['split']='validation';raw['request_records'][0]['worker_response']=Body();write_roles(directory,raw,plan,context)
            with self.assertRaisesRegex(ValueError,'non-validation'):v.qualify_roles(raw)
    def test_probe_and_noanswer_scope_unchanged(self):
        raw=base.fixture();self.assertEqual(v.validate_raw(raw),old.validate_raw(raw))
        row=raw['request_records'][0];row['task']='text';row['worker_response'].update(raw_output='<No Answer>',prediction='')
        self.assertTrue(v.valid_prediction(row,v.BASE_NO_ANSWER));self.assertFalse(v.valid_prediction(row,None))
        row['worker_response']['raw_output']=' ';self.assertFalse(v.valid_prediction(row,v.BASE_NO_ANSWER))


if __name__=='__main__':unittest.main()
