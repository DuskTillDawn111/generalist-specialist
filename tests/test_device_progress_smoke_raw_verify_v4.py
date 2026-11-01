"""CPU descriptor identity isolation and unchanged frozen probe semantics."""
import json
from pathlib import Path
import tempfile
import unittest

from tests import test_device_progress_smoke_raw_verify_v3 as oldtests
from tools.predictor_validation_20261027 import device_progress_smoke_raw_verify_v3 as old
from tools.predictor_validation_20261027 import device_progress_smoke_raw_verify_v4 as v
from tools.predictor_validation_20261027.device_progress_review_contract_v4 import V6,qualify_roles


def packet(directory):
    raw,plan,context=oldtests.packet(directory);context.update(protocol=V6.protocol_prefix+'natural',supervisor=V6.supervisor_context,monitoring_scope=V6.monitoring_scope)
    context['execution_sources']=[m for m in context['execution_sources'] if Path(m['original_path']).name in old.FROZEN]
    for name,sha in V6.pipeline:
        actual=Path(v.__file__).with_name(name).resolve();frozen=Path(directory)/('snapshot-'+name);frozen.write_bytes(actual.read_bytes())
        context['execution_sources'].append(dict(original_path=str(actual),frozen=old.receipt(frozen)))
    context['sources']=[m['frozen'] for m in context['execution_sources']];plan.update(schema=V6.campaign_schema,execution_context_sha256=old.context_identity(context));raw['execution_context_sha256']=plan['execution_context_sha256']
    oldtests.write_roles(directory,raw,plan,context);return raw,plan,context


class DescriptorRoleTests(unittest.TestCase):
    def test_exact_v6_roles_and_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            raw,_,_=packet(d);role=qualify_roles(raw);self.assertEqual(role['plan_schema'],V6.campaign_schema);self.assertEqual(len(role['exact_pipeline_sources']),10)
    def test_v5_cannot_borrow_v6_or_reverse(self):
        with tempfile.TemporaryDirectory() as d:
            raw,_,_=oldtests.packet(d)
            with self.assertRaises(ValueError):qualify_roles(raw)
            raw,_,_=packet(d)
            with self.assertRaises(ValueError):old.qualify_roles(raw)
    def test_exact_context_descriptor_and_source_receipts_required(self):
        for mutation in ('protocol','supervisor','scope','source'):
            with tempfile.TemporaryDirectory() as d:
                raw,plan,context=packet(d)
                if mutation=='source':context['execution_sources'][-1]['frozen']['sha256']='0'*64
                else:context[dict(protocol='protocol',supervisor='supervisor',scope='monitoring_scope')[mutation]]='old'
                oldtests.write_roles(d,raw,plan,context)
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):qualify_roles(raw)
    def test_probe_algorithm_is_exact_frozen_v3_function(self):
        self.assertIs(v.core.validate_raw,old.validate_raw);self.assertIs(v.core.reparse_outputs,old.reparse_outputs)


if __name__=='__main__':unittest.main()
