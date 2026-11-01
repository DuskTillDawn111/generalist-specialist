import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.historical_evidence_v4 import SCHEMA as HISTORICAL_SCHEMA
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.scoped_training_evidence_v15 import resolve_closure, training_json_read_guard, METADATA_REASON
from tools.predictor_validation_20261027.profiling import digest


class ScopedTrainingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/'old_test.py';self.source.write_text('old source')
        self.old_ref=receipt(self.source);self.source.write_text('changed source')
        self.failed=self.write('failed/raw.json',dict(old_test=self.old_ref))
        self.context=self.write('training/context.json',dict(training_only=True,residents={'holder':'text'},
            execution_sources=[],prior_failed_evidence=[receipt(self.failed)]))
        self.context_ref=receipt(self.context)

    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p

    def run_closure(self,roots=None,**kwargs):
        return resolve_closure(roots or [self.context],training_context_references=[self.context_ref],**kwargs)

    def test_exact_opaque_failed_history_bytes_bound_without_parse(self):
        original=Path.read_text;parsed=[]
        def read(path,*a,**kw):
            parsed.append(str(path));self.assertNotEqual(path,self.failed)
            return original(path,*a,**kw)
        with patch.object(Path,'read_text',read):result=self.run_closure()
        self.assertIn(receipt(self.failed),result['materialized_sources'])
        self.assertNotIn(self.old_ref,result['materialized_sources'])
        b=result['boundaries'][0]
        self.assertEqual(b['ref'],receipt(self.failed));self.assertEqual(b['parent'],self.context_ref)
        self.assertEqual(b['jsonpath'],['prior_failed_evidence','0'])
        self.assertTrue(b['effective_opaque_boundary']);self.assertFalse(b['also_normally_reachable'])
        self.assertFalse(result['full_historical_transitive_closure_verified'])

    def test_arbitrary_model_same_field_does_not_get_boundary(self):
        model=self.write('model.json',dict(training_only=True,prior_failed_evidence=[receipt(self.failed)]))
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
            self.run_closure([model])

    def test_unqualified_context_same_field_does_not_get_boundary(self):
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
            resolve_closure([self.context])

    def test_nested_context_field_is_normal(self):
        context=self.write('other/context.json',dict(training_only=True,residents={},execution_sources=[],
            other=dict(prior_failed_evidence=[receipt(self.failed)])))
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
            resolve_closure([context],training_context_references=[receipt(context)])

    def test_nontraining_context_cannot_qualify(self):
        for mode in ({'training_only':False},{'split':'validation'}):
            value=json.loads(self.context.read_text());value.update(mode)
            self.context.write_text(json.dumps(value));self.context_ref=receipt(self.context)
            with self.assertRaisesRegex(ValueError,'not an explicit actual training context'):self.run_closure()

    def test_opaque_target_byte_drift_is_rejected(self):
        self.failed.write_text('changed failed history')
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):self.run_closure()

    def test_normal_model_edge_upgrades_opaque_target_and_cannot_hide_missing_old_source(self):
        model=self.write('model.json',dict(current_fit=receipt(self.failed)))
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
            self.run_closure([model,self.context])

    def test_direct_root_upgrades_opaque_target(self):
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
            self.run_closure([self.failed,self.context])

    def test_upgraded_normal_target_must_trigger_validation_guard_before_parse(self):
        def guard(ref):
            if ref['path']==str(self.failed):raise ValueError('VALIDATION_LABEL_READ_REFUSED')
        # Opaque branch needs no JSON parse and therefore does not hit guard.
        self.assertTrue(self.run_closure(before_json_read=guard)['boundaries'])
        with self.assertRaisesRegex(ValueError,'VALIDATION_LABEL_READ_REFUSED'):
            self.run_closure([self.failed,self.context],before_json_read=guard)

    def test_normal_first_and_opaque_later_does_not_suppress_validation_guard(self):
        def guard(ref):
            if ref['path']==str(self.failed):raise ValueError('VALIDATION_LABEL_READ_REFUSED')
        with self.assertRaisesRegex(ValueError,'VALIDATION_LABEL_READ_REFUSED'):
            self.run_closure([self.context,self.failed],before_json_read=guard)

    def test_valid_normal_upgrade_expands_and_marks_boundary_inactive(self):
        self.failed.write_text(json.dumps(dict(value=3)))
        value=json.loads(self.context.read_text());value['prior_failed_evidence']=[receipt(self.failed)]
        self.context.write_text(json.dumps(value));self.context_ref=receipt(self.context)
        result=self.run_closure([self.failed,self.context])
        b=result['boundaries'][0]
        self.assertFalse(b['effective_opaque_boundary']);self.assertTrue(b['also_normally_reachable'])
        self.assertTrue(b['normal_JSON_traversal_performed'])

    def test_existing_exact_historical_snapshot_rules_preserved(self):
        snapshot=self.root/'snapshot.py';snapshot.write_text('old source')
        manifest=self.write('manifest.json',dict(schema=HISTORICAL_SCHEMA,historical_artifacts_rewritten=False,
            mappings=[dict(historical=self.old_ref,snapshot=receipt(snapshot))]))
        result=self.run_closure([self.failed,self.context],manifest_path=manifest)
        self.assertIn(receipt(snapshot),result['materialized_sources'])
        self.assertEqual(result['historical_substitutions'],[dict(historical=self.old_ref,snapshot=receipt(snapshot))])

    def test_snapshot_alias_normal_reachability_also_upgrades_target(self):
        historical=self.root/'missing_raw.json';historical.write_bytes(self.failed.read_bytes())
        historical_ref=receipt(historical);historical.write_text('changed old raw')
        value=json.loads(self.context.read_text());value['prior_failed_evidence']=[historical_ref]
        self.context.write_text(json.dumps(value));self.context_ref=receipt(self.context)
        manifest=self.write('manifest.json',dict(schema=HISTORICAL_SCHEMA,historical_artifacts_rewritten=False,
            mappings=[dict(historical=historical_ref,snapshot=receipt(self.failed))]))
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
            self.run_closure([self.failed,self.context],manifest_path=manifest)

    def test_repeated_parent_edges_are_individually_recorded(self):
        value=json.loads(self.context.read_text());value['prior_failed_evidence'].append(receipt(self.failed))
        self.context.write_text(json.dumps(value));self.context_ref=receipt(self.context)
        result=self.run_closure()
        self.assertEqual([r['jsonpath'] for r in result['boundaries']],
            [['prior_failed_evidence','1'],['prior_failed_evidence','0']])
        self.assertEqual(result['effective_opaque_boundary_edges'],2)

    def test_audited_raw_guard_is_also_required_on_normal_upgrade(self):
        guard=training_json_read_guard([])
        self.assertTrue(self.run_closure(before_json_read=guard)['boundaries'])
        with self.assertRaisesRegex(ValueError,'Normal nonindexed raw JSON parse refused'):
            self.run_closure([self.failed,self.context],before_json_read=guard)

    def test_guard_allows_only_exact_audited_training_or_smoke_raw(self):
        self.failed.write_text(json.dumps(dict(value=1)))
        for kwargs in (dict(training_raw_references=[receipt(self.failed)]),
                       dict(training_raw_references=[],inherited_smoke_references=[receipt(self.failed)])):
            guard=training_json_read_guard(**kwargs)
            result=resolve_closure([self.failed],before_json_read=guard)
            self.assertIn(receipt(self.failed),result['normal_JSON_sources'])
        changed=dict(receipt(self.failed),sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'Normal nonindexed raw JSON parse refused'):
            training_json_read_guard([changed])(receipt(self.failed))

    def test_validation_named_outcome_is_refused_even_without_raw_filename(self):
        validation=self.write('validation/labels.json',dict(labels=['heldout']))
        with self.assertRaisesRegex(ValueError,'Validation outcome JSON parse refused'):
            resolve_closure([validation],before_json_read=training_json_read_guard([]))

    def test_final_reverification_catches_drift_after_cached_opaque_resolution(self):
        later=self.write('later.json',dict(value=1))
        def guard(ref):
            if ref['path']==str(later):self.failed.write_text('changed during traversal')
        with self.assertRaisesRegex(ValueError,'changed during traversal'):
            self.run_closure([later,self.context],before_json_read=guard)

    def metadata_fixture(self):
        inventory=self.write('prep/source-before.json',dict(sources=[self.old_ref]))
        parent=self.write('prep/prepared-repeat.json',dict(schema='gsdeploy.corun_exact_repeat_preparation.v8',
            status='CPU_PREPARED_SOURCE_RESOURCE_AUDIT_PASS',training_only=True,source_before=receipt(inventory)))
        policy=dict(schema='gsdeploy.explicit_operational_metadata_boundary_policy.v15',implementation=receipt(__file__),
            boundaries=[dict(parent=receipt(parent),jsonpath=['source_before'],ref=receipt(inventory),reason=METADATA_REASON)])
        policy['policy_sha256']=digest(policy)
        return parent,inventory,policy

    def test_metadata_inventory_edge_binds_bytes_but_does_not_recurse(self):
        parent,inventory,policy=self.metadata_fixture()
        result=resolve_closure([parent],metadata_boundary_policy=policy)
        self.assertIn(receipt(inventory),result['materialized_sources'])
        self.assertEqual(result['boundaries'][0]['reason'],METADATA_REASON)
        self.assertEqual(result['metadata_boundary_policy'],policy)

    def test_metadata_same_field_in_other_parent_does_not_qualify(self):
        parent,inventory,policy=self.metadata_fixture()
        other=self.write('other/prepared-repeat.json',json.loads(parent.read_text()))
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
            resolve_closure([other],metadata_boundary_policy=policy)

    def test_metadata_exact_parent_byte_change_does_not_qualify(self):
        parent,inventory,policy=self.metadata_fixture()
        value=json.loads(parent.read_text());value['extra']=True;parent.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
            resolve_closure([parent],metadata_boundary_policy=policy)

    def test_metadata_sibling_dependency_is_still_normal(self):
        parent,inventory,policy=self.metadata_fixture()
        value=json.loads(parent.read_text());value['runs']=dict(actual_fit=self.old_ref);parent.write_text(json.dumps(value))
        policy['boundaries'][0]['parent']=receipt(parent)
        policy['policy_sha256']=digest({k:v for k,v in policy.items() if k!='policy_sha256'})
        with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
            resolve_closure([parent],metadata_boundary_policy=policy)

    def test_metadata_normal_other_path_upgrades_inventory_and_missing_source_rejects(self):
        parent,inventory,policy=self.metadata_fixture()
        for roots in ([inventory,parent],[parent,inventory]):
            with self.assertRaisesRegex(ValueError,'without an exact historical snapshot'):
                resolve_closure(roots,metadata_boundary_policy=policy)

    def test_metadata_normal_upgrade_triggers_guard_before_inventory_parse(self):
        parent,inventory,policy=self.metadata_fixture()
        def guard(ref):
            if ref['path']==str(inventory):raise ValueError('INVENTORY_PARSE_REFUSED')
        resolve_closure([parent],metadata_boundary_policy=policy,before_json_read=guard)
        for roots in ([inventory,parent],[parent,inventory]):
            with self.assertRaisesRegex(ValueError,'INVENTORY_PARSE_REFUSED'):
                resolve_closure(roots,metadata_boundary_policy=policy,before_json_read=guard)

    def test_metadata_inventory_drift_after_resolution_rejected_by_final_hash(self):
        parent,inventory,policy=self.metadata_fixture();later=self.write('later.json',dict(value=2))
        def guard(ref):
            if ref['path']==str(later):inventory.write_text('drifted inventory')
        with self.assertRaisesRegex(ValueError,'changed during traversal'):
            resolve_closure([later,parent],metadata_boundary_policy=policy,before_json_read=guard)

    def test_metadata_policy_seal_or_implementation_drift_rejected(self):
        parent,inventory,policy=self.metadata_fixture();policy['boundaries'][0]['ref']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Source-bound exact'):
            resolve_closure([parent],metadata_boundary_policy=policy)

    def renamed_historical_fixture(self,filename='raw.json'):
        snapshot=self.write('renamed-snapshot.json',dict(value='historical outcome'))
        original=self.root/'unindexed'/filename;original.parent.mkdir(exist_ok=True)
        original.write_bytes(snapshot.read_bytes());old=receipt(original);original.write_text('now changed')
        manifest=self.write('renamed-manifest.json',dict(schema=HISTORICAL_SCHEMA,historical_artifacts_rewritten=False,
            mappings=[dict(historical=old,snapshot=receipt(snapshot))]))
        return old,snapshot,manifest

    def test_normal_unindexed_raw_cannot_hide_behind_renamed_snapshot(self):
        old,snapshot,manifest=self.renamed_historical_fixture()
        model=self.write('model.json',dict(source=old))
        with self.assertRaisesRegex(ValueError,'Normal nonindexed raw JSON parse refused'):
            resolve_closure([model],manifest,before_json_read=training_json_read_guard([]))

    def test_normal_observed_document_cannot_hide_behind_renamed_snapshot(self):
        old,snapshot,manifest=self.renamed_historical_fixture('observed.json')
        model=self.write('model.json',dict(source=old))
        with self.assertRaisesRegex(ValueError,'Validation outcome JSON parse refused'):
            resolve_closure([model],manifest,before_json_read=training_json_read_guard([]))

    def test_already_parsed_snapshot_cannot_suppress_original_identity_guard(self):
        old,snapshot,manifest=self.renamed_historical_fixture()
        for sources in ([old,receipt(snapshot)],[receipt(snapshot),old]):
            model=self.write('model.json',dict(sources=sources))
            with self.assertRaisesRegex(ValueError,'Normal nonindexed raw JSON parse refused'):
                resolve_closure([model],manifest,before_json_read=training_json_read_guard([]))

    def test_manifest_snapshot_alias_never_parsed_before_historical_guard(self):
        for filename,message in (('raw.json','Normal nonindexed raw'),('observed.json','Validation outcome')):
            old,snapshot,manifest=self.renamed_historical_fixture(filename)
            model=self.write('model.json',dict(source=old));original_read=Path.read_text;opened=[]
            def spy(path,*args,**kwargs):
                opened.append(str(path));return original_read(path,*args,**kwargs)
            with patch.object(Path,'read_text',spy):
                with self.assertRaisesRegex(ValueError,message):
                    resolve_closure([model],manifest,before_json_read=training_json_read_guard([]))
            self.assertNotIn(str(snapshot),opened)

    def test_transitive_manifest_snapshot_alias_guard_runs_before_any_sensitive_parse(self):
        old,snapshot,manifest=self.renamed_historical_fixture()
        final=self.write('final-alias.json',json.loads(snapshot.read_text()))
        value=json.loads(manifest.read_text());value['mappings'].append(dict(historical=receipt(snapshot),snapshot=receipt(final)))
        manifest.write_text(json.dumps(value));model=self.write('model.json',dict(source=old))
        original_read=Path.read_text;opened=[]
        def spy(path,*args,**kwargs):
            opened.append(str(path));return original_read(path,*args,**kwargs)
        with patch.object(Path,'read_text',spy):
            with self.assertRaisesRegex(ValueError,'Normal nonindexed raw'):
                resolve_closure([model],manifest,before_json_read=training_json_read_guard([]))
        self.assertNotIn(str(snapshot),opened);self.assertNotIn(str(final),opened)


if __name__=='__main__':unittest.main()
