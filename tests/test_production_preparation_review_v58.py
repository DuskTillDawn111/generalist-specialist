"""Small CPU fixtures for independent full-artifact v58 review, no live labels."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import production_preparation_review_v58 as v
from tests import test_projected_handler_campaign_seal_v55_independent_review as fixtures


def document(root, name, value):
    path = root/name; path.write_text(json.dumps(value)); return v.receipt(path)


def configuration(**kwargs):
    value = dict(schema=v.FORECAST_SCHEMA, base_v15_configuration={}, options=copy.deepcopy(v.EXPECTED_OPTIONS),
                 observable_context_model={}, state_SAVE_model={})
    value.update(kwargs)
    return value


class IndependentPreparationReviewTests(unittest.TestCase):
    def test_real_streaming_gzip_full_sha_crc_unselected_duplicate_and_trailing_bytes(self):
        parser, _ = v.parser_runtime('/tmp/v16-preflight-readonly-ijson')
        result, _ = fixtures.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result['irrelevant_tail'] = [{'large':i} for i in range(10000)]
            data = json.dumps(result).encode()
            def summary(payload, *, compressed=None, sha=None, length=None):
                path = root/'fixture.gz'; path.write_bytes(gzip.compress(payload) if compressed is None else compressed)
                return dict(result=dict(artifact=v.receipt(path), complete_lossless_roundtrip_verified=True,
                    uncompressed_sha256=sha or hashlib.sha256(payload).hexdigest(),
                    uncompressed_bytes=len(payload) if length is None else length))
            projected = v.project_gzip(summary(data), parser)
            self.assertEqual(projected, {k:x for k,x in result.items() if k != 'irrelevant_tail'})
            # Invalid data is after every selected field. A partial projection cannot approve it.
            for payload in (data[:-1]+b',"late":{"x":1,"x":2}}', data+b' garbage'):
                with self.assertRaises((ValueError, parser.common.JSONError)):
                    v.project_gzip(summary(payload), parser)
            for kwargs in (dict(sha='0'*64), dict(length=1)):
                with self.assertRaisesRegex(ValueError, 'SHA/length'):
                    v.project_gzip(summary(data, **kwargs), parser)
            corrupted = bytearray(gzip.compress(data)); corrupted[-8] ^= 1
            with self.assertRaises((ValueError, OSError, EOFError)):
                v.project_gzip(summary(data, compressed=bytes(corrupted)), parser)

    def test_original_scorer_and_queue_cell_recomputed_not_flags(self):
        result, args = fixtures.fixture()
        summary, stored, plan, plan_ref, config, rows, candidate, phase_start, quality = args
        summary.update(status=v.COMPLETE)
        contract = dict(forecast_schema=result['schema'], lifecycle_scheduler_version=44)
        values = (summary, config, plan_ref, plan, candidate, rows, quality, phase_start, contract)
        self.assertEqual(v.verify_forecast(result, *values), stored)
        for mutation in ('start', 'bool_start', 'receipt', 'metric', 'cohort', 'route', 'drop', 'startup'):
            changed = copy.deepcopy(result)
            if mutation == 'start': changed['forecast']['starts']['cpu:0'] += 1
            elif mutation == 'bool_start': changed['forecast']['starts']['cpu:0'] = True
            elif mutation == 'receipt': changed['forecast']['finishes']['cpu:0']['finish_ns'] = 151_000_000_000
            elif mutation == 'metric': changed['metrics']['terms']['E'] += 1
            elif mutation == 'cohort': changed['forecast']['cohort'].pop('cpu:0')
            elif mutation == 'route': changed['forecast']['finishes']['cpu:0']['alias'] = 'ocr#0'
            elif mutation == 'drop': changed['static_provider_compilation']['nodes_or_edges_dropped'] = True
            else: changed['startup_prediction']['energy_includes_priming'] = False
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, UnknownEvidence)):
                v.verify_forecast(changed, *values)

    def test_producer_snapshot_complete_closure_and_source_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factory = document(root, 'factory.json', {})
            source = document(root, 'closure.json', {})
            before = v.strict_references([factory, source])
            producer_before = document(root, 'producer-before.json', before)
            producer_after = document(root, 'producer-after.json', before)
            sources = v.strict_references([before, producer_before, producer_after])
            p = dict(forecast_contract=dict(schema='gsdeploy.projected_handler_artifact_contract.v55',
                         factory_source=factory, preflight_source=factory), producer=factory,
                sources=sources, source_before=document(root, 'source-before.json', sources),
                producer_sources_before=producer_before, producer_sources_after=producer_after,
                numeric_preflight=factory, training_index=factory, verified_cells=factory)
            with patch.object(v, 'implementation_sources', return_value=before), patch.object(v, 'collection_source_paths', return_value=[]):
                self.assertEqual(v.source_snapshots(p), (before, sources))
                changed = copy.deepcopy(p); changed['sources'] = [factory]
                with self.assertRaisesRegex(ValueError, 'source-before'):
                    v.source_snapshots(changed)
                incomplete = [factory]
                changed = copy.deepcopy(p)
                changed['producer_sources_before'] = document(root, 'incomplete-before.json', incomplete)
                changed['producer_sources_after'] = document(root, 'incomplete-after.json', incomplete)
                changed['sources'] = v.strict_references([sources,changed['producer_sources_before'],changed['producer_sources_after']])
                changed['source_before'] = document(root, 'changed-source-before.json', changed['sources'])
                with self.assertRaisesRegex(ValueError, 'closure'):
                    v.source_snapshots(changed)
                Path(source['path']).write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'immutable'):
                    v.source_snapshots(p)

    def test_training_role_checked_before_ever_opening_outcome_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_plan = document(root, 'validation-plan.json', dict(split='validation', training_only=False))
            context = document(root, 'context.json', dict(training_only=True))
            forbidden_raw = {'path':str(root/'forbidden-raw.json'), 'sha256':'do-not-open'}
            original = document(root, 'source-manifest.json', dict(captures=[dict(raw=forbidden_raw, plan=bad_plan, context=context)]))
            legacy = dict(index={'path':'not-read','sha256':'not-read'}, artifacts={'source-manifest.json':original})
            captures = {str(i):dict(route=str(i)) for i in range(4)}
            physical = document(root, 'physical.json', dict(schema=v.PHYSICAL_SCHEMA,training_only=True, validation_labels_used=False, captures=captures))
            state = document(root, 'state.json', dict(schema=v.STATE_SCHEMA,training_only=True,validation_labels_used=False,captures=captures))
            p = dict(legacy_training_audit=legacy, supplemental_training_captures=[dict(raw={'path':'unused','sha256':str(i)}) for i in range(4)])
            config = configuration(observable_context_model=physical, state_SAVE_model=state)
            actual = v.read; opened = []
            def guarded(ref):
                opened.append(ref)
                if ref == forbidden_raw: raise AssertionError('validation labels must never be opened')
                return actual(ref)
            with patch.object(v, '_complete_index', return_value=legacy), patch.object(v, 'read', side_effect=guarded):
                with self.assertRaisesRegex(ValueError, 'before raw open'):
                    v.independent_training(p, config, {})
            self.assertNotIn(forbidden_raw, opened)

    def test_exact_v58_configuration_rejects_ambiguous_keys_schema_and_cap(self):
        value = configuration()
        v.validate_configuration(value)
        for field, bad in [('idle_SAVE_model',{}), ('schema','gsdeploy.full_cohort.v59'),
                           ('options',dict(v.EXPECTED_OPTIONS,global_request_cap=2))]:
            changed = copy.deepcopy(value); changed[field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'Exact v58'):
                v.validate_configuration(changed)
        contract = dict(schema='gsdeploy.projected_handler_artifact_contract.v55',
            forecast_schema=v.FORECAST_SCHEMA, preflight_schema=v.PREFLIGHT_SCHEMA,
            lifecycle_scheduler_version=58,
            factory_source=v.receipt(Path(v.__file__).with_name('projected_handler_forecast_v58.py')),
            preflight_source=v.receipt(Path(v.__file__).with_name('projected_handler_preflight_v58.py')))
        v.validate_configuration(value,contract)
        for key in ('forecast_schema','preflight_schema','lifecycle_scheduler_version'):
            changed = copy.deepcopy(contract); changed[key] = 'unauthorized'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'factory/preflight'):
                v.validate_configuration(value,changed)

    def test_state_and_physical_capture_mismatch_or_unqualified_state_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); captures={str(i):dict(route=str(i)) for i in range(4)}
            physical=document(root,'physical.json',dict(schema=v.PHYSICAL_SCHEMA,captures=captures,
                training_only=True,validation_labels_used=False))
            for index,mutation in enumerate(('captures','schema','validation')):
                state=dict(schema=v.STATE_SCHEMA,captures=copy.deepcopy(captures),training_only=True,validation_labels_used=False)
                if mutation=='captures':state['captures']['0']['route']='changed'
                elif mutation=='schema':state['schema']='unauthorized'
                else:state['validation_labels_used']=True
                ref=document(root,'state-'+str(index)+'.json',state)
                with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'same four'):
                    v.training_models(configuration(observable_context_model=physical,state_SAVE_model=ref))

    def test_snapshot_references_must_themselves_be_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=document(root,'source.json',{})
            before=document(root,'before.json',[source]); after=document(root,'after.json',[source])
            values=v.strict_references([source,before,after])
            p=dict(sources=values,source_before=document(root,'sources.json',values),
                producer_sources_before=before,producer_sources_after=after,numeric_preflight=source,
                training_index=source,verified_cells=source)
            # A cryptographically valid reference still cannot be outside the preparation membership.
            for field in ('numeric_preflight','producer_sources_before','producer_sources_after'):
                changed=copy.deepcopy(p); changed[field]=document(root,field+'.json',{})
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'referenced artifact omitted'):
                    v.source_snapshots(changed)

    def test_implementation_review_or_unknown_never_gets_production_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preparation = document(root, 'preparation.json', dict(schema='gsdeploy.artifact_binding_implementation_independent_review.v55',
                                      status='PASS_IMPLEMENTATION_REVIEW_NO_PRODUCTION_PREPARATION_OR_SEAL'))
            with self.assertRaisesRegex(ValueError, 'Actual unsealed'):
                v.review(preparation['path'], 'not-opened', root/'no-output')
            self.assertFalse((root/'no-output').exists())


if __name__ == '__main__': unittest.main()
