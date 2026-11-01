"""CPU-only strict v55 binding and inherited supervisor resource regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import main_prediction_gate_supervisor_v2 as m
from tools.predictor_validation_20261027 import known_input_gate_v5 as gate
from tools.predictor_validation_20261027.implementation_source_closure_v27 import strict_references
from gsdeploy.resources import identity
from tests import test_main_prediction_gate_supervisor_v1 as original
from tests import test_main_prediction_gate_supervisor_v1_independent_review as independent


class InheritedBudgetAndCleanupTests(unittest.TestCase):
    def setUp(self):
        self.patch=patch.object(original,'m',m);self.patch.start();self.addCleanup(self.patch.stop)

    test_complete_all_four_and_any_unknown_refused=original.SupervisorTests.test_complete_all_four_and_any_unknown_refused
    test_actual_owned_group_timeout_and_disk_margin_cleanup=original.SupervisorTests.test_actual_owned_group_timeout_and_disk_margin_cleanup
    test_manifest_budget_and_low_disk_reject_before_launch=original.SupervisorTests.test_manifest_budget_and_low_disk_reject_before_launch
    test_nonzero_child_preserved_and_source_drift_cannot_pass=original.SupervisorTests.test_nonzero_child_preserved_and_source_drift_cannot_pass


class StrictBindingTests(unittest.TestCase):
    def fixture(self,root,*,unbound_review=False,changed_threshold=False,changed_cells=False):
        plan_path=m.ROOT/'artifacts/predictor-gpu-20261027/known-input-availability-v11-20261028-prepared01/plan.json'
        plan=m.read(plan_path);context=m.checked(plan['context'])
        catalog=next(s['frozen']['path'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl'))
        model=root/'model.json';m.save(model,dict(synthetic_CPU_only=True))
        original_cell=m.read(m.ROOT/'artifacts/predictor-projected-handler-v44-20261028-preflight01/shared_balanced-unsealed-cell.json')
        cells=[]
        for candidate in plan['candidates']:
            cell=copy.deepcopy(original_cell);cell.update(candidate_id=candidate['id'],execution_context_sha256=candidate['execution_context_sha256']);cells.append(cell)
        m.save(root/'verified-cells.json',dict(cells=cells))
        train=m.receipt(m.ROOT/'artifacts/predictor-complete-training-index-v12-20261028-built02/training-index.json')
        prepared_sources=strict_references([m.implementation_sources(m.__file__),m.receipt(plan_path),m.receipt(model),m.receipt(catalog)])
        m.save(root/'source-before.json',prepared_sources)
        m.save(root/'preparation.json',dict(schema=m.CAMPAIGN_SCHEMA,status='CPU_PREPARED_REQUIRES_INDEPENDENT_ARTIFACT_REVIEW_NO_PROTOCOL_OR_SEAL',
            plan=m.receipt(plan_path),all_four_results_independently_reprojected=True,verified_cells=m.receipt(root/'verified-cells.json'),training_index=train,
            source_before=m.receipt(root/'source-before.json'),sources=prepared_sources,configuration=m.receipt(model)))
        prep=m.receipt(root/'preparation.json')
        m.save(root/'review.json',dict(schema=m.REVIEW_SCHEMA,status='PASS',issues=[],preparation=prep,complete_four_forecasts_reprojected=True));review=m.receipt(root/'review.json')
        refs=strict_references([prepared_sources,m.receipt(root/'verified-cells.json'),train,m.receipt(root/'source-before.json'),[] if unbound_review else [prep,review]])
        thresholds=copy.deepcopy(plan['validation_thresholds'])
        if changed_threshold:thresholds['service_mare_max']=999.
        protocol=gate.freeze_gate_protocol(model_paths={'CPU_fixture':str(model)},source_paths=[r['path'] for r in refs],training_data_paths=[train['path']],
            candidates=[dict(id=c['id'],model_id='CPU_fixture',execution_context_sha256=c['execution_context_sha256']) for c in plan['candidates']],
            baseline_candidate_id=plan['baseline_candidate_id'],validation_domain=[dict(run_id=plan['run_id'],requests=[{k:r[k] for k in ('request_id','pool_id','task')} for r in plan['trace']])],
            thresholds=thresholds,created_unix_ns=1,weights=plan['weights'],queue_sample_offsets_ns=plan['queue_sample_offsets_ns'],public_input_catalog_path=catalog)
        sealed_cells=copy.deepcopy(cells)
        if changed_cells:sealed_cells[0]['requests'][0]['queue_ms']+=1
        predictions=gate.seal_predictions(protocol,sealed_cells,created_unix_ns=2)
        m.save(root/'protocol.json',protocol);m.save(root/'predictions.json',predictions)
        m.save(root/'production.json',dict(schema=m.CAMPAIGN_SCHEMA,status='ARTIFACT_BOUND_PREDICTIONS_SEALED_REQUIRES_POSTSEAL_GPU_VALIDATION',
            protocol=m.receipt(root/'protocol.json'),predictions=m.receipt(root/'predictions.json'),preparation=prep,independent_review=review,
            original_v5_precollection_bindings_passed=True,all_GPU_must_start_after_seal=True,GPU_started=False))
        return [plan_path,root/'protocol.json',root/'predictions.json',root/'production.json'],protocol,plan

    def test_actual_original_gate_and_v55_receipt_contract_accept_exact_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            args,protocol,plan=self.fixture(Path(d));actual,sources=m.prerequisites(*args)
            self.assertEqual(actual,plan);self.assertTrue(all(ref in protocol['sources'] for ref in sources))

    def test_review_protocol_membership_thresholds_and_exact_verified_cells_required(self):
        for mutation,pattern in [('unbound_review','frozen'),('changed_threshold','thresholds'),('changed_cells','prepared cells')]:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                args,_,_=self.fixture(Path(d),**{mutation:True})
                with self.assertRaisesRegex(ValueError,pattern):m.prerequisites(*args)

    def test_foreign_schema_unknown_cohort_and_model_binding_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,protocol,_=self.fixture(root)
            production=m.read(args[3]);production['schema']='gsdeploy.projected_handler_artifact_binding_campaign.v48';args[3].write_text(json.dumps(production))
            with self.assertRaisesRegex(ValueError,'v55'):m.prerequisites(*args)
        for kind in ('model','unknown','grid','pool'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);args,protocol,_=self.fixture(root);prediction=m.read(args[2])
                if kind=='model':
                    m.save(root/'foreign-model.json',dict(foreign=True));protocol['models']['CPU_fixture']=m.receipt(root/'foreign-model.json')
                    protocol['protocol_sha256']=identity({k:v for k,v in protocol.items() if k!='protocol_sha256'});args[1].write_text(json.dumps(protocol))
                    prediction['protocol_sha256']=protocol['protocol_sha256']
                elif kind=='unknown':prediction['cells'][0]['requests'][0]['service_ms']=None
                elif kind=='grid':prediction['cells'][0]['queue_depth_samples'].pop()
                else:prediction['cells'][0]['requests'][0]['pool_id']='foreign'
                prediction['prediction_sha256']=identity({k:v for k,v in prediction.items() if k!='prediction_sha256'});args[2].write_text(json.dumps(prediction))
                production=m.read(args[3]);production['protocol']=m.receipt(args[1]);production['predictions']=m.receipt(args[2]);args[3].write_text(json.dumps(production))
                with self.assertRaises(ValueError):m.prerequisites(*args)


class IndependentProcessBoundaryTests(unittest.TestCase):
    def setUp(self):
        context=patch.object(independent,'m',m);context.start();self.addCleanup(context.stop)

    test_timeout_kills_owned_descendant_and_preserves_unrelated_session=independent.IndependentSupervisorTests.test_timeout_kills_owned_descendant_and_preserves_unrelated_session
    test_exact_13GiB_floor_and_6GiB_increment_boundaries=independent.IndependentSupervisorTests.test_exact_13GiB_floor_and_6GiB_increment_boundaries


if __name__=='__main__':unittest.main()
