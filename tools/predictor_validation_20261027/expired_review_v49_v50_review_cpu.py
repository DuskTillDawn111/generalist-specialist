"""Reproduce CPU-only v49 scheduling and v50 private-process regressions."""
import argparse
import io
import json
from pathlib import Path
import unittest

from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt
from .unknown_preflight_archive_v1 import save_synced
from .learned_availability_forecast_v20 import verify_sources


MODULES = [
    'tests.test_learned_lifecycle_scheduler_v49_independent_review',
    'tests.test_lifecycle_expired_empty_review_diagnostic',
    'tests.test_projected_handler_bridge_v44_independent_review',
    'tests.test_learned_lifecycle_scheduler_v27',
    'tests.test_learned_lifecycle_oracle_v25',
    'tests.test_learned_context_oracle_v14',
    'tests.test_projected_handler_v50_independent_review',
]


def run(output):
    root = Path(__file__).resolve().parents[2]
    out = Path(output).absolute()
    out.mkdir(parents=True, exist_ok=False)
    source = root/'tools/predictor_validation_20261027'
    diagnostic_path = root/'artifacts/predictor-projected-handler-v44-skewed-skipped-boundary-independent-diagnostic-20261028-cpu01/report.json'
    diagnostic_ref = receipt(diagnostic_path)
    if diagnostic_ref['sha256'] != 'b6152d93fab41195391bce0c9b396a4b1929f1a24d0c7eb23bee7963fc5ad44e':
        raise ValueError('Frozen actual-v44 diagnostic must retain its reviewed identity')
    diagnostic = json.loads(diagnostic_path.read_text())
    frozen = [diagnostic[key] for key in ('scheduler_source','original_no_rewind_guard_source','independent_diagnostic_test')]
    verify_sources(frozen)
    refs = strict_references([implementation_sources(source/'projected_handler_preflight_v50.py'),
        receipt(__file__), diagnostic_ref,
        [receipt(root.joinpath(*name.split('.')).with_suffix('.py')) for name in MODULES]])
    verify_sources(refs)
    stream = io.StringIO()
    suite = unittest.TestLoader().loadTestsFromNames(MODULES)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (out/'tests.txt').write_text(stream.getvalue())
    verify_sources(refs)
    report = dict(schema='gsdeploy.expired_lifecycle_review_v49_and_private_fork_v50_independent_CPU_review.v1',
        status='PASS' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        test_log=receipt(out/'tests.txt'), sources=refs, frozen_actual_v44_diagnostic=diagnostic_ref,
        frozen_original_guard_and_diagnostic_sources_unchanged=True,
        v49=dict(implementation=receipt(source/'learned_lifecycle_scheduler_v49.py'),
            independent_tests=receipt(root/'tests/test_learned_lifecycle_scheduler_v49_independent_review.py'),
            new_numeric_regressions=6,
            absent_and_expired_empty_frontier_reviews_reach_engine=True,
            future_review_preserved_when_SAVE_candidate_moves_later=True,
            expired_review_republished_without_budget_redraw_or_age_reset=True,
            ordinary_active_frontier_rates_and_timing_preserved=True,
            pending_switch_immutable=True, same_holder_UNKNOWN_preserved=True,
            skipped_SAVE_no_rewind_UNKNOWN_preserved=True,
            original_real_numeric_v42_candidate_v44_bridge_v24_controller_used=True),
        v50=dict(factory=receipt(source/'projected_handler_forecast_v50.py'),
            preflight=receipt(source/'projected_handler_preflight_v50.py'),
            independent_tests=receipt(root/'tests/test_projected_handler_v50_independent_review.py'),
            new_independent_regressions=4,
            normalized_factory_AST_matches_v44_except_version_v49_scheduler_and_metadata=True,
            engine_replaces_mark_oracle_scheduler_and_keeps_compiled_input_identity=True,
            four_actual_small_CPU_forks_have_four_distinct_process_ids=True,
            numeric_numpy_arrays_and_mutable_history_private_to_each_candidate=True,
            parent_numeric_and_history_state_unchanged=True,
            injected_child_failure_retains_other_three_results_and_reaps_children=True,
            exact_gzip_roundtrip_used_in_fixture=True,
            fixture_requests_per_candidate=57,
            fitted_or_full_forecast_model_initialized=False),
        fit_performed=False, full_cohort_forecast_executed=False,
        GPU_started=False, validation_labels_read=False, source_bytes_unchanged=True,
        limitations=['This repairs event publication, not physical SAVE or service identification.',
            'Small CPU process fixtures do not establish full-cohort forecast accuracy or six-term utility.',
            'Projected worker-handler features and active-service bridge remain unvalidated hypotheses.',
            'The original no-rewind and same-holder UNKNOWN outcomes remain mandatory.'])
    save_synced(out/'report.json', report)
    if not result.wasSuccessful():
        raise RuntimeError('Independent CPU regression failed; inspect tests.txt')
    return receipt(out/'report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(run(parser.parse_args().output)))
