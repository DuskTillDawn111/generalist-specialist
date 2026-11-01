"""CPU-only reproducible audit; no native execution or validation label fitting."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import sys
import unittest
from tests import test_lifecycle_engine_checkpoint_v59 as fixture
from gsdeploy.forecasting.metrics import score_forecast
from . import lifecycle_engine_checkpoint_v59 as cp
from . import lifecycle_scheduler_checkpoint_v1 as scp
from .implementation_source_closure_v27 import implementation_sources
from .owner_lifecycle_mark_v21 import receipt
from .profiling import digest

MODULES=('tests.test_lifecycle_engine_checkpoint_v59',
         'tests.test_lifecycle_scheduler_checkpoint_v1',
         'tests.test_learned_lifecycle_scheduler_v49_independent_review',
         'tests.test_owner_lifecycle_v21')


def audit(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).resolve().parents[2]
    sources=implementation_sources(__file__,source_root/'tools/predictor_validation_20261027/owner_lifecycle_engine_v59.py')
    log=io.StringIO()
    suite=unittest.TestLoader().loadTestsFromNames(MODULES)
    test_sources=[]
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith('tests.') and path and str(path).endswith('.py'):
            test_sources.append(receipt(path))
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'tests.log').write_text(log.getvalue())
    example={}
    if result.wasSuccessful():
        x=fixture.EngineCheckpointTests();x.setUp()
        try:
            evidence=output/'cpu_fixture_evidence.txt'
            evidence.write_text('Real v42 numeric CPU fixture, no fitted artifact')
            x.component.fixture.source=evidence
            baseline_engine=x.engine();baseline=x.run_engine(baseline_engine)
            checkpoint=None;cutoffs=[0,500,1000,5001,12000,17001,35000,70000,200000,350000,
                                     1377296,1400000,1419295,1500000]
            for cutoff in cutoffs:
                paused=x.run_engine(x.engine(),checkpoint=checkpoint,pause_at_ns=cutoff)
                checkpoint=json.loads(json.dumps(paused['checkpoint'],allow_nan=False))
                if cutoff==1400000:
                    (output/'pending_handoff_checkpoint.json').write_text(json.dumps(checkpoint,indent=2,allow_nan=False)+'\n')
            resumed_engine=x.engine();resumed=x.run_engine(resumed_engine,checkpoint=checkpoint)
            if resumed!=baseline:raise AssertionError('Actual multi-pause scenario differs from uninterrupted result')
            ready={owner:0 for owner in x.owners}
            if scp._state(resumed_engine.scheduler,ready)!=scp._state(baseline_engine.scheduler,ready):
                raise AssertionError('Final scheduler/oracle/mark differs after resume')
            score=score_forecast(resumed,[],x.aliases,{'text':10.},{'text':1.},{'f':{'text':.2}},
                phase_start_ns=0,now_ns=0,deadline_ns=x.deadline,memory_budget_mib=100.,
                energy_reference_watts=10.,observed_peak_memory_mib=30.,
                observed_board_energy_joules=0.,memory_safe_factor=1.1)
            (output/'uninterrupted_forecast.json').write_text(json.dumps(baseline,indent=2,allow_nan=False)+'\n')
            (output/'resumed_forecast.json').write_text(json.dumps(resumed,indent=2,allow_nan=False)+'\n')
            example=dict(fixture_kind='CPU_SYNTHETIC_EXPLICIT_MODEL_NOT_TRAINED_GPU_EVIDENCE',
                all_four_resident_owners=x.owners,public_requests=x.rows,
                original_measurement_window_ns=[0,x.deadline],original_drain_ns=x.drain,
                exact_pause_cutoffs_ns=cutoffs,uninterrupted_and_resumed_full_output_identical=True,
                final_scheduler_oracle_mark_identical=True,
                finishes={rid:dict(start_ns=row['start_ns'],handler_end_ns=row['handler_end_ns'],
                    parent_receipt_ns=row['finish_ns'],worker_next_ready_ns=row['worker_next_ready_ns'])
                    for rid,row in resumed['finishes'].items()},
                actual_context_events=resumed_engine.scheduler.transitions,
                actual_original_six_term_score=score,
                board_energy_joules=resumed['board_energy_joules'],peak_memory_mib=resumed['peak_memory_mib'],
                events=resumed['events'],artifacts=[receipt(output/name) for name in
                    ('cpu_fixture_evidence.txt','pending_handoff_checkpoint.json','uninterrupted_forecast.json','resumed_forecast.json')])
        finally:x.doCleanups()
    unchanged=all(receipt(ref['path'])==ref for ref in sources+test_sources)
    frozen_report=source_root/'artifacts/predictor-lifecycle-scheduler-checkpoint-v1-20261028-cpu01/report.json'
    old=json.loads(frozen_report.read_text())
    original_sources=old['module_source_closure']
    old_unchanged=all(receipt(ref['path'])==ref for ref in original_sources)
    profile_names=('paired_profile_campaign_v6.py','profile_training_supervisor_v3.py','profile_owned_collect_v3.py')
    known=('bc22011c4b12d9c0baaa2d4f2a207ca52ba04f355ecbe6ae6a527ad24644c17d',
           '854dfc77b43b2bf7af87935bbe145e7202b81d53359143f61834048d72f4e6a5',
           '1c3222addcafbabbda813ab680822090df966062a9430e7fc3d83dc814c52d83')
    profile=[receipt(source_root/'tools/predictor_validation_20261027'/name) for name in profile_names]
    profile_unchanged=all(ref['sha256']==expected for ref,expected in zip(profile,known))
    report=dict(schema='gsdeploy.lifecycle_engine_checkpoint_cpu_audit.v59',
        status='PASS_SUPPORTED_SIMULATED_ENGINE_ONLY' if result.wasSuccessful() and unchanged and old_unchanged and profile_unchanged else 'FAIL',
        GPU_started=False,labels_fit=False,test_modules=list(MODULES),test_count=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),test_log=receipt(output/'tests.log'),
        simulated_main_engine_resume_supported=True,native_observation_assimilation_supported=False,
        physical_progress_observed=False,physical_accuracy_validated=False,deployment_qualified=False,
        no_prefix_event_replay=True,no_future_RESULT_assimilation=True,
        original_cohort_routes_FIFO_weights_thresholds_stability_preserved=True,
        original_node_integer_completion_rounding_preserved=True,
        no_budget_redraw_on_import=True,no_global_request_cap_added=True,
        unsupported=['native observation assimilation','unknown scheduler/component/engine fields or subclasses',
            'non-v6 queue sidecars','initial pre-SERVICE component state',
            'stateful/stochastic providers or process-global RNG restoration',
            'legacy event-at-or-after pause API','same-holder/all-idle statistical UNKNOWN cases'],
        example=example,source_closure_unchanged=unchanged,
        prior_component_full_source_closure_unchanged=old_unchanged,
        frozen_profile_sources_unchanged=profile_unchanged,frozen_profile_sources=profile,
        prior_component_report=receipt(frozen_report),module_source_closure=sources,
        test_fixture_sources=test_sources)
    path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report,receipt(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();report,ref=audit(args.output)
    print(json.dumps(dict(status=report['status'],tests=report['test_count'],report=ref)))
    raise SystemExit(0 if report['status'].startswith('PASS') else 1)
