"""Resume Module 3B after host power-off without treating it as model instability.

This is a one-time correction for the identified interrupted attempt. It never
repeats a completed raw run, preserves every old artifact, and reduces the
effective configuration budget so physical attempts remain <=18.
"""
import json
from pathlib import Path
import shutil
import time

from deployment_core.models import load_hardware, load_model_catalog
from deployment_core.workload import Workload

from . import GuidedDeploymentSearch, SearchConfig
from .measurement_runtime import MeasurementRuntimeEvaluator
from .pilot_provenance import verify_assets_unchanged
from .real_objective import NoiseAwareRuntimeObjective
from .real_pilot import OUT, SETTINGS, SEEDS, CONFIG, MIN_RELATIVE_IMPROVEMENT, write_reports
from .runtime_cache import RuntimeEvaluationCache, file_hash, save_json
from .tests.fixtures import forbid_enumeration

INTERRUPTED_KEY = 'f420f00ba40f7451517d5d89f8559650d154adf56f6dff0b1a04f6c09659df0e'


def archive(path, destination):
    path = Path(path)
    if path.exists():
        if destination.exists():
            raise RuntimeError('Archive destination exists; refusing to overwrite ' + str(destination))
        shutil.move(str(path), str(destination))
    return dict(path=str(destination), sha256=file_hash(destination)) if destination.exists() else None


def correct_manifest(output=OUT):
    cache = RuntimeEvaluationCache(output, 6, 3)
    with cache.locked() as locked:
        data = locked.data
        entry = data['configurations'][INTERRUPTED_KEY]
        if entry.get('interrupted_attempts'):
            return
        matching = [r for r in entry['repetitions'] if r['index'] == 1]
        if len(matching) != 1 or matching[0]['status'] != 'complete':
            raise RuntimeError('Unexpected interrupted attempt record; manual audit required')
        record = matching[0]
        if not record['runtime']['queue_metrics'].get('reconstruction_error'):
            raise RuntimeError('Attempt 1 has measured evidence; cannot relabel as external interruption')
        archive_dir = Path(output) / 'interruption_archive'
        log_to_check = (Path(record['log_path']) if Path(record['log_path']).exists()
                        else archive_dir / Path(record['log_path']).name)
        if log_to_check.exists() and 'Starting measurement window.' not in log_to_check.read_text():
            raise RuntimeError('Interrupted log does not show a measurement window')
        # The original harness process was absent, and no harness raw JSON existed
        # when the user clarified that the host was powered off. The JSON below
        # was an adapter-created failure envelope, not a GPU runtime result.
        envelope = (Path(record['raw_result_path']) if Path(record['raw_result_path']).exists()
                    else archive_dir / Path(record['raw_result_path']).name)
        if not json.loads(envelope.read_text()).get('adapter_failure'):
            raise RuntimeError('Unexpected real harness raw result; refusing to amend')
        archive_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, path in [('failure_envelope', record['raw_result_path']),
                           ('execution_log', record['log_path']),
                           ('process_metadata', Path(record['raw_result_path']).with_suffix('.process.json')),
                           ('config', record['config_path'])]:
            source = Path(path)
            paths[name] = archive(source, archive_dir / source.name)
        old_aggregate = Path(entry['aggregate'])
        paths['old_aggregate'] = archive(old_aggregate, archive_dir / 'aggregate_before_correction.json')
        trace_path = Path(output) / 'search_trace.json'
        if trace_path.exists():
            old_trace = archive_dir / 'search_trace_before_correction.json'
            if not old_trace.exists():
                shutil.copy2(trace_path, old_trace)
            paths['old_trace'] = dict(path=str(old_trace), sha256=file_hash(old_trace))
        attempt = next(r for r in data['run_attempts'] if r['cache_key'] == INTERRUPTED_KEY and r['repetition'] == 1)
        attempt.update(interrupted_external=True, outcome='host_power_off_no_harness_raw_result',
                       original_artifacts=paths, original_raw_result_path=attempt['raw_result_path'])
        attempt['raw_result_path'] = paths['failure_envelope']['path']
        entry['interrupted_attempts'] = [dict(record, external_interruption=True, archived_artifacts=paths)]
        entry['repetitions'] = [r for r in entry['repetitions'] if r['index'] != 1]
        # The user correction arrived while a different repetition was running.
        # Its harness wrote a complete raw result before the paused parent was
        # stopped. Recover that raw result without inventing a runtime failure
        # from the missing parent-side subprocess return code.
        active = [r for configuration in data['configurations'].values()
                  for r in configuration['repetitions'] if r['status'] == 'started']
        if len(active) != 1:
            raise RuntimeError('Expected exactly one recoverable in-flight repetition')
        active_record = active[0]
        completed_raw = Path(active_record['raw_result_path'])
        if not completed_raw.exists():
            raise RuntimeError('In-flight repetition has no saved harness raw result')
        raw = json.loads(completed_raw.read_text())
        if (raw.get('status') != 'completed' or raw.get('errors') or not raw.get('measurement_window')
                or not raw.get('system_metrics') or raw['system_metrics'].get('completion_ratio') != 1):
            raise RuntimeError('In-flight raw result is not a complete, error-free measurement')
        active_record['process_outcome'] = dict(returncode=0, recovered_from_completed_raw=True,
                                                raw_sha256=file_hash(completed_raw),
                                                parent_stopped_after_harness_output=True)
        entry.update(status='reserved', aggregate=None, aggregate_sha256=None)
        locked.flush()
        print(json.dumps(dict(event='external_interruption_reclassified', cache_key=INTERRUPTED_KEY,
                              physical_attempts=len(data['run_attempts']), valid_repetitions=len(entry['repetitions']))), flush=True)


def resume(output=OUT):
    output = Path(output)
    if (output / 'decision_audit.json').exists():
        print('Pilot already finalized; no repeat runs.', flush=True)
        return
    correct_manifest(output)
    assets = json.loads((output / 'asset_hashes.json').read_text())
    verify_assets_unchanged(assets)
    source_hashes = json.loads((output / 'source_model_hashes.json').read_text())
    workload = Workload(json.loads((output / 'pilot_workload.json').read_text())['task_arrival_rates'])
    catalog, hardware = load_model_catalog(), load_hardware()
    effective = SearchConfig(**dict(CONFIG.__dict__, runtime_evaluation_budget=5))
    save_json(output / 'recovery_correction.json', dict(
        reason='Host power-off during qwen+resnet50 repetition 1, no harness raw result. '
               'External interruption is not a model stability label.',
        interrupted_cache_key=INTERRUPTED_KEY, initial_plan_configuration_limit=6,
        effective_configuration_limit=5, physical_run_attempt_limit=18,
        required_valid_repetitions_per_configuration=3,
        policy='One replacement for interrupted seed 7101; no completed run repeated. '
               'Remaining candidate budget reduced to keep physical attempts <=18.',
        original_search_source_sha256=json.loads((output / 'pilot_freeze.json').read_text())['search_sources']['deployment_search/real_pilot.py'],
        corrected_search_source_sha256=file_hash(Path(__file__).with_name('real_pilot.py')),
        correction_source_sha256=file_hash(__file__),
        frozen_measurement_source_hashes_reused=True,
        original_artifacts_preserved='interruption_archive/',
        effective_search_config=effective))
    evaluator = MeasurementRuntimeEvaluator(output, catalog, source_hashes, SETTINGS, SEEDS, 6)
    last = None
    def checkpoint(result):
        nonlocal last
        save_json(output / 'search_trace.json', result.to_dict())
        info = dict(event='search_checkpoint', status=result.status,
                    runtime_evaluations=result.runtime_evaluation_count, lp_solves=result.lp_solve_count,
                    deployment=result.deployment, iterations=result.search_iterations,
                    top_b=result.iterations[-1]['selected_top_b'] if result.iterations else [])
        if info != last:
            print(json.dumps(info), flush=True)
            last = info
    with forbid_enumeration() as guards:
        result = GuidedDeploymentSearch(evaluator, predictor=None,
            runtime_objective=NoiseAwareRuntimeObjective(1., MIN_RELATIVE_IMPROVEMENT),
            trace_callback=checkpoint).search(workload, catalog, hardware, effective)
    prior = json.loads(evaluator.cache.path.read_text())
    evaluator.evaluate(result.deployment, workload, result.canonical.routing)
    after = json.loads(evaluator.cache.path.read_text())
    if len(after['run_attempts']) != len(prior['run_attempts']):
        raise RuntimeError('Cache replay caused a new physical run')
    if len(after['run_attempts']) > 18:
        raise RuntimeError('Physical run budget exceeded')
    verify_assets_unchanged(assets)
    audit = write_reports(output, result, sum(g.call_count for g in guards))
    if not audit['passed']:
        raise RuntimeError('Decision audit failed: ' + str(audit['errors']))
    print(json.dumps(dict(event='pilot_complete_after_external_interruption',
                          unique_runtime_evaluations=audit['unique_runtime_evaluations'],
                          physical_run_attempts=audit['actual_real_runs'],
                          interrupted_external=audit['externally_interrupted_runs'],
                          output=str(output))), flush=True)


if __name__ == '__main__':
    resume()
