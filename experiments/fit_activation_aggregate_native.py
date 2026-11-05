"""Low-dimensional coarse-native fit; aggregate replay determines usability.

Uses the existing native progress-law representation and FIFO provider. The
fitting policy is explicitly new: regularized additive peer effects, no maximum
individual work-error veto. Old fitters and their decisions are unchanged.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import least_squares
from experiments.collect_activation_training import POOL
from experiments.fit_activation_native import load_bundle
from experiments.fit_activation_declared_idle_energy import fit_declared_idle_age_energy
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.energy import integrate_power
from gsdeploy.forecasting.lifecycle import fit_lifecycle
from gsdeploy.forecasting.native_context import SCHEMA, NativeContextProvider, categories, counts, design
from gsdeploy.forecasting.native_training import build_template, normalize_native, stage_layout
from gsdeploy.paper_training import export_segments

FIT_POLICY = dict(schema='gsdeploy.aggregate_native_fitting_policy.v1',
                  descriptor='additive_log_progress_peer_counts', ridge=.01,
                  robust_loss='soft_l1', loss_scale=.1, coefficient_bounds=[-3., 3.],
                  minimum_independent_runs=2, maximum_work_error_is_gate=False,
                  repeated_activity_vertices_required=True,
                  independent_aggregate_validation_required=True)
IDLE_EDGES = [0., 2., 10., 20., 121.]


def fit_episode_envelope(bundles):
    """Training-only restart and controller overhead, never a heldout clock."""
    records = []
    for bundle in bundles:
        raw = bundle['raw']
        start, end = raw['startup_begin_time_ns'], raw['observation_start_ns']
        energy = integrate_power(raw['board_samples'], start, end, max_gap_seconds=.25)
        if energy is None:
            raise ValueError('Startup energy telemetry coverage missing')
        rows = {r['request_id']: r for r in raw['request_records']}
        prefix_end = max(s['end_ns'] for r in raw['priming_request_records']
                         for s in stage_layout(r) if s['kind'] == 'gpu')
        closures, gaps = [], []
        batches = raw['batches']
        for i, batch in enumerate(batches):
            ready = [max(rows[r]['completion_time_ns'],
                         rows[r]['previous_handler_next_request_ready']['perf_ns'])
                     for r in batch['request_ids']]
            ideal = max([batch['start_ns'], batch.get('arrival_end_ns') or batch['start_ns'], *ready])
            closures.append((batch['end_ns'] - ideal) / 1e6)
            if i:
                gaps.append((batch['start_ns'] - batches[i-1]['end_ns']) / 1e6)
        record = dict(run_id=raw['run_id'], source=bundle['source'],
            startup_ms=(end-start)/1e6, startup_board_joules=energy,
            initial_idle_age_ms=(end-prefix_end)/1e6,
            formal_lead_ms=(batches[0]['start_ns']-end)/1e6,
            batch_closure_ms=statistics.mean(closures),
            between_batch_ms=statistics.mean(gaps),
            formal_tail_ms=(raw['observation_end_ns']-batches[-1]['end_ns'])/1e6)
        if any(v < 0 for k, v in record.items() if k.endswith(('_ms', '_joules'))):
            raise ValueError('Noncausal startup/controller boundary')
        records.append(record)
    keys = [k for k in records[0] if k.endswith(('_ms', '_joules'))]
    return dict(schema='gsdeploy.aggregate_episode_envelope.v1', training_records=records,
                means={k: statistics.mean(r[k] for r in records) for k in keys},
                policy='equal weight per independent restart; constant expected controller overhead',
                scope='launch through observation tail; no post-observation process teardown or CPU/RAM energy',
                independent_accuracy_qualified=False)


def fit_progress(template, observations):
    profiles = {p['id']: p for p in template['profiles']}
    names = categories(profiles)
    groups = defaultdict(list)
    seen = set()
    for row in observations:
        if (row.get('split') != 'training' or not row.get('run_id')
                or row.get('execution_context_sha256') != template['execution_context_sha256']
                or row['id'] in seen):
            raise ValueError('Distinct matched training observations required')
        seen.add(row['id'])
        profile = profiles[row['profile_id']]
        segment = next(s for s in profile['segments'] if s['id'] == row['segment_id'])
        if segment['kind'] != 'gpu':
            continue
        baseline = segment['duration_ms']['mean']
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError('Positive baseline required')
        intervals = []
        for interval in row['intervals']:
            if interval['members'].get(row['target_alias']) != dict(profile_id=profile['id'], segment_id=segment['id']):
                raise ValueError('Observed target absent from exposure interval')
            duration = interval['duration_ms']
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError('Positive complete exposure required')
            intervals.append((duration / baseline, counts(interval['members'], profiles, names, omit=row['target_alias'])))
        if not intervals:
            raise ValueError('Empty target exposure')
        groups[profile['family'], profile['task'], segment['id']].append((row, intervals))
    fitted, diagnostics = [], []
    for (family, task, segment_id), rows in sorted(groups.items()):
        vertex_runs = defaultdict(set)
        for row, intervals in rows:
            for _, vertex in intervals:
                vertex_runs[tuple(vertex)].add(row['run_id'])
        vertices = sorted(v for v, runs in vertex_runs.items() if len(runs) >= 2)
        used = [i for i in range(len(names)) if any(v[i] for v in vertex_runs)]
        terms = [[]] + [[i] for i in used]
        xs = [np.array([design(v, terms) for _, v in intervals]) for _, intervals in rows]
        ws = [np.array([duration for duration, _ in intervals]) for _, intervals in rows]
        run_ids = sorted({row['run_id'] for row, _ in rows})
        def data_residual(beta):
            return np.array([float(w @ np.exp(x @ beta)) - 1. for x, w in zip(xs, ws)])
        def residual(beta):
            return np.r_[data_residual(beta), math.sqrt(FIT_POLICY['ridge']) * beta]
        fit = least_squares(residual, np.zeros(len(terms)), bounds=FIT_POLICY['coefficient_bounds'],
                            loss=FIT_POLICY['robust_loss'], f_scale=FIT_POLICY['loss_scale'], max_nfev=1000)
        errors = abs(data_residual(fit.x))
        data_jacobian = np.array([((w * np.exp(x @ fit.x))[:, None] * x).sum(axis=0) for x, w in zip(xs, ws)])
        usable = bool(fit.success and len(run_ids) >= 2 and vertices and np.isfinite(fit.x).all())
        diagnostics.append(dict(family=family, task=task, segment_id=segment_id,
            fitted=usable, observations=len(rows), independent_run_ids=run_ids, parameters=len(terms),
            unregularized_rank=int(np.linalg.matrix_rank(data_jacobian)),
            maximum_work_error=float(errors.max()), mean_work_error=float(errors.mean()),
            p95_work_error=float(np.quantile(errors, .95)), bound_hits=fit.active_mask.tolist(),
            repeated_activity_vertices=len(vertices), unrepeated_activity_vertices=len(vertex_runs)-len(vertices),
            maximum_work_error_is_gate=False, physical_accuracy_qualified=False))
        if usable:
            fitted.append(dict(family=family, task=task, segment_id=segment_id, terms=terms,
                coefficients=fit.x.tolist(), support_vertices=[list(v) for v in vertices],
                independent_run_ids=run_ids, observation_ids=[r['id'] for r, _ in rows]))
    return dict(schema=SCHEMA, status='FITTED_UNVALIDATED',
        execution_context_sha256=template['execution_context_sha256'], categories=[list(n) for n in names],
        native_scope='mixed_host_transfers_CUDA', non_gpu_progress_assumption='pinned_solo_duration',
        physical_bound=False, progress_groups=fitted, resources=None,
        fit_diagnostics=diagnostics, fitting_policy=FIT_POLICY)


def fit_captures(bundles, pool_items):
    if len({b['raw']['run_id'] for b in bundles}) != len(bundles):
        raise ValueError('Distinct training run identities required')
    template, evidence = build_template([b for b in bundles if b['plan']['mode'] == 'solo'], pool_items)
    documents, observations, lifecycle, energy_runs = [], [], [], []
    for bundle in bundles:
        doc = normalize_native(bundle, template)
        documents.append(doc)
        exported, _ = export_segments(doc, template, bundle['context']['resident_aliases'])
        observations.extend(dict(row, run_id=doc['run_id']) for row in exported)
        lifecycle.extend(doc['lifecycle_observations'])
        prefix = [s['end_ns'] for row in bundle['raw']['priming_request_records']
                  for s in stage_layout(row) if s['kind'] == 'gpu']
        if not prefix or max(prefix) > doc['observation_window_ns'][0]:
            raise ValueError('Completed causal priming prefix required')
        energy_runs.append(dict(run_id=doc['run_id'], split='training', window_ns=doc['observation_window_ns'],
            initial_last_native_end_ns=max(prefix), telemetry_errors=bundle['raw']['telemetry_errors'],
            board_samples=bundle['raw']['board_samples'],
            spans=[dict(alias=r['alias'], profile_id=r['profile_id'], **s) for r in doc['requests'] for s in r['segments']]))
    template['native_activity_law'] = fit_progress(template, observations)
    try:
        energy, diagnostic = fit_declared_idle_age_energy(
            {p['id']: p for p in template['profiles']}, energy_runs, idle_edges=IDLE_EDGES)
    except UnknownEvidence as exc:
        energy, diagnostic = None, dict(accepted=False, reason=str(exc))
    if energy is not None and diagnostic['accepted']:
        energy['execution_context_sha256'] = template['execution_context_sha256']
        template['native_idle_age_energy'] = energy
    template['provenance']['sources'] = sorted(b['source'] for b in bundles)
    NativeContextProvider(template, template['execution_context_sha256'], allow_cpu_fixture=True)
    return dict(schema='gsdeploy.aggregate_native_fit.v1', status='FITTED_UNVALIDATED',
        fitting_policy=FIT_POLICY, profiles=template, baseline_evidence=evidence,
        resident_aliases=dict(bundles[0]['context']['resident_aliases']),
        lifecycle=fit_lifecycle(lifecycle, template['execution_context_sha256']),
        normalized_runs=documents, source_receipts=[b['source'] for b in bundles],
        idle_energy_diagnostic=diagnostic, idle_edges_seconds=IDLE_EDGES,
        episode_envelope=fit_episode_envelope(bundles),
        energy_scope='conditional formal window with observed priming prefix, not cold-start episode',
        admission_enabled=False, physical_predictor_qualified=False, formal_utility_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundles = [load_bundle(p.resolve()) for p in args.capture]
    content = POOL.read_bytes()
    if any(b['context']['input_pool_sha256'] != hashlib.sha256(content).hexdigest() for b in bundles):
        raise ValueError('Frozen pool changed')
    sources = [Path(__file__).resolve(), ROOT/'experiments/fit_activation_native.py',
               ROOT/'experiments/fit_activation_declared_idle_energy.py', ROOT/'gsdeploy/paper_training.py',
               ROOT/'gsdeploy/paper_capture.py', ROOT/'gsdeploy/energy.py', *sorted((ROOT/'gsdeploy/forecasting').glob('*.py'))]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = fit_captures(bundles, [json.loads(line) for line in content.decode().splitlines()])
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha for p, sha in hashes.items()):
        raise ValueError('Fitting source changed during execution')
    result['fitting_sources'] = hashes
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), groups=len(result['profiles']['native_activity_law']['progress_groups']),
                          energy_fitted=result['idle_energy_diagnostic']['accepted'], physical_predictor_qualified=False)))
