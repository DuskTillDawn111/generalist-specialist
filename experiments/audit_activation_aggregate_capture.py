"""Full-cohort output, runtime and board-energy audit for aggregate captures.

Repeated inputs retain their offered multiplicity. Output changes are reported
with actual frozen-reference quality, not silently rejected as timing noise.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_activation_native import load_bundle
from experiments.analyze_activation_semantic import output_identity
from experiments.analyze_activation_trim_episode_energy import describe as describe_energy
from gsdeploy.forecasting.native_training import validate_capture
from gsdeploy.normalization import quality_loss
from gsdeploy.quality import quality_digest, sample_totals
from gsdeploy.tasks import page_ocr_text, score_result


def score_output(response, task, pool_id, data):
    if response.get('success') is not True:
        raise ValueError('Unsuccessful output cannot enter quality averages')
    result = dict(response)
    result['task'] = result.get('task', result.get('task_type'))
    if result['task'] != task:
        raise ValueError('Response task differs from offered task')
    if task == 'ocr':
        result['prediction'] = page_ocr_text(result['prediction'])
    return sample_totals(task, score_result(data['quality']['references'][task][pool_id], result))


def summarize_quality(rows, data, task):
    a = sum(r['error_numerator'] for r in rows)
    z = sum(r['reference_denominator'] for r in rows)
    ba = sum(r['baseline_error_numerator'] for r in rows)
    bz = sum(r['baseline_reference_denominator'] for r in rows)
    if not rows or z <= 0 or bz <= 0:
        raise ValueError('Complete positive quality denominator required')
    current, baseline = quality_loss(data, task, a/z), quality_loss(data, task, ba/bz)
    return dict(requests=len(rows), distinct_inputs=len({r['pool_id'] for r in rows}),
                native_loss=a/z, matched_solo_native_loss=ba/bz,
                normalized_loss=current, matched_solo_normalized_loss=baseline,
                normalized_loss_delta=current-baseline if current is not None and baseline is not None else None,
                changed_outputs=sum(r['output_changed'] for r in rows))


def baseline_outputs(bundles, data):
    if len(bundles) != 2 or len({b['source'] for b in bundles}) != 2:
        raise ValueError('Two distinct complete solo baselines required')
    maps = []
    for bundle in bundles:
        validate_capture(bundle)
        if bundle['plan']['mode'] != 'solo':
            raise ValueError('Solo baseline required')
        mapping = {}
        for row in bundle['raw']['request_records']:
            key = bundle['context']['resident_aliases'][row['model']], row['task'], row['pool_id']
            if key in mapping:
                raise ValueError('Duplicate solo input')
            mapping[key] = dict(output_sha256=output_identity(row['worker_response']),
                quality_totals=score_output(row['worker_response'], row['task'], row['pool_id'], data))
        maps.append(mapping)
    if bundles[0]['context'] != bundles[1]['context'] or maps[0] != maps[1]:
        raise ValueError('Solo baseline runtime/outputs are not identical')
    return maps[0]


def audit(bundle, baselines, data, expected_context):
    validate_capture(bundle, allowed_splits=('training', 'development_validation'))
    if bundle['context'] != expected_context:
        raise ValueError('Capture and matched solo runtime differ')
    quality = data['quality']
    if (quality_digest(quality['references']) != quality['identity']['reference_sha256']
            or expected_context['input_pool_sha256'] != quality['identity']['input_pool_sha256']):
        raise ValueError('Frozen input/reference identity changed')
    raw, context = bundle['raw'], bundle['context']
    issues, changed, scored = [], [], []
    for alias, family in context['resident_aliases'].items():
        actual = raw['ready'][alias].get('runtime_scope', {})
        strict = context['runtime_policy']['scope'] == 'all_strict' or family == 'generalist_2'
        if (actual.get('scope') != context['runtime_policy']['scope'] or actual.get('slot') != family
                or actual.get('strict_requested') is not strict
                or actual.get('deterministic_algorithms') is not strict
                or actual.get('cublas_workspace_config') != (':4096:8' if strict else None)):
            issues.append(alias+':RUNTIME_SCOPE_CHANGED')
        if strict and (actual.get('deterministic_warn_only') is not False
                       or actual.get('cudnn_benchmark') is not False or actual.get('cudnn_deterministic') is not True):
            issues.append(alias+':STRICT_RUNTIME_INCOMPLETE')
    for phase in ('priming_request_records', 'request_records'):
        for row in raw[phase]:
            family = context['resident_aliases'][row['model']]
            response = row['worker_response']
            if response.get('success') is not True:
                issues.append(row['request_id']+':OUTPUT_UNSUCCESSFUL')
                continue
            if family == 'generalist_2':
                modifier = response.get('memory_optimizations', {}).get('exllama_reconstruct_runtime', {})
                if (modifier.get('matmul_recons_thd') != 1 or modifier.get('configured_after_model_load') is not True
                        or modifier.get('quantized_weights_changed') is not False or modifier.get('quantized_layers', 0) <= 0):
                    issues.append(row['request_id']+':G2_RUNTIME_PATH_UNVERIFIED')
            if phase != 'request_records':
                continue
            key = family, row['task'], row['pool_id']
            baseline = baselines[key]
            a, z = score_output(response, row['task'], row['pool_id'], data)
            ba, bz = baseline['quality_totals']
            identity = output_identity(response)
            item = dict(request_id=row['request_id'], family=family, task=row['task'], pool_id=row['pool_id'],
                        job_id=row['profile_job_id'], error_numerator=a, reference_denominator=z,
                        baseline_error_numerator=ba, baseline_reference_denominator=bz,
                        output_changed=identity != baseline['output_sha256'])
            scored.append(item)
            if item['output_changed']:
                changed.append(dict(item, output_sha256=identity, baseline_output_sha256=baseline['output_sha256']))
    groups = defaultdict(list)
    for item in scored:
        groups[item['family']+'/'+item['task']].append(item)
    energy = describe_energy(raw)
    energy['scope'] = 'Measured GPU board launch/priming/formal/drain/tail energy, no post-window teardown or CPU/RAM energy'
    if not energy['coverage_valid']:
        issues.append('EPISODE_ENERGY_COVERAGE_MISSING')
    if len(scored) != len(raw['request_records']):
        issues.append('QUALITY_COHORT_INCOMPLETE')
    complete = len(scored) == len(raw['request_records'])
    return dict(run_id=raw['run_id'], source=bundle['source'], issues=issues,
        execution_integrity_passed=not issues, requested=len(raw['request_records']), quality_scored=len(scored),
        per_path={key: summarize_quality(values, data, values[0]['task']) for key, values in groups.items()} if complete else {},
        quality_cohort_complete=complete, changed_output_count=len(changed), changed_outputs=changed,
        scores=scored, energy=energy,
        observed_owned_peak_mib=max(s['owned_used_mib'] for s in raw['board_samples']),
        physical_predictor_qualified=False, formal_utility_qualified=False,
        quality_change_is_automatic_failure=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, action='append', required=True)
    p.add_argument('--capture', type=Path, action='append', required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.reference.read_text())['catalog']
    bundles = [load_bundle(path.resolve()) for path in args.baseline]
    baselines = baseline_outputs(bundles, data)
    result = dict(schema='gsdeploy.aggregate_capture_integrity.v1',
        baseline_sources=[b['source'] for b in bundles],
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        runs=[audit(load_bundle(path.resolve()), baselines, data, bundles[0]['context']) for path in args.capture],
        quality_anchors_unchanged=True, physical_predictor_qualified=False, formal_utility_qualified=False, U=None)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), runs=[dict(run_id=r['run_id'], issues=r['issues'],
                          changed_outputs=r['changed_output_count']) for r in result['runs']])))
