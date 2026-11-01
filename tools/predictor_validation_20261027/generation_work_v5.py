"""Exact public-content generation work for new requests of trained inputs.

Unknown content never falls back to a family/task distribution. Full source
closure and all v4 training observations remain unchanged and auditable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from fractions import Fraction
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .generation_work_v4 import KEY_FIELDS, WORK_FIELDS, SCHEMA as BASE_SCHEMA, integer, input_content_identity
from .historical_evidence_v4 import resolve_closure
from .native_collect import receipt, save
from .profiling import digest

SCHEMA = 'gsdeploy.exact_content_generation_work_hypothesis.v5'
EXACT_KEY_FIELDS = KEY_FIELDS + ('input_content_sha256', 'actual_input_tokens')


def checked(ref):
    if receipt(ref['path']) != ref:
        raise ValueError('Changed exact-content generation evidence: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text())


def fit(observations, *, allow_cpu_fixture=False):
    observations = list(observations)
    groups, seen = defaultdict(list), set()
    for row in copy.deepcopy(list(observations)):
        if ((row.get('kind') != 'measured' and not (allow_cpu_fixture and row.get('kind') == 'cpu_fixture'))
                or row.get('split') != 'training'):
            raise ValueError('Actual measured training generation work required')
        identity = row['source'], row['request_id']
        if identity in seen:
            raise ValueError('Duplicate exact training request')
        seen.add(identity)
        if row['task'] == 'ocr':
            continue
        if (row.get('status') != 'OBSERVED_GENERATION_WORK'
                or digest(row['effective_generation']) != row['effective_generation_sha256']):
            raise ValueError('Actual generation work and bound effective policy required')
        integer(row['actual_input_tokens'], 'actual training input tokens', minimum=1)
        if not isinstance(row['input_content_sha256'], str) or not row['input_content_sha256']:
            raise ValueError('Public input-content identity required')
        groups[tuple(row[k] for k in EXACT_KEY_FIELDS)].append(row)
    profiles = []
    for key, rows in sorted(groups.items()):
        by_capture = defaultdict(list)
        for row in rows:
            by_capture[row['source']].append(row)
        supported = len(by_capture) >= 2
        weighted, support = defaultdict(Fraction), defaultdict(set)
        for capture, members in sorted(by_capture.items()):
            weight = Fraction(1, len(by_capture) * len(members))
            for row in members:
                outcome = tuple(row[k] for k in WORK_FIELDS)
                weighted[outcome] += weight
                support[outcome].add(capture)
        scenarios = [dict(zip(WORK_FIELDS, outcome), weight=float(weight),
                         independent_capture_count=len(support[outcome]),
                         distinct_input_content_count=1, replicated_outcome=len(support[outcome]) >= 2)
                     for outcome, weight in sorted(weighted.items())] if supported else []
        profiles.append(dict(zip(EXACT_KEY_FIELDS, key), effective_generation=rows[0]['effective_generation'],
            status='EXACT_CONTENT_SCENARIOS_AVAILABLE' if supported else 'UNKNOWN',
            reason=None if supported else 'TWO_INDEPENDENT_CAPTURES_FOR_THIS_EXACT_CONTENT_AND_INPUT_SHAPE_REQUIRED',
            observations=len(rows), capture_ids=sorted(by_capture),
            pool_ids=sorted({row['pool_id'] for row in rows}), scenarios=scenarios))
    eligible = sum(p['status'] == 'EXACT_CONTENT_SCENARIOS_AVAILABLE' for p in profiles)
    return dict(schema=SCHEMA, kind='training_exact_public_content_length_and_work_hypothesis',
        status=('FITTED' if eligible == len(profiles) else 'FITTED_PARTIAL') if eligible else 'UNKNOWN',
        profiles=profiles, supported_profiles=eligible, observed_profiles=len(profiles),
        input_identity_used_as_prediction_feature=True, minimum_independent_captures_per_content=2,
        weighting='Equal independent captures, then measured requests within each exact-content/capture cell.',
        scope='New requests of the exact known public input content, native projection, protocol, policy and tokenized input size.',
        unseen_input_generalization_claimed=False, silent_family_task_fallback=False,
        collector_invariance_is_hypothesis=True, physical_accuracy_validated=False,
        confidence_bound=False, validation_started=False,
        CPU_fixture=any(p['kind'] == 'cpu_fixture' for p in observations))


def fit_artifact(base_model_path, output, *, historical_resolution_path=None):
    base_ref = receipt(base_model_path)
    base = checked(base_ref)
    if base.get('schema') != BASE_SCHEMA or base.get('CPU_fixture'):
        raise ValueError('Unchanged real v4 generation-work training model required')
    source = base['training_observations']
    observations = checked(source)['observations']
    resolution = receipt(historical_resolution_path) if historical_resolution_path else None
    if resolution:
        checked(resolution)
    closure = resolve_closure([base_ref['path']], resolution['path'] if resolution else None)
    result = fit(observations)
    result.update(base_model=base_ref, training_observations=source, historical_resolution=resolution,
        training_evidence_closure=closure,
        implementation_sources=[receipt(__file__), receipt(Path(__file__).with_name('generation_work_v4.py')),
                                receipt(Path(__file__).with_name('historical_evidence_v4.py'))],
        base_v4_model_rewritten=False, training_observations_rewritten=False)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'model.json', result)
    return result


class ExactContentGenerationWorkProvider:
    def __init__(self, document, native_projection, *, allow_exact_content_hypothesis=False,
                 allow_cpu_fixture=False):
        if not allow_exact_content_hypothesis:
            raise ValueError('Explicit exact-content generation hypothesis opt-in required')
        if document.get('schema') != SCHEMA or (document.get('CPU_fixture') and not allow_cpu_fixture):
            raise ValueError('Qualified exact-content generation-work model required')
        if not document.get('CPU_fixture'):
            if not document.get('implementation_sources'):
                raise ValueError('Immutable exact-content implementation provenance required')
            for ref in document['implementation_sources']:
                if receipt(ref['path']) != ref:
                    raise ValueError('Changed exact-content evaluator')
            base = checked(document['base_model'])
            if base.get('schema') != BASE_SCHEMA or base.get('CPU_fixture'):
                raise ValueError('Actual source-bound v4 generation work required')
            if base['training_observations'] != document['training_observations']:
                raise ValueError('Exact provider must retain original training observations')
            observations = checked(document['training_observations'])['observations']
            expected = fit(observations)
            if any(document.get(key) != value for key, value in expected.items()):
                raise ValueError('Exact model differs from immutable actual training fit')
            resolution = document['historical_resolution']
            if resolution:
                checked(resolution)
            closure = resolve_closure([document['base_model']['path']], resolution['path'] if resolution else None)
            if closure != document['training_evidence_closure']:
                raise ValueError('Exact generation training evidence closure changed')
        self.document = copy.deepcopy(document)
        self.native_projection_sha256 = native_projection
        self.physical_evidence = not document.get('CPU_fixture')

    def scenarios(self, request):
        allowed = {'family', 'task', 'protocol_sha256', 'effective_generation', 'input_tokens',
                   'native_execution_projection_sha256', 'request_id', 'input_content_sha256'}
        if set(request) - allowed:
            raise ValueError('Only public request fields allowed; future actual outputs forbidden')
        if request['native_execution_projection_sha256'] != self.native_projection_sha256:
            raise ValueError('Native execution projection mismatch')
        integer(request['input_tokens'], 'public input tokens', minimum=1)
        key = (self.native_projection_sha256, request['family'], request['task'], request['protocol_sha256'],
               digest(request['effective_generation']), request['input_content_sha256'], request['input_tokens'])
        profiles = [p for p in self.document['profiles'] if tuple(p[k] for k in EXACT_KEY_FIELDS) == key]
        if len(profiles) != 1 or profiles[0]['status'] != 'EXACT_CONTENT_SCENARIOS_AVAILABLE':
            raise UnknownEvidence('NO_TWO_CAPTURE_EXACT_CONTENT_GENERATION_WORK_PROFILE_NO_FALLBACK')
        scenarios = copy.deepcopy(profiles[0]['scenarios'])
        for scenario in scenarios:
            batch = scenario['forward_batch_size']
            scenario['prefill'] = dict(forward_count=1, input_sequence_length=request['input_tokens'], input_batch_size=batch)
            scenario['decode'] = dict(forward_count=scenario['decode_forward_count'], input_sequence_length=1, input_batch_size=batch)
            scenario['remaining_forward_count'] = scenario['forward_count']
            scenario['remaining_prefill_forward_count'] = 1
            scenario['remaining_decode_forward_count'] = scenario['decode_forward_count']
        return dict(known=True, scenarios=scenarios, observed_prefix_tokens=0,
            completion_already_observed=False, prefix_source=None,
            input_content_sha256=request['input_content_sha256'],
            scope='EXACT_KNOWN_PUBLIC_CONTENT_NEW_REQUEST_GENERATION_HYPOTHESIS',
            future_actual_output_used=False, silent_family_task_fallback=False,
            unseen_input_generalization_claimed=False, physical_accuracy_validated=False, confidence_bound=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--historical-resolution')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = fit_artifact(args.base_model, args.output, historical_resolution_path=args.historical_resolution)
    print(json.dumps(dict(status=result['status'], supported=result['supported_profiles'], profiles=result['observed_profiles'])))


if __name__ == '__main__':
    main()
