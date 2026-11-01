"""Source-bound empirical generation work; no future completion is an input.

The discrete length/work distribution is a declared training hypothesis, not a
tail bound. Inputs and captures receive equal weight within their respective
levels, so extra replicas do not create extra independent input support.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .native_collect import receipt, save
from .native_execution_projection_v5 import native_execution_projection
from .profiling import RESIDENTS, digest
from .smoke_verifier_v3 import check_receipt

SCHEMA = 'gsdeploy.empirical_generation_work_hypothesis.v4'
PREFIX_SCHEMA = 'gsdeploy.observed_generation_prefix.v4'
STRICT_SCHEMAS = ('gsdeploy.training_collection_strict_review.v3',
                  'gsdeploy.startup_training_collection_strict_review.v4')
KEY_FIELDS = ('native_execution_projection_sha256', 'family', 'task',
              'protocol_sha256', 'effective_generation_sha256')
WORK_FIELDS = ('generated_tokens', 'finish_reason', 'forward_count',
               'prefill_forward_count', 'decode_forward_count',
               'forward_batch_size', 'selected_token_forward_relation')


def integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer ' + name)
    return value


def actual_work(response, request_id):
    """Check real completion metadata against all observed language forwards.

    A selected beam can terminate before the other beams: its returned token
    count must not silently replace the separately observed search work.
    """
    policy = response['effective_generation']
    budget = integer(policy['max_new_tokens'], 'generation budget', minimum=1)
    beams = integer(policy.get('num_beams', 1), 'internal beam batch', minimum=1)
    generated = integer(response['generated_tokens'], 'actual generated tokens', minimum=1)
    if generated > budget:
        raise ValueError('Actual generated tokens exceed policy budget')
    eos = response['eos_token_ids']
    if (not isinstance(eos, list) or not eos or any(type(t) is not int or t < 0 for t in eos)
            or len(set(eos)) != len(eos)):
        raise ValueError('Explicit actual EOS IDs required')
    policy_eos = policy.get('eos_token_id')
    policy_eos = [policy_eos] if type(policy_eos) is int else policy_eos
    if policy_eos != eos:
        raise ValueError('Effective and observed EOS policy differ')
    terminal = integer(response['terminal_token_id'], 'actual terminal token')
    reason = response['finish_reason']
    if reason == 'eos':
        if terminal not in eos:
            raise ValueError('EOS completion without an actual EOS terminal')
    elif reason == 'length':
        if generated != budget or terminal in eos:
            raise ValueError('Length completion must actually exhaust the budget')
    else:
        raise ValueError('Only actual EOS/length autoregressive completion is qualified')
    semantic = response['semantic_forward_observations']
    forwards = semantic['records']
    if (semantic.get('schema') != 'gsdeploy.semantic_forward_observation.v1'
            or semantic.get('request_id') != request_id or semantic.get('instrumentation_errors')
            or not forwards):
        raise ValueError('Complete actual semantic forward observation required')
    if len({f['id'] for f in forwards}) != len(forwards):
        raise ValueError('Duplicate language forward')
    input_tokens = integer(response['input_tokens'], 'actual input tokens', minimum=1)
    for index, forward in enumerate(forwards):
        phase = 'prefill' if index == 0 else 'decode'
        sequence = input_tokens if index == 0 else 1
        if (forward.get('forward_index') != index or forward.get('semantic_phase') != phase
                or forward.get('input_sequence_length') != sequence
                or forward.get('input_batch_size') != beams
                or forward.get('input_token_positions') != sequence * beams
                or forward.get('end_basis') != 'always_call_forward_hook'):
            raise ValueError('Actual prefill/decode order, dimensions or internal beam batch mismatch')
    if (beams == 1 and len(forwards) != generated) or len(forwards) < generated:
        raise ValueError('Actual selected tokens and language forward count disagree')
    return dict(generated_tokens=generated, finish_reason=reason,
                forward_count=len(forwards), prefill_forward_count=1,
                decode_forward_count=len(forwards) - 1, forward_batch_size=beams,
                selected_token_forward_relation=('one_forward_per_selected_token'
                    if len(forwards) == generated else 'beam_search_extra_steps'),
                actual_input_tokens=input_tokens, effective_generation=copy.deepcopy(policy))


def input_content_identity(pool):
    """Count actual public content rather than sample aliases or references."""
    sample, task = pool['sample'], pool['task_type']
    if task == 'text':
        content = dict(task=task, context=sample['context'], question=sample['question'])
    elif task in ('image', 'asr', 'ocr'):
        content = dict(task=task, input_sha256=sample['input_sha256'])
    else:
        raise ValueError('Unsupported training input task')
    return digest(content)


def load_training(path):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if (report.get('schema') not in STRICT_SCHEMAS or report.get('status') != 'PASS'
            or report.get('issues') or not report.get('training_only')):
        raise ValueError('Strict PASS training collection review required')
    refs = [report[k] for k in ('raw', 'plan', 'context')] + report.get('evaluator_sources', [])
    if len(refs) == 3 or not all(check_receipt(ref) for ref in refs):
        raise ValueError('Changed or missing strict raw/plan/context/evaluator receipt')
    plan = json.loads(Path(report['plan']['path']).read_text())
    raw = json.loads(Path(report['raw']['path']).read_text())
    context = json.loads(Path(report['context']['path']).read_text())
    projection = native_execution_projection(context)
    full_context = projection['original_full_context_sha256']
    run = 'raw-sha256:' + report['raw']['sha256']
    if (plan.get('domain') != 'natural' or projection['measurement_domain'] != 'natural'
            or plan.get('split') != 'training' or raw.get('kind') != 'measured'
            or raw.get('status') != 'completed' or raw.get('not_for_training', False)
            or any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors'))
            or plan.get('execution_context_sha256') != full_context
            or raw.get('execution_context_sha256') != full_context
            or report.get('capture_run_id') != run):
        raise ValueError('Actual measured exclusively training natural capture required')
    frozen_pools = [s['frozen'] for s in context['execution_sources']
                    if s['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(frozen_pools) != 1 or not check_receipt(frozen_pools[0]):
        raise ValueError('Exact immutable public input catalog required')
    pools = {p['pool_id']: p for p in map(json.loads, Path(frozen_pools[0]['path']).read_text().splitlines())}
    expected = Counter()
    for job in plan['jobs']:
        if job.get('split') != 'training':
            raise ValueError('Validation job cannot train a generation model')
        expected.update((job['id'], alias, pool, job['tasks'][alias]) for alias, pool in job['inputs'].items())
    measured = raw['request_records']  # Never use priming_request_records.
    observed = Counter((r['profile_job_id'], r['model'], r['pool_id'], r['task']) for r in measured)
    if expected != observed or len({r['request_id'] for r in measured}) != len(measured):
        raise ValueError('Exact nonduplicated measured training request cohort required')
    observations = []
    for row in measured:
        if row.get('split') != 'training' or row.get('status') != 'completed':
            raise ValueError('Completed training request required')
        response = row['worker_response']
        family = RESIDENTS[row['model']]
        pool = pools[row['pool_id']]
        if pool['task_type'] != row['task'] or response.get('task') != row['task']:
            raise ValueError('Actual task differs from immutable input')
        observation = dict(kind='measured', split='training', source=run,
            request_id=row['request_id'], alias=row['model'], family=family,
            task=row['task'], pool_id=row['pool_id'], input_content_sha256=input_content_identity(pool),
            execution_context_sha256=full_context,
            native_execution_projection_sha256=projection['projection_sha256'])
        if row['task'] == 'ocr':
            observation.update(status='UNKNOWN_NON_AUTOREGRESSIVE_OCR', actual_finish_reason=response.get('finish_reason'))
        else:
            observation.update(actual_work(response, row['request_id']), status='OBSERVED_GENERATION_WORK',
                protocol_sha256=response['protocol_sha256'], request_sha256=response['request_sha256'])
            observation['effective_generation_sha256'] = digest(observation['effective_generation'])
            if row['task'] == 'asr':
                observation['public_audio_duration_s'] = pool['sample']['duration_s']
        observations.append(observation)
    return observations, dict(review=receipt(path), raw=report['raw'], plan=report['plan'],
        context=report['context'], run_id=run, projection=projection,
        input_catalog=frozen_pools[0], evaluator_sources=report['evaluator_sources'])


def fit(observations, *, allow_cpu_fixture=False):
    observations = copy.deepcopy(list(observations))
    seen, groups, ocr = set(), defaultdict(list), []
    for row in observations:
        if (row.get('kind') != 'measured' and not (allow_cpu_fixture and row.get('kind') == 'cpu_fixture')) or row.get('split') != 'training':
            raise ValueError('Measured training observations required')
        identity = row['source'], row['request_id']
        if identity in seen:
            raise ValueError('Duplicate actual training request')
        seen.add(identity)
        if row['task'] == 'ocr':
            ocr.append(row)
            continue
        if row.get('status') != 'OBSERVED_GENERATION_WORK' or digest(row['effective_generation']) != row['effective_generation_sha256']:
            raise ValueError('Source-bound actual work and complete policy required')
        groups[tuple(row[k] for k in KEY_FIELDS)].append(row)
    profiles = []
    for key, rows in sorted(groups.items()):
        cells = defaultdict(list)
        for row in rows:
            cells[row['input_content_sha256'], row['source']].append(row)
        inputs = sorted({k[0] for k in cells})
        captures = sorted({r['source'] for r in rows})
        supported = len(inputs) >= 2 and len(captures) >= 2
        weighted, outcome_rows = defaultdict(Fraction), defaultdict(list)
        for (content, run), members in sorted(cells.items()):
            input_captures = {c for i, c in cells if i == content}
            weight = Fraction(1, len(inputs) * len(input_captures) * len(members))
            for row in members:
                outcome = tuple(row[k] for k in WORK_FIELDS)
                weighted[outcome] += weight
                outcome_rows[outcome].append(row)
        scenarios = []
        if supported:
            for outcome, weight in sorted(weighted.items()):
                members = outcome_rows[outcome]
                scenarios.append(dict(zip(WORK_FIELDS, outcome), weight=float(weight),
                    independent_capture_count=len({r['source'] for r in members}),
                    distinct_input_content_count=len({r['input_content_sha256'] for r in members}),
                    replicated_outcome=len({r['source'] for r in members}) >= 2))
        profiles.append(dict(zip(KEY_FIELDS, key), effective_generation=rows[0]['effective_generation'],
            status='EMPIRICAL_SCENARIOS_AVAILABLE' if supported else 'UNKNOWN',
            reason=None if supported else 'TWO_INDEPENDENT_CAPTURES_AND_TWO_DISTINCT_INPUT_CONTENTS_REQUIRED',
            observations=len(rows), capture_ids=captures, distinct_input_content_count=len(inputs),
            per_input_support=[dict(input_content_sha256=i,
                capture_ids=sorted({c for content, c in cells if content == i}),
                pool_ids=sorted({r['pool_id'] for r in rows if r['input_content_sha256'] == i})) for i in inputs],
            scenarios=scenarios))
    supported_count = sum(p['status'] == 'EMPIRICAL_SCENARIOS_AVAILABLE' for p in profiles)
    return dict(schema=SCHEMA, kind='training_empirical_joint_length_work_hypothesis',
        status=('FITTED' if supported_count == len(profiles) else 'FITTED_PARTIAL') if supported_count else 'UNKNOWN',
        profiles=profiles, non_autoregressive_ocr=dict(status='UNKNOWN_NON_AUTOREGRESSIVE_OCR', observations=len(ocr)),
        observation_count=len(observations), input_identity_used_as_prediction_feature=False,
        weighting='Equal input-content weights, then equal independent captures within input, then measured requests within each cell.',
        assumptions=['Native projection collector invariance is hypothetical; full contexts remain separate.',
            'Future family/task lengths and joint beam work follow this frozen empirical training distribution.',
            'Semantic Python forward dimensions describe work, not calibrated GPU phases or verified KV reuse.'],
        unseen_length_extrapolation=False, policy_maximum_substituted_for_actual_length=False,
        asr_duration_regression=dict(status='UNKNOWN_INSUFFICIENT_DISTINCT_DURATION_SUPPORT', fitted=False),
        physical_accuracy_validated=False, confidence_bound=False, validation_started=False,
        CPU_fixture=any(r.get('kind') == 'cpu_fixture' for r in observations))


class GenerationWorkProvider:
    def __init__(self, document, native_projection_sha256, *, allow_empirical_length_hypothesis=False,
                 allow_cpu_fixture=False):
        if not allow_empirical_length_hypothesis:
            raise ValueError('Explicit empirical length hypothesis opt-in required')
        if document.get('schema') != SCHEMA or (document.get('CPU_fixture') and not allow_cpu_fixture):
            raise ValueError('Qualified generation work model required')
        if not document.get('CPU_fixture'):
            refs = document.get('implementation_sources', [])
            refs = refs + [document.get('training_observations')]
            captures = document.get('captures', [])
            if not captures or not refs or not all(check_receipt(ref) for ref in refs):
                raise ValueError('Immutable real training model provenance required')
            for capture in captures:
                bound = [capture[k] for k in ('review', 'raw', 'plan', 'context', 'input_catalog')]
                if not all(check_receipt(ref) for ref in bound + capture['evaluator_sources']):
                    raise ValueError('Changed real generation training evidence')
                context = json.loads(Path(capture['context']['path']).read_text())
                if native_execution_projection(context) != capture['projection']:
                    raise ValueError('Changed native projection or source identity')
        self.document = copy.deepcopy(document)
        self.native_projection_sha256 = native_projection_sha256
        self.physical_evidence = not document.get('CPU_fixture')

    def scenarios(self, request, *, observed_prefix=None, prediction_time_ns=None):
        allowed = {'family', 'task', 'protocol_sha256', 'effective_generation', 'input_tokens',
                   'native_execution_projection_sha256', 'request_id'}
        if set(request) - allowed:
            raise ValueError('Only public request fields allowed; future actual outputs are forbidden')
        if request['native_execution_projection_sha256'] != self.native_projection_sha256:
            raise ValueError('Native execution projection mismatch')
        integer(request['input_tokens'], 'public input tokens', minimum=1)
        key = (self.native_projection_sha256, request['family'], request['task'],
               request['protocol_sha256'], digest(request['effective_generation']))
        profiles = [p for p in self.document['profiles'] if tuple(p[k] for k in KEY_FIELDS) == key]
        if len(profiles) != 1 or profiles[0]['status'] != 'EMPIRICAL_SCENARIOS_AVAILABLE':
            raise UnknownEvidence('NO_SUPPORTED_GENERATION_WORK_PROFILE')
        profile = profiles[0]
        scenarios = copy.deepcopy(profile['scenarios'])
        observed_count, completed = 0, False
        prefix_ref = None
        if observed_prefix is not None:
            # A source-bound prefix artifact is separate from final worker_response.
            if not check_receipt(observed_prefix):
                raise ValueError('Immutable already-observed prefix receipt required')
            prefix_ref = observed_prefix
            prefix = json.loads(Path(observed_prefix['path']).read_text())
            keys = {'schema', 'request_id', 'protocol_sha256', 'native_execution_projection_sha256',
                    'effective_generation_sha256', 'observation_time_ns', 'token_observations', 'status', 'clock'}
            if (set(prefix) != keys or prefix['schema'] != PREFIX_SCHEMA or prefix['status'] != 'OBSERVED_PREFIX'
                    or prefix['clock'] != 'host_perf_counter_ns'
                    or not isinstance(request.get('request_id'), str) or not request['request_id']
                    or prefix['request_id'] != request.get('request_id')
                    or prefix['protocol_sha256'] != request['protocol_sha256']
                    or prefix['native_execution_projection_sha256'] != self.native_projection_sha256
                    or prefix['effective_generation_sha256'] != digest(request['effective_generation'])):
                raise ValueError('Exact source-bound runtime prefix contract required; no final outputs')
            integer(prediction_time_ns, 'prediction time')
            integer(prefix['observation_time_ns'], 'prefix observation time')
            if prefix['observation_time_ns'] > prediction_time_ns:
                raise ValueError('Future prefix observation is forbidden')
            policy = request['effective_generation']
            if policy.get('num_beams', 1) != 1:
                raise UnknownEvidence('BEAM_PREFIX_PATH_OR_FINAL_SEARCH_WORK_NOT_QUALIFIED')
            eos = policy['eos_token_id']
            eos = [eos] if type(eos) is int else eos
            previous = -1
            tokens = prefix['token_observations']
            for index, token in enumerate(tokens):
                if set(token) != {'token_id', 'observed_time_ns'}:
                    raise ValueError('Only actual timestamped prefix tokens allowed')
                integer(token['token_id'], 'observed token')
                integer(token['observed_time_ns'], 'observed token time')
                if not previous <= token['observed_time_ns'] <= prefix['observation_time_ns']:
                    raise ValueError('Future or unordered token observation')
                previous = token['observed_time_ns']
                if token['token_id'] in eos and index != len(tokens) - 1:
                    raise ValueError('Prefix continues after observed EOS')
            observed_count = len(tokens)
            budget = policy['max_new_tokens']
            if observed_count > budget:
                raise ValueError('Observed prefix exceeds actual policy budget')
            completed = bool(tokens) and (tokens[-1]['token_id'] in eos or observed_count == budget)
            if completed:
                scenarios = [dict(generated_tokens=observed_count,
                    finish_reason='eos' if tokens[-1]['token_id'] in eos else 'length',
                    forward_count=observed_count, prefill_forward_count=1, decode_forward_count=observed_count - 1,
                    forward_batch_size=1, selected_token_forward_relation='one_forward_per_selected_token',
                    weight=1., basis='ALREADY_OBSERVED_GREEDY_PREFIX_COMPLETION')]
            else:
                scenarios = [s for s in scenarios if s['generated_tokens'] > observed_count]
                if not scenarios:
                    raise UnknownEvidence('OBSERVED_PREFIX_EXCEEDS_FROZEN_EMPIRICAL_SUPPORT')
                total = sum(s['weight'] for s in scenarios)
                for scenario in scenarios:
                    scenario['weight'] /= total
        for scenario in scenarios:
            batch = scenario['forward_batch_size']
            scenario['prefill'] = dict(forward_count=1, input_sequence_length=request['input_tokens'], input_batch_size=batch)
            scenario['decode'] = dict(forward_count=scenario['decode_forward_count'], input_sequence_length=1, input_batch_size=batch)
            scenario['remaining_forward_count'] = max(0, scenario['forward_count'] - observed_count)
            scenario['remaining_prefill_forward_count'] = 0 if observed_count else 1
            scenario['remaining_decode_forward_count'] = max(0, scenario['forward_count'] - max(1, observed_count))
        return dict(known=True, scenarios=scenarios, observed_prefix_tokens=observed_count,
            completion_already_observed=completed, prefix_source=prefix_ref,
            scope='FROZEN_TRAINING_EMPIRICAL_LENGTH_AND_JOINT_WORK_HYPOTHESIS',
            future_actual_output_used=False, physical_accuracy_validated=False, confidence_bound=False)


def fit_artifact(reviews, output):
    observations, captures, seen = [], [], set()
    for path in reviews:
        rows, capture = load_training(path)
        if capture['run_id'] in seen:
            raise ValueError('Repeated raw is not an independent capture')
        seen.add(capture['run_id'])
        observations.extend(rows)
        captures.append(capture)
    if not captures:
        raise ValueError('Nonempty strict natural training evidence required')
    model = fit(observations)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'training-observations.json', dict(observations=observations))
    model['captures'] = captures
    model['training_observations'] = receipt(out / 'training-observations.json')
    model['implementation_sources'] = [receipt(__file__), receipt(Path(__file__).with_name('native_execution_projection_v5.py'))]
    for capture in captures:
        refs = [capture[k] for k in ('review', 'raw', 'plan', 'context', 'input_catalog')] + capture['evaluator_sources']
        if not all(check_receipt(ref) for ref in refs):
            raise ValueError('Training evidence changed during fit')
    save(out / 'model.json', model)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = fit_artifact(args.review, args.output)
    print(json.dumps(dict(status=result['status'], profiles=len(result['profiles']),
                         observations=result['observation_count'], output=args.output)))


if __name__ == '__main__':
    main()
