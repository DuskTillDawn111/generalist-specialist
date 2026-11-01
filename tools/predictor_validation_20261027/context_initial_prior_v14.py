"""Capture-balanced joint initial-context scenarios, retaining missing mass.

Conditioning uses only the first pending frontier and public execution setup.
Future job composition/count, future SAVE endpoints and future residual work
are not prediction features. Service-ready uncertainty remains an interval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .context_initial_transition_v13 import SCHEMA as SUPERVISION_SCHEMA
from .multi_native_composite_v5 import checked
from .native_collect import receipt, save
from .profiling import digest

SCHEMA = 'gsdeploy.capture_balanced_joint_initial_context_prior.v14'
CORE_SETUP_FIELDS = ('residents', 'execution', 'threads', 'device')
OBSERVER_SETUP_FIELDS = ('measurement_domain', 'profiler', 'capture_mode')
SETUP_FIELDS = (*CORE_SETUP_FIELDS, *OBSERVER_SETUP_FIELDS)
POLICY = dict(weighting='Equal supporting original captures, then equal matching job observations within each capture',
    exact_conditioning='First pending owner/task/input/phase frontier and public setup only',
    optional_backoff=['First pending owner/task set and public setup', 'Public setup only'],
    unknown_holder_mass_preserved=True, unknown_readiness_mass_preserved=True,
    whole_future_job_composition_used=False, numeric_readiness_choice_fitted=False)


def public_setup(context):
    if any(key not in context for key in SETUP_FIELDS):
        raise ValueError('Complete public execution and observer setup required')
    return {key:copy.deepcopy(context[key]) for key in SETUP_FIELDS}


def frontier_key(frontier, *, coarse=False):
    if not frontier:
        raise ValueError('An explicit nonempty first pending frontier is required')
    result = []
    owners = set()
    for row in frontier:
        if (not isinstance(row, list) or len(row) != 4 or any(not isinstance(v, str) or not v for v in row[:3])
                or not isinstance(row[3], list) or not row[3]
                or any(not isinstance(v, str) or not v for v in row[3]) or len(set(row[3])) != len(row[3])
                or row[0] in owners):
            raise ValueError('Distinct public first-pending owner/task/input/phase rows required')
        owners.add(row[0])
        result.append(row[:2] if coarse else [*row[:3], sorted(row[3])])
    return digest(sorted(result))


def scenario_from_observation(row, setup):
    known = row['status'] == 'OBSERVED_PAIRED_CONTEXT_STATE'
    if known:
        age = row['context_age_ns']
        if (type(age) is not int or age < 0 or age != row['clock_ns']-row['restore_start_ns']
                or row['held_owner'] not in setup['residents']):
            raise ValueError('Initial holder/original restore age does not match source')
        bounds = row['service_ready_bounds']
        upper = bounds['upper_ns']
        if bounds.get('exact') is not False or bounds['lower_ns'] != row['restore_start_ns']:
            raise ValueError('Source interval-censored service readiness required')
        if upper is not None and (type(upper) is not int or upper < bounds['lower_ns']):
            raise ValueError('Owned-kernel service upper witness cannot precede RESTORE_START')
        delay_upper = max(0, upper-row['clock_ns']) if upper is not None else None
        readiness = 'WITNESSED_READY_BY_QUERY' if delay_upper == 0 else 'INTERVAL_CENSORED' if delay_upper is not None else 'NO_SERVICE_UPPER_WITNESS'
    else:
        if row['status'] not in ('UNKNOWN_INITIAL_CONTEXT_GAP', 'UNKNOWN_INITIAL_CONTEXT_UNPAIRED_EVENT'):
            raise ValueError('Explicit source missing-initial-context reason required')
        if row['held_owner'] is not None or row['context_age_ns'] is not None:
            raise ValueError('Unknown initial observation must not contain a fabricated holder/age')
        age, delay_upper, readiness = None, None, 'UNKNOWN_HOLDER_AND_READINESS'
    return dict(scenario_id=digest(dict(capture=row['capture'], job=row['job_id'], clock=row['clock_ns'])),
        capture=row['capture'], source_job_id=row['job_id'], source_job_name=row['job_name'],
        first_pending_frontier=copy.deepcopy(row['pending_frontier']), public_setup=setup,
        observation_status=row['status'], held_owner=row['held_owner'], context_age_ns=age,
        service_ready_delay_bounds_ns=dict(lower_ns=0 if known else None, upper_ns=delay_upper),
        readiness_status=readiness, holder_identified=known,
        service_ready_point_identified=known and delay_upper == 0)


class InitialContextPrior:
    def __init__(self, model, *, allow_cpu_fixture=False, exclude_training_capture=None,
            allow_single_capture_training_diagnostic=False):
        if (model.get('schema') != SCHEMA or model.get('training_only') is not True
                or model.get('model_sha256') != digest({k:v for k,v in model.items() if k != 'model_sha256'})
                or model.get('policy') != POLICY):
            raise ValueError('Source-bound joint initial-context candidate required')
        if model.get('CPU_fixture'):
            if allow_cpu_fixture is not True:
                raise ValueError('CPU fixture does not qualify a measured initial prior')
        else:
            if receipt(model['implementation']['path']) != model['implementation']:
                raise ValueError('Initial prior implementation changed')
            checked(model['source_supervision'])
        captures = {r['capture'] for r in model['scenarios']}
        if exclude_training_capture is not None and (
                allow_single_capture_training_diagnostic is not True or exclude_training_capture not in captures):
            raise ValueError('Explicit whole-training-capture diagnostic exclusion required')
        self.rows = [r for r in model['scenarios'] if r['capture'] != exclude_training_capture]
        self.model_sha256 = model['model_sha256']
        self.excluded = exclude_training_capture

    def predict(self, *, first_pending_frontier, public_execution_setup,
            allow_unvalidated_frontier_pooling=False, allow_unvalidated_observer_transfer=False):
        if set(public_execution_setup) != set(SETUP_FIELDS):
            raise ValueError('Only the explicit public setup projection is accepted, not future job/count fields')
        exact, coarse = frontier_key(first_pending_frontier), frontier_key(first_pending_frontier, coarse=True)
        candidates = [r for r in self.rows if all(r['public_setup'][key] == public_execution_setup[key]
            for key in CORE_SETUP_FIELDS)]
        if not candidates:
            raise UnknownEvidence('INITIAL_PRIOR_PUBLIC_EXECUTION_SETUP_UNSUPPORTED')
        observer_transfer = any(any(r['public_setup'][key] != public_execution_setup[key]
            for key in OBSERVER_SETUP_FIELDS) for r in candidates)
        if observer_transfer and allow_unvalidated_observer_transfer is not True:
            raise UnknownEvidence('INITIAL_PRIOR_OBSERVER_TRANSFER_REQUIRES_EXPLICIT_HYPOTHESIS')
        if not observer_transfer:
            candidates = [r for r in candidates if all(r['public_setup'][key] == public_execution_setup[key]
                for key in OBSERVER_SETUP_FIELDS)]
        matching = [r for r in candidates if frontier_key(r['first_pending_frontier']) == exact]
        tier = 'EXACT_FIRST_PENDING_FRONTIER'
        if not matching and allow_unvalidated_frontier_pooling is True:
            matching = [r for r in candidates if frontier_key(r['first_pending_frontier'], coarse=True) == coarse]
            tier = 'POOLED_INPUT_AND_PHASE_HYPOTHESIS'
            if not matching:
                matching, tier = candidates, 'POOLED_FRONTIER_HYPOTHESIS'
        if not matching:
            raise UnknownEvidence('INITIAL_PRIOR_FIRST_FRONTIER_UNSEEN_EXPLICIT_POOLING_HYPOTHESIS_REQUIRED')
        captures = defaultdict(list)
        for row in matching:
            captures[row['capture']].append(row)
        scenarios = []
        for capture, rows in sorted(captures.items()):
            weight = 1/(len(captures)*len(rows))
            for row in rows:
                # Keep each joint holder/age/readiness observation intact.
                scenarios.append(dict(copy.deepcopy(row), probability=weight))
        unknown_holder = sum(r['probability'] for r in scenarios if not r['holder_identified'])
        unknown_readiness = sum(r['probability'] for r in scenarios if not r['service_ready_point_identified'])
        return dict(status='UNCALIBRATED_JOINT_INITIAL_CONTEXT_PRIOR', scenarios=scenarios,
            probability_sum=sum(r['probability'] for r in scenarios),
            unknown_holder_mass=unknown_holder, unresolved_point_readiness_mass=unknown_readiness,
            supporting_capture_ids=sorted(captures), conditioning_tier=tier,
            observer_transfer_hypothesis=observer_transfer, model_sha256=self.model_sha256,
            excluded_training_capture=self.excluded, independent_validation=False,
            missing_mass_renormalized_away=False, first_pending_owner_used_as_default_holder=False,
            whole_future_job_composition_used=False, complete_initialization_decision=False,
            assumptions=['Training jobs/captures are conditionally exchangeable for this first frontier and public setup',
                'Observed missing-holder mass transfers with its missingness mechanism; no MAR assumption removes it',
                'Any numerical resolution of a nonpoint readiness interval is an additional declared scenario hypothesis',
                'Profiler/observer and feature pooling transfer require the explicitly requested hypotheses'])


def realize_initial_scenario(scenario, *, query_clock_ns, readiness_rule):
    """Explicitly named timing scenario, never an implicit complete oracle."""
    if type(query_clock_ns) is not int or query_clock_ns < 0:
        raise ValueError('Nonnegative integer prospective query clock required')
    if not scenario['holder_identified']:
        raise UnknownEvidence('INITIAL_SCENARIO_HOLDER_UNIDENTIFIED_MASS_MUST_NOT_BE_DROPPED')
    bounds = scenario['service_ready_delay_bounds_ns']
    if readiness_rule == 'witnessed_point_only':
        if not scenario['service_ready_point_identified']:
            raise UnknownEvidence('INITIAL_SCENARIO_SERVICE_READY_NOT_POINT_IDENTIFIED')
        delay, hypothesis = 0, False
    elif readiness_rule == 'lower_bound_hypothesis':
        delay, hypothesis = bounds['lower_ns'], not scenario['service_ready_point_identified']
    elif readiness_rule == 'upper_bound_hypothesis':
        if bounds['upper_ns'] is None:
            raise UnknownEvidence('INITIAL_SCENARIO_SERVICE_UPPER_BOUND_UNAVAILABLE')
        delay, hypothesis = bounds['upper_ns'], not scenario['service_ready_point_identified']
    else:
        raise ValueError('An explicit witnessed/lower/upper readiness rule is required')
    return dict(held_owner=scenario['held_owner'], context_age_ns=scenario['context_age_ns'],
        restore_start_ns=query_clock_ns-scenario['context_age_ns'], service_ready_delay_ns=delay,
        readiness_rule=readiness_rule, readiness_hypothesis_used=hypothesis,
        source_scenario_id=scenario['scenario_id'], physical_SM_service_identified=False)


def train(supervision_path, output):
    source = receipt(supervision_path)
    supervision = checked(source)
    if (supervision.get('schema') != SUPERVISION_SCHEMA or supervision.get('training_only') is not True
            or supervision.get('model_sha256') != digest({k:v for k,v in supervision.items() if k != 'model_sha256'})):
        raise ValueError('Source-bound measured training initial observations required')
    from .context_initial_transition_v13 import extract
    regenerated = extract([audit['report'] for audit in supervision['training_audits']])
    if regenerated != supervision:
        raise ValueError('Initial supervision differs from complete original-source rederivation')
    setups = {}
    for audit in supervision['training_audits']:
        plan, context = checked(audit['plan']), checked(audit['context'])
        if plan.get('split') != 'training' or plan.get('training_only') is not True:
            raise ValueError('Explicit training initial prior sources required')
        setups[audit['raw']['sha256']] = public_setup(context)
    for ref in supervision['sources']:
        if receipt(ref['path']) != ref:
            raise ValueError('Initial prior source changed')
    rows = [scenario_from_observation(r, setups[r['capture']]) for r in supervision['initial_observations']]
    model = dict(schema=SCHEMA, training_only=True, scenarios=rows, policy=POLICY,
        source_supervision=source, implementation=receipt(__file__),
        initial_observations_preserved=len(rows), unknown_holder_rows=sum(not r['holder_identified'] for r in rows),
        source_supervision_completely_rederived=True,
        numeric_readiness_policy_fitted=False, independent_accuracy_validated=False,
        whole_future_job_composition_used=False, GPU_started=False)
    model['model_sha256'] = digest(model)
    folds = []
    for excluded in sorted(setups):
        prior = InitialContextPrior(model, exclude_training_capture=excluded,
            allow_single_capture_training_diagnostic=True)
        tested = [r for r in rows if r['capture'] == excluded]
        predictions = []
        for row in tested:
            predicted = prior.predict(first_pending_frontier=row['first_pending_frontier'],
                public_execution_setup=row['public_setup'], allow_unvalidated_frontier_pooling=True)
            known_scenarios = [s for s in predicted['scenarios'] if s['holder_identified']]
            known_weight = sum(s['probability'] for s in known_scenarios)
            owner_probability = sum(s['probability'] for s in known_scenarios if s['held_owner'] == row['held_owner']) if row['holder_identified'] else None
            ages = [s['context_age_ns'] for s in known_scenarios if s['held_owner'] == row['held_owner']] if row['holder_identified'] else []
            predicted_mean = sum(s['probability']*s['context_age_ns'] for s in known_scenarios)/known_weight if known_weight else None
            predictions.append(dict(source_scenario_id=row['scenario_id'], observed_holder=row['held_owner'],
                observed_holder_known=row['holder_identified'], true_age_ns=row['context_age_ns'],
                predicted_true_holder_probability=owner_probability,
                true_age_in_same_holder_empirical_range=min(ages) <= row['context_age_ns'] <= max(ages) if ages else False,
                paired_only_age_mean_absolute_error_ns=abs(predicted_mean-row['context_age_ns']) if predicted_mean is not None and row['holder_identified'] else None,
                conditioning_tier=predicted['conditioning_tier'], predicted_unknown_holder_mass=predicted['unknown_holder_mass'],
                predicted_unresolved_point_readiness_mass=predicted['unresolved_point_readiness_mass'],
                full_prior=predicted))
        known = [p for p in predictions if p['observed_holder_known']]
        folds.append(dict(excluded_capture=excluded, test_rows=len(tested), predicted_rows=len(predictions),
            observed_paired_rows=len(known), observed_missing_holder_rows=len(tested)-len(known),
            mean_true_holder_unconditional_probability=sum(p['predicted_true_holder_probability'] for p in known)/len(known),
            same_holder_age_empirical_range_coverage=sum(p['true_age_in_same_holder_empirical_range'] for p in known)/len(known),
            paired_only_age_mean_MAE_ns=sum(p['paired_only_age_mean_absolute_error_ns'] for p in known if p['paired_only_age_mean_absolute_error_ns'] is not None)/len(known),
            interpretation='All initial rows retained. Missing physical holder labels are not scored as known outcomes. Paired-only age means are descriptive and require missingness assumptions for broader inference.',
            predictions=predictions, independent_validation=False))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output/'model.json', model)
    by_capture = defaultdict(list)
    for row in rows:
        by_capture[row['capture']].append(row)
    unconditional_unknown_mass = sum(sum(not r['holder_identified'] for r in members)/len(members)
        for members in by_capture.values())/len(by_capture)
    result = dict(schema='gsdeploy.initial_context_prior_grouped_training_diagnostic.v14',
        status='JOINT_PRIOR_CANDIDATE_TRAINING_ONLY_WITH_UNKNOWN_MASS', model=receipt(output/'model.json'),
        initial_rows=len(rows), known_holder_rows=sum(r['holder_identified'] for r in rows),
        unknown_holder_rows=sum(not r['holder_identified'] for r in rows),
        point_readiness_witness_rows=sum(r['service_ready_point_identified'] for r in rows),
        unknown_point_readiness_rows=sum(not r['service_ready_point_identified'] for r in rows),
        unconditional_capture_balanced_unknown_holder_mass=unconditional_unknown_mass,
        source_supervision_completely_rederived=True,
        folds=folds, conditioning_uses_future_job_composition=False,
        missing_holder_mass_removed=False, physical_SM_service_identified=False,
        independent_validation=False, full_initialization_oracle=False, GPU_started=False)
    save(output/'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervision', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = train(args.supervision, args.output)
    print(json.dumps(dict(status=result['status'], initial_rows=result['initial_rows'],
        known_holder_rows=result['known_holder_rows'], unknown_holder_rows=result['unknown_holder_rows'],
        point_readiness_witness_rows=result['point_readiness_witness_rows'],
        folds=[{k:v for k,v in f.items() if k != 'predictions'} for f in result['folds']],
        report=receipt(Path(args.output)/'report.json'))))


if __name__ == '__main__':
    main()
