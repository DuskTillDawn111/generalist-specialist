"""Opt-in utility review v3; no changes to the frozen v2 optimizer objective.

Complete successful cohorts only. SLOs and latency scales are independent;
quality/on-time joint delivery is a diagnostic, not a guessed cross-task value.
"""
from collections import Counter
from itertools import combinations
import math

import numpy as np

SCHEMA = 'gsdeploy.utility_tradeoffs.v3'
TERMS = ('Q', 'T', 'L', 'P', 'M', 'E')
WEIGHT_PROFILES = {
    'six_equal_v2': dict.fromkeys(TERMS, 1 / 6),
    'five_groups': dict(Q=.2, T=.2, L=.1, P=.1, M=.2, E=.2),
    'four_groups_v3': dict(Q=.25, T=1 / 12, L=1 / 12, P=1 / 12, M=.25, E=.25),
}


def number(value, label, *, minimum=None, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or minimum is not None and value < minimum or positive and value <= 0):
        raise ValueError('Invalid finite number: ' + label)
    return value


def weights_valid(weights, keys):
    if set(weights) != set(keys) or not weights:
        raise ValueError('Complete weights required')
    for k, value in weights.items():
        number(value, 'weight:' + k, minimum=0)
    if not math.isclose(math.fsum(weights.values()), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError('Weights must sum to one')


def score_terms(terms, weights, *, quality_span_factor=1., energy_scale_factor=1.):
    """Shared positive scale perturbations, with the quality best anchor fixed.

    Factors are sensitivity parameters, not updated empirical reference values.
    Negative normalized Q is valid when quality exceeds the development anchor.
    """
    weights_valid(weights, TERMS)
    if set(terms) != set(TERMS):
        raise ValueError('All six terms required; missing values cannot become zero')
    number(quality_span_factor, 'quality span factor', positive=True)
    number(energy_scale_factor, 'energy scale factor', positive=True)
    norms = {k: number(terms[k], k, minimum=None if k == 'Q' else 0) for k in TERMS}
    if norms['T'] > 1:
        raise ValueError('Deadline miss fraction exceeds one')
    norms['Q'] /= quality_span_factor
    norms['E'] /= energy_scale_factor
    contributions = {k: -weights[k] * norms[k] for k in TERMS}
    value = math.fsum(contributions.values())
    number(value, 'utility')
    return dict(schema=SCHEMA, terms=norms, weights=dict(weights), contributions=contributions,
                U=value, service_contribution=sum(contributions[k] for k in ('T', 'L', 'P')),
                quality_span_factor=quality_span_factor, energy_scale_factor=energy_scale_factor)


def episode_metrics(requests, *, expected_tasks_by_id, task_references, task_weights,
                    episode_joules, energy_scale_joules, owned_peak_mib, memory_budget_mib):
    """Derive loss terms from a complete cohort and explicit fixed references.

    Callers verify source receipts, episode energy coverage and feasibility.
    expected_tasks_by_id must come from the offered plan, not returned outputs.
    Request rows require request_id, task, success, latency_ms, errors, units.
    """
    if not expected_tasks_by_id or not requests:
        raise ValueError('Nonempty offered cohort required')
    tasks = set(task_references)
    if not tasks or not tasks <= {'text', 'image', 'ocr', 'asr'}:
        raise ValueError('Supported task references required')
    weights_valid(task_weights, tasks)
    if set(expected_tasks_by_id.values()) != tasks:
        raise ValueError('Every referenced task must occur in the offered cohort')
    ids = [r['request_id'] for r in requests]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_tasks_by_id):
        raise ValueError('Complete unique offered cohort required')
    for name, value in [('episode joules', episode_joules), ('owned peak', owned_peak_mib)]:
        number(value, name, minimum=0)
    number(energy_scale_joules, 'energy scale', positive=True)
    number(memory_budget_mib, 'memory budget', positive=True)
    groups = {t: [] for t in tasks}
    for row in requests:
        task = row['task']
        if task != expected_tasks_by_id[row['request_id']] or row['success'] is not True:
            raise ValueError('Complete successful task-matched cohort required')
        for k in ('latency_ms', 'errors'):
            number(row[k], k, minimum=0)
        number(row['units'], 'reference units', positive=True)
        if task in ('text', 'image') and (row['units'] != 1 or row['errors'] not in (0, 1)):
            raise ValueError('Binary correctness requires one unit and zero/one error')
        groups[task].append(row)
    per_task = {}
    for task in sorted(tasks):
        ref, rows = task_references[task], groups[task]
        best = number(ref['best_loss'], 'best loss', minimum=0)
        worst = number(ref['worst_loss'], 'worst loss', minimum=0)
        if worst <= best:
            raise ValueError('Positive quality anchor span required')
        scale = number(ref['latency_scale_ms'], 'latency scale', positive=True)
        deadline = number(ref['deadline_ms'], 'deadline', positive=True)
        errors, units = (math.fsum(r[k] for r in rows) for k in ('errors', 'units'))
        latency = [r['latency_ms'] for r in rows]
        timely = sum(r['latency_ms'] <= deadline for r in rows)
        joint = (sum(r['errors'] == 0 and r['latency_ms'] <= deadline for r in rows)
                 if task in ('text', 'image') else None)
        n = len(rows)
        per_task[task] = dict(offered=n, error_total=errors, reference_unit_total=units,
            native_loss=errors / units, normalized_quality_loss=(errors / units - best) / (worst - best),
            mean_ms=float(np.mean(latency)), p95_ms=float(np.percentile(latency, 95)),
            latency_scale_ms=scale, deadline_ms=deadline, timely_count=timely,
            deadline_miss_fraction=1 - timely / n, correct_and_timely_count=joint,
            correct_and_timely_fraction=joint / n if joint is not None else None,
            joint_metric_reason='binary correctness' if joint is not None else
                'Undefined: CER/WER is an edit ratio, not a correctness probability')
    terms = dict(
        Q=math.fsum(task_weights[t] * per_task[t]['normalized_quality_loss'] for t in tasks),
        T=1 - sum(r['timely_count'] for r in per_task.values()) / len(requests),
        L=math.fsum(task_weights[t] * per_task[t]['mean_ms'] / per_task[t]['latency_scale_ms'] for t in tasks),
        P=math.fsum(task_weights[t] * per_task[t]['p95_ms'] / per_task[t]['latency_scale_ms'] for t in tasks),
        M=owned_peak_mib / memory_budget_mib, E=episode_joules / energy_scale_joules)
    return dict(terms=terms, per_task=per_task, offered=len(requests), task_weights=dict(task_weights),
        task_counts=dict(Counter(r['task'] for r in requests)),
        task_weighted_deadline_miss_fraction=math.fsum(task_weights[t] * per_task[t]['deadline_miss_fraction'] for t in tasks),
        episode_joules=episode_joules, joules_per_offered=episode_joules / len(requests),
        energy_scale_joules=energy_scale_joules, owned_peak_mib=owned_peak_mib)


def compare_panel(rows):
    """Paired two-round diagnostic, selecting only predicted-feasible candidates.

    Observed infeasibility of the selected candidate yields unknown regret;
    it must never be hidden by selecting retrospectively from feasible outcomes.
    """
    candidates = sorted({r['candidate_id'] for r in rows})
    keys = [(r['candidate_id'], r['round']) for r in rows]
    if not candidates or len(keys) != len(set(keys)) or set(keys) != {(c, i) for c in candidates for i in (1, 2)}:
        raise ValueError('Complete unique paired two-round panel required')
    for row in rows:
        for k in ('observed_feasible', 'predicted_feasible'):
            if type(row[k]) is not bool:
                raise ValueError('Explicit feasibility required')
        for k in ('observed', 'predicted'):
            number(row[k]['U'], k + ' utility')
    means = {kind: {c: math.fsum(r[kind]['U'] for r in rows if r['candidate_id'] == c) / 2
                    for c in candidates if all(r[kind + '_feasible'] for r in rows if r['candidate_id'] == c)}
             for kind in ('observed', 'predicted')}
    def winners(values):
        return sorted(c for c, v in values.items() if v == max(values.values())) if values else []
    predicted_winners = winners(means['predicted'])
    selected = predicted_winners[0] if predicted_winners else None
    regrets = []
    for i in (1, 2):
        values = {r['candidate_id']: r['observed']['U'] for r in rows if r['round'] == i and r['observed_feasible']}
        regrets.append(max(values.values()) - values[selected] if selected in values else None)
    pairs = []
    for a, b in combinations(candidates, 2):
        by_key = {(r['candidate_id'], r['round']): r for r in rows}
        delta = [by_key[b, i]['observed']['U'] - by_key[a, i]['observed']['U'] for i in (1, 2)]
        pairs.append(dict(a=a, b=b, delta_U_b_minus_a=delta, mean_delta_U=sum(delta) / 2,
            mean_contribution_delta={t: sum(by_key[b, i]['observed']['contributions'][t] -
                by_key[a, i]['observed']['contributions'][t] for i in (1, 2)) / 2 for t in TERMS}))
    observed = means['observed']
    return dict(mean_U=means, predicted_selected=selected, predicted_tied_winners=predicted_winners,
        observed_best=winners(observed), per_round_regret=regrets,
        mean_regret=max(observed.values()) - observed[selected] if selected in observed else None,
        pairs=pairs, independent_confirmation=False, population_confidence_claim=False)
