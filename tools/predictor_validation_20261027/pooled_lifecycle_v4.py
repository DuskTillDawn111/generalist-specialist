"""Explicit training hypothesis for handoff on previously unseen input IDs.

Pool joint host handoff vectors by family/task under an unchanged execution
identity. This estimates a mean, not a bound or a load-independent distribution.
At least two independent captures and two actual input IDs are required per
group. Validation request IDs never become fitted parameters.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.lifecycle import LifecycleProvider, fit_lifecycle
from .native_collect import receipt, save
from .smoke_verifier_v3 import check_receipt

SCHEMA = 'gsdeploy.pooled_joint_lifecycle_hypothesis.v4'
POOLED_INPUT = '__declared_family_task_pool__'


def fit(observations, context, *, allow_cpu_fixture=False):
    observations = copy.deepcopy(list(observations))
    groups = defaultdict(list)
    for row in observations:
        if row.get('kind') != 'measured' and not (allow_cpu_fixture and row.get('kind') == 'cpu_fixture'):
            raise ValueError('Measured training handoff observations required')
        groups[row['family'], row['task'], row['permit_scope']].append(row)
    supported, support = [], []
    for key, rows in sorted(groups.items()):
        runs, pools = {r['source'] for r in rows}, {r['pool_id'] for r in rows}
        eligible = len(runs) >= 2 and len(pools) >= 2
        support.append(dict(family=key[0], task=key[1], permit_scope=key[2],
                            capture_ids=sorted(runs), input_ids=sorted(pools),
                            observations=len(rows), supported=eligible))
        if eligible:
            supported.extend(dict(row, pool_id=POOLED_INPUT) for row in rows)
    # Validate all observations, including rejected groups, through the original
    # source/context/split/boundary checks. Pool only the grouping key in a copy.
    if observations:
        all_pooled = [dict(row, pool_id=POOLED_INPUT) for row in observations]
        try:
            fit_lifecycle(all_pooled, context)
        except UnknownEvidence:
            pass  # Insufficient support is represented explicitly below.
    model = fit_lifecycle(supported, context) if supported else None
    return dict(schema=SCHEMA, kind='trained_family_task_handoff_hypothesis',
                execution_context_sha256=context, lifecycle=model, support=support,
                status=('FITTED' if len(supported) == len(observations) else 'FITTED_PARTIAL') if supported else 'UNKNOWN',
                input_identity_used_as_fit_feature=False,
                pooling_assumption='Joint mean host handoff is exchangeable across inputs within family/task/permit scope and the identical runtime execution context.',
                unvalidated_dimensions=['input-dependent response size', 'host load', 'co-resident queue pressure', 'held-out input handoff'],
                physical_accuracy_validated=False, confidence_bound=False,
                permit_release_substituted_from_receipt=False)


class PooledLifecycleProvider:
    def __init__(self, document, context, *, allow_cpu_fixture=False, allow_pooling_hypothesis=False):
        if not allow_pooling_hypothesis:
            raise ValueError('Explicit family/task pooling hypothesis opt-in required')
        if document.get('schema') != SCHEMA or document.get('execution_context_sha256') != context:
            raise ValueError('Pooled handoff execution identity mismatch')
        if document.get('lifecycle') is None:
            raise UnknownEvidence('NO_SUPPORTED_POOLED_HANDOFF_PROFILES')
        support = {(r['family'], r['task'], r['permit_scope']) for r in document['support'] if r['supported']}
        for profile in document['lifecycle']['profiles']:
            key = profile['family'], profile['task'], profile['permit_scope']
            if profile['pool_id'] != POOLED_INPUT or key not in support:
                raise ValueError('Unbound pooled handoff profile')
        self.provider = LifecycleProvider(document['lifecycle'], context, allow_cpu_fixture=allow_cpu_fixture)
        self.execution_context_sha256 = context
        self.physical_evidence = self.provider.physical_evidence

    @staticmethod
    def _row(row):
        return dict(row, pool_id=POOLED_INPUT)

    @staticmethod
    def _annotate(result):
        return {**result, 'scope': 'training_family_task_joint_mean_hypothesis',
                'physical_accuracy_validated': False, 'load_independence_proven': False}

    def delays(self, row, aliases, scenario):
        return self._annotate(self.provider.delays(self._row(row), aliases, scenario))

    def residual_handoff(self, row, aliases, scenario, now_ns, observed, *, boundary_observation_status=None):
        return self._annotate(self.provider.residual_handoff(self._row(row), aliases, scenario, now_ns, observed,
                              boundary_observation_status=boundary_observation_status))


def fit_artifact(observations_path, context, output, training_report_path):
    source = receipt(observations_path)
    data = json.loads(Path(observations_path).read_text())
    training_ref = receipt(training_report_path)
    training = json.loads(Path(training_report_path).read_text())
    if (training.get('schema') not in ('gsdeploy.natural_training_lifecycle_fit.v2',
                                      'gsdeploy.natural_training_handoff_quality_fit.v4')
            or training.get('observations') != source
            or training.get('execution_context_sha256') != context
            or not check_receipt(training.get('clock_evidence'))):
        raise ValueError('Source-bound natural training report and clock evidence required')
    captures = training.get('captures', [])
    raw_by_source = {}
    for capture in captures:
        if not check_receipt(capture['raw']) or not check_receipt(capture['review']):
            raise ValueError('Changed underlying natural training capture')
        if capture['run_id'] != 'raw-sha256:' + capture['raw']['sha256']:
            raise ValueError('Independent run must bind the actual raw SHA')
        raw = json.loads(Path(capture['raw']['path']).read_text())
        if raw.get('kind') != 'measured' or raw.get('execution_context_sha256') != context:
            raise ValueError('Actual measured same-context natural captures required')
        raw_by_source[capture['run_id']] = raw
    for row in data['observations']:
        raw = raw_by_source.get(row['source'])
        if raw is None:
            raise ValueError('Pooled observation lacks an indexed actual training capture')
        actual = [r for r in raw['request_records'] if row['id'] == row['source'] + ':' + r['request_id']]
        if (len(actual) != 1 or actual[0].get('split') != 'training'
                or actual[0].get('status') != 'completed'
                or actual[0]['pool_id'] != row['pool_id'] or actual[0]['task'] != row['task']):
            raise ValueError('Pooled observation differs from actual training input or request')
    result = fit(data['observations'], context)
    result['sources'] = [source, training_ref, receipt(__file__), receipt(Path(__file__).resolve().parents[2] / 'gsdeploy/forecasting/lifecycle.py')]
    if not all(check_receipt(ref) for ref in result['sources']):
        raise ValueError('Handoff fit inputs changed during computation')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'model.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', required=True)
    parser.add_argument('--execution-context', required=True)
    parser.add_argument('--training-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = fit_artifact(args.observations, args.execution_context, args.output, args.training_report)
    print(json.dumps(dict(status=result['status'], support=result['support'])))


if __name__ == '__main__':
    main()
