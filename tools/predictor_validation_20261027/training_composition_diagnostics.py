"""CPU-only arrival-third/input composition diagnostics for audited training.

Ranges are observed min/max, not confidence intervals. Composition-adjusted
means are descriptive reweighting, not causal aging or interference estimates.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics

from gsdeploy.metrics import percentile
from .native_flow import checked, receipt, save
from .probe_validation import read


def sample_stats(values):
    return dict(n=len(values), mean_ms=statistics.mean(values),
        minimum_ms=min(values), maximum_ms=max(values), p95_ms=percentile(values,95))


def summarize(raw, plan):
    cells = defaultdict(list)
    overall = defaultdict(Counter)
    left, width = raw['phase_start_ns'], raw['deadline_ns']-raw['phase_start_ns']
    for row in raw['request_records']:
        if row.get('status') != 'completed':
            raise ValueError('Complete measurement cohort required')
        offset = row['scheduled_time_ns']-left
        if not 0 <= offset < width:
            raise ValueError('Public scheduled arrival must lie in measurement window')
        third = min(2, (3*offset)//width)
        alias, task, pool = row['model'], row['task'], row['pool_id']
        family = plan['aliases'][alias]['family']
        service = (row['worker_response']['end_perf_ns']-row['observed_worker_start_time_ns'])/1e6
        if service <= 0:
            raise ValueError('Positive observed worker service required')
        cells[(alias,family,task,pool,third)].append(service)
        overall[(alias,family,task)][pool] += 1
    groups = []
    for (alias,family,task), mix in sorted(overall.items()):
        total = sum(mix.values())
        periods = []
        for third in range(3):
            inputs = [{**dict(pool_id=pool), **sample_stats(cells[(alias,family,task,pool,third)])}
                for pool in sorted(mix) if cells[(alias,family,task,pool,third)]]
            observed = [v for pool in mix for v in cells[(alias,family,task,pool,third)]]
            missing = [pool for pool in sorted(mix) if not cells[(alias,family,task,pool,third)]]
            standardized = (sum(mix[pool]/total*statistics.mean(cells[(alias,family,task,pool,third)])
                for pool in mix) if not missing else None)
            periods.append(dict(arrival_third=third, inputs=inputs,
                observed_service=sample_stats(observed) if observed else None,
                full_cohort_mix_standardized_service_mean_ms=standardized,
                missing_reference_inputs=missing))
        groups.append(dict(alias=alias,family=family,task=task,
            full_cohort_input_counts=dict(sorted(mix.items())),arrival_thirds=periods))
    return dict(groups=groups, request_count=sum(sum(m.values()) for m in overall.values()),
        standardization='Within each alias/task, reweight per-third per-input service means to its full-cohort input frequencies; UNKNOWN if any reference input is missing',
        uncertainty='Observed min/max and P95 only; no confidence interval, causal age effect or population variance claim')


def run(audit_path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    audit = read(audit_path)
    if audit.get('schema') != 'gsdeploy.training_run_postcollection_audit.v1' or audit.get('status') != 'PASS':
        raise ValueError('Passed prospective training source/quality audit required')
    checked(audit['source'])
    raw = read(checked(audit['raw']))
    plan = read(checked(audit['plan']))
    if plan.get('phase') != 'predictor_training' or plan.get('split') != 'training':
        raise ValueError('Training-only diagnostics required')
    value = dict(schema='gsdeploy.training_input_composition_diagnostics.v1',
        audit=receipt(audit_path),raw=audit['raw'],plan=audit['plan'],source=receipt(__file__),
        candidate_id=audit['candidate_id'],seed=audit['seed'],**summarize(raw,plan),
        GPU_started=False,independent_validation=False,final_gain_eligible=False)
    save(output,value)
    return value


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=run(a.audit,a.output)
    print('COMPLETE',v['request_count'],'requests; composition descriptive; GPU_started=False')
