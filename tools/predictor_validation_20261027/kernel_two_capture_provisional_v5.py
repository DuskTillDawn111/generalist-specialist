"""Explicit two-raw-capture group means, separate from strict regression fits.

Training targets define this new estimand. Query measured durations never do.
Original three-observation/25% strict fit requirements remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import copy
import json
import math
from pathlib import Path

from .kernel_baseline_intake_v3 import checked, _verify
from .kernel_semantic_query_v5 import ExactSourceSemanticHypothesisResolver
from .native_collect import receipt, save


def provisional_mean(rows):
    if len(rows)!=2 or len({row['run_id'] for row in rows})!=2:
        return None
    targets=[row['duration_ns'] for row in rows]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in targets):
        raise ValueError('Positive actual training duration targets required')
    return sum(targets)/2


def build(fit_path, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    fit_source=receipt(fit_path);fit=checked(fit_source)
    for source in fit['implementation_sources']:_verify(source)
    models=checked(fit['models']);_verify(fit['observations'])
    requests=checked(fit['request_sequences'])
    raw_sources={request['run_id']:request['source_raw'] for request in requests}
    for run,source in raw_sources.items():
        _verify(source)
        if run!='raw-sha256:'+source['sha256']:raise ValueError('Actual raw capture run required')
    by_group=defaultdict(list)
    with Path(fit['observations']['path']).open() as stream:
        for line in stream:
            row=json.loads(line);model=models['models'][row['group_id']]
            if model['reasons']==['INSUFFICIENT_OBSERVATIONS_OR_RUNS']:
                by_group[row['group_id']].append(row)
    groups={};family=Counter();excluded=Counter()
    for gid,rows in by_group.items():
        value=provisional_mean(rows)
        if value is None:
            excluded[models['groups'][gid]['family']]+=len(rows)
            continue
        if any(row['run_id'] not in raw_sources for row in rows):raise ValueError('Missing actual capture receipt')
        model=models['models'][gid]
        groups[gid]=dict(group=models['groups'][gid],duration_hypothesis_ns=value,
            estimand='arithmetic_mean_of_two_actual_training_capture_group_observations',
            training_samples=[dict(observation_id=r['id'],run_id=r['run_id'],source_raw=raw_sources[r['run_id']],
                exact_launch=r['exact_launch'],features=r['features'],training_duration_ns=r['duration_ns']) for r in rows],
            original_strict_model=copy.deepcopy(model),strict_known=False,known=False,
            original_strict_failure_reasons=list(model['reasons']),minimum_observations_met=False,
            strict_training_fit_accepted=False,strict_residual_qualification_passed=False,
            feature_transfer_validated=False,hypothesis_only=True,deployment_qualified=False)
        family[models['groups'][gid]['family']]+=len(rows)
    report=dict(schema='gsdeploy.two_capture_provisional_kernel_group_mean.v5',fit=fit_source,
        observations=fit['observations'],groups=groups,group_count=len(groups),
        family_provisional_launch_counts=dict(family),excluded_single_capture_launch_counts=dict(excluded),
        explicit_opt_in_required='allow_two_capture_provisional_hypothesis=True',
        original_strict_thresholds=dict(min_observations=3,min_runs=2,max_relative_training_error=0.25),
        original_model_modified=False,query_measured_target_fallback_used=False,
        heldout_validation_performed=False,natural_timing_validated=False,deployment_qualified=False,
        GPU_workload_executed=False,model_weights_loaded=False,
        implementation_sources=[receipt(__file__),receipt(Path(__file__).with_name('kernel_semantic_query_v5.py'))])
    save(output/'report.json',report);return report


class ExactSourceTwoCaptureProvisionalResolver:
    def __init__(self,provisional_report,*,allow_two_capture_provisional_hypothesis=False,base_resolver=None):
        self.allowed=allow_two_capture_provisional_hypothesis is True
        self.source=receipt(provisional_report);report=checked(self.source)
        if report['schema']!='gsdeploy.two_capture_provisional_kernel_group_mean.v5':raise ValueError('Two-capture report required')
        for source in report['implementation_sources']:_verify(source)
        self.base=base_resolver or ExactSourceSemanticHypothesisResolver(report['fit']['path'])
        if self.base.fit_source!=report['fit']:raise ValueError('Provisional and base fit receipts differ')
        self.by_observation={sample['observation_id']:group for group in report['groups'].values() for sample in group['training_samples']}

    def __call__(self,kernel):
        result=self.base(kernel)
        if not self.allowed or result['numeric_hypothesis_available']:return result
        group=self.by_observation.get(result.get('observation_id'))
        if group is None:return result
        result=copy.deepcopy(result)
        if result['known'] or result['strict_known'] or result['strict_failure_reasons']!=group['original_strict_failure_reasons']:
            raise ValueError('Provisional mean cannot change strict eligibility')
        result.update(numeric_hypothesis_available=True,duration_ns=group['duration_hypothesis_ns'],
            status='PROVISIONAL_TWO_CAPTURE_GROUP_MEAN_HYPOTHESIS',
            provisional_two_capture_hypothesis=True,allow_two_capture_provisional_hypothesis=True,
            provisional_fit_source=copy.deepcopy(self.source),provisional_evidence=copy.deepcopy(group),
            known=False,strict_known=False,hypothesis_only=True,physical_accuracy=False,
            deployment_qualified=False,measured_target_fallback_used=False)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=build(args.fit,args.output)
    print({k:report[k] for k in ('group_count','family_provisional_launch_counts','excluded_single_capture_launch_counts')})


if __name__=='__main__':main()
