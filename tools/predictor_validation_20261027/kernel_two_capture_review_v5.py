"""CPU exact-source review of all previously missing numerical kernel models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .kernel_two_capture_provisional_v5 import ExactSourceTwoCaptureProvisionalResolver
from .native_collect import receipt, save


def review(provisional_report,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    enabled=ExactSourceTwoCaptureProvisionalResolver(provisional_report,allow_two_capture_provisional_hypothesis=True)
    disabled=ExactSourceTwoCaptureProvisionalResolver(provisional_report,base_resolver=enabled.base)
    counts=defaultdict(Counter);samples=[]
    for entry in enabled.base.index.values():
        model=enabled.base.models[entry['group_id']]
        if 'coefficients' in model:continue
        kernel=dict(entry['signature'],**entry['identity'],source_sqlite=entry['source'])
        original=disabled(kernel);prediction=enabled(kernel)
        if original['numeric_hypothesis_available']:raise ValueError('Default resolver changed')
        for field in ('known','strict_known','strict_failure_reasons','prediction'):
            if prediction[field]!=original[field]:raise ValueError('Original strict result changed')
        if enabled(dict(kernel,duration_ns=10**30))!=prediction:
            raise ValueError('Query target changed provisional mean')
        invalid=dict(kernel,correlation_id=-1)
        if enabled(invalid)['numeric_hypothesis_available']:raise ValueError('Exact identity bypassed')
        family=enabled.base.groups[entry['group_id']]['family']
        if prediction['numeric_hypothesis_available']:
            evidence=prediction['provisional_evidence'];family=evidence['group']['family']
            if len(evidence['training_samples'])!=2 or len({s['run_id'] for s in evidence['training_samples']})!=2:
                raise ValueError('Two actual raw captures required')
            label='provisional_numeric_strict_unknown'
            samples.append(dict(observation_id=entry['observation_id'],family=family,
                duration_ns=prediction['duration_ns'],strict_failure_reasons=prediction['strict_failure_reasons']))
        else:
            label='single_capture_remains_unknown'
        counts[family][label]+=1
    report=dict(schema='gsdeploy.two_capture_provisional_exact_source_review.v5',
        provisional_fit=enabled.source,family_counts={k:dict(v) for k,v in counts.items()},samples=samples,
        explicit_opt_in_default_disabled=True,original_strict_results_unchanged=True,
        changed_exact_identity_rejected=True,query_measured_duration_has_no_effect=True,
        single_capture_numeric_forbidden=True,heldout_validation_performed=False,
        deployment_qualified=False,GPU_workload_executed=False,
        implementation_sources=[receipt(__file__)])
    save(output/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provisional-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=review(args.provisional_report,args.output)
    print(report['family_counts'])


if __name__=='__main__':main()
