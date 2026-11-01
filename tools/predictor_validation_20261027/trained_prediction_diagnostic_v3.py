"""Run actual trained serial provider coverage, without a GPU or timing fallback.

This diagnostic deliberately includes every training-profile kernel, including
unsupported ones. Partial supported sums are never whole-request predictions.
It is not a prospective independent validation or a prediction accuracy report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .forecast_gate_v3 import BUNDLE_SCHEMA
from .kernel_baseline_query_v3 import ExactSourceBaselineResolver
from .native_collect import receipt
from .trained_supply_provider_v3 import TrainedKernelSupplyProvider
from .forecast_gate_v2 import _write


def diagnose(bundle_path,output):
    bundle_path=Path(bundle_path).resolve();bundle=json.loads(bundle_path.read_text())
    if bundle.get('schema')!=BUNDLE_SCHEMA or bundle.get('kind')!='trained_hypothesis':
        raise ValueError('Real trained-hypothesis bundle required')
    refs=bundle['artifacts']
    for ref in [*refs.values(),*bundle['baseline_intakes']]:
        if receipt(ref['path'])!=ref:raise ValueError('Changed trained artifact')
    resolver=ExactSourceBaselineResolver(refs['baseline_fit']['path'],[ref['path'] for ref in bundle['baseline_intakes']])
    ordered=json.loads(Path(refs['ordered']['path']).read_text())
    residual=json.loads(Path(refs['residual']['path']).read_text())
    provider=TrainedKernelSupplyProvider(ordered,residual,ordered['execution_context_sha256'],
        baseline_resolver=resolver,allow_analytical_hypothesis=True,**bundle['hypothesis_options'])
    profile_results=[]
    for profile in provider.profiles.values():
        row={key:profile[key] for key in ('alias','task','pool_id')}
        result=dict(profile_id=profile['id'],**row,coverage=provider.baseline_coverage[profile['id']])
        try:
            provider.plan(row,bundle['forecast_defaults']['aliases'],'profiled')
            result.update(status='EXECUTABLE_TRAINED_PROFILE',reason=None)
        except UnknownEvidence as exc:result.update(status='UNKNOWN',reason=str(exc))
        profile_results.append(result)
    report=dict(schema='gsdeploy.trained_provider_coverage_diagnostic.v3',
        status='UNKNOWN' if any(row['status']=='UNKNOWN' for row in profile_results) else 'EXECUTABLE_TRAINING_PROFILES',
        bundle=receipt(bundle_path),artifacts=refs,baseline_intakes=bundle['baseline_intakes'],
        profile_results=profile_results,
        known_kernels=sum(row['coverage']['known_kernels'] for row in profile_results),
        unknown_kernels=sum(row['coverage']['unknown_kernels'] for row in profile_results),
        accepted_exact_source_launches=len(resolver.index),
        actual_regression_resolver_executed=True,actual_supply_provider_executed=True,
        measured_target_fallback_used=False,whole_request_predictions_available=all(row['status']!='UNKNOWN' for row in profile_results),
        full_original_request_domain=ordered['denominator'],
        GPU_started=False,independent_prediction_validation=False,prediction_accuracy_established=False,
        physical_gain_certified=False,unseen_input_feature_map_available=False,
        source_receipts=[receipt(__file__),receipt(Path(__file__).with_name('trained_supply_provider_v3.py')),
                         receipt(Path(__file__).with_name('kernel_baseline_query_v3.py'))],
        limitation='Supported kernel sums cover a strict subset, not request completion; unsupported work is never replaced by measured targets.')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);_write(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=diagnose(args.bundle,args.output)
    print(json.dumps({key:result[key] for key in ('status','known_kernels','unknown_kernels','accepted_exact_source_launches',
                                               'measured_target_fallback_used','whole_request_predictions_available')}))


if __name__=='__main__':main()
