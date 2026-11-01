"""Read-only independent native-v3 audit; incomplete attempt is metadata only."""
import argparse
import json
from pathlib import Path

from . import twoFIFO_owned_training_review_v3 as review
from . import twoFIFO_composable_macros_v3 as macros
from . import twoFIFO_observed_resources_v2 as resources
from .stage_response_features_v1 import receipt,checked
from .implementation_source_closure_v27 import implementation_sources,strict_references


def run(attempt_path,output):
    attempt=Path(attempt_path).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);execution=attempt/'collection/execution.json'
    result=dict(schema='gsdeploy.twoFIFO_native_v3_independent_review.v1',status='INCOMPLETE',
        attempt=str(attempt),raw_opened=False,training_feature_extraction_eligible=False,
        actual_predictor_qualified=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        GPU_started=False,validation_labels_read=False)
    if execution.is_file():
        execution_ref=receipt(execution);executed=checked(execution_ref)
        if executed.get('status')!='CHILD_EXITED_REQUIRES_PROFILE_REVIEW':
            result.update(status='FAILED_OR_INCOMPLETE_FINAL_EXECUTION',execution=execution_ref)
        else:
            rebuilt=review.review(attempt/'collection/raw.json',output/'native-review',execution)
            checked_review=checked(rebuilt);result.update(raw_opened=True,execution=execution_ref,review=rebuilt,
                native_review_status=checked_review['status'],
                training_feature_extraction_eligible=checked_review['training_feature_extraction_eligible'])
            prior=attempt/'native-review01/report.json'
            if prior.exists():review.require(checked(receipt(prior))==checked_review,'Original native review did not rederive exactly')
            if checked_review['training_feature_extraction_eligible']:
                doc=macros.extract(rebuilt['path']);raw=checked(checked_review['raw']);plan=checked(raw['plan'])
                facts=resources.observed(raw)
                (output/'observed-resources.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
                (output/'macros.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
                clocks=[review.inspect_request_clock(r) for r in raw['priming_request_records']+raw['request_records']]
                all_E=(facts['startup']['energy_joules'] is not None and all(r['measurement_energy_joules'] is not None for r in facts['cases']))
                result.update(status='PASS_NATIVE_CONTRACTS_AND_RESOURCE_COVERAGE' if all_E else 'HOLD_UNOBSERVABLE_ENERGY',
                    raw=checked_review['raw'],plan=raw['plan'],cohort=review.inspect_cohort(plan,raw),
                    histories=review.inspect_histories(plan,raw),activity=checked_review['checkpoint_activity'],
                    telemetry=review.inspect_samples(raw,executed['process_group']),rechecked_request_clocks=len(clocks),
                    late_write_after_RESULT_requests=sum(r['late_write_after_RESULT'] for r in clocks),
                    complete_measurement_and_startup_energy_observable=all_E,resource_facts=receipt(output/'observed-resources.json'),
                    macros=receipt(output/'macros.json'),macro_rows=doc['macros'],measured_macro_requests=doc['measured_requests'],
                    native_zero_observation_overhead_proven=False)
            else:result['status']='NOT_ELIGIBLE_NATIVE_REVIEW'
    result['sources']=strict_references([implementation_sources(__file__),
        [] if not execution.is_file() else [receipt(execution)]])
    review.require(all(receipt(s['path'])==s for s in result['sources']),'Independent audit source changed')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return receipt(output/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run(args.attempt,args.output)))
