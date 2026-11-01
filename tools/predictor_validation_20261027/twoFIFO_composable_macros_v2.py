"""New-schema complete-attempt macros; priming informs history, never labels."""
import argparse
import json
import tempfile
from pathlib import Path

from .stage_response_features_v1 import checked, receipt, ordered_events
from .composable_response_macros_v1 import request_macros
from .stage_host_partition_v1 import partition_request
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .twoFIFO_owned_training_review_v2 import SCHEMA as REVIEW_SCHEMA, inspect_cohort, attempt_events, require, review as recheck_review


def request_rows(raw, plan, capture_sha256):
    """Pure complete-cohort partition; caller separately proves native receipts."""
    cohort=inspect_cohort(plan,raw);require(not cohort['issues'], 'Incomplete cohort cannot become selected labels')
    rows=raw['request_records'];priming=raw['priming_request_records']
    events=attempt_events(raw)
    result=[]
    for row in rows:
        clock=partition_request(row)
        value=request_macros(row,events,capture_sha256,'natural')
        require(value['worker_start_to_ready_ns']==clock['worker_handler_to_ready_ns'],
            'Disjoint original worker occupancy reconstruction required')
        value.update(capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],execution_id=plan['execution_id'])
        for macro in value['macros']:
            macro.update(training_replication_unit=plan['capture_id'],attempt_id=plan['attempt_id'])
        result.append(value)
    return result,events


def extract(training_review):
    review_ref=receipt(training_review);review=checked(review_ref)
    require(review.get('schema')==REVIEW_SCHEMA
        and review.get('status')=='PASS_COMPLETE_NATIVE_TRAINING_NOT_PREDICTOR_QUALIFICATION'
        and review.get('training_feature_extraction_eligible') is True and review.get('issues')==[]
        and review.get('validation_labels_read') is False,
        'Exact complete native new-schema review required')
    require(review.get('execution') is not None,'Owned execution receipt required')
    for source in review['sources']:
        require(receipt(source['path'])==source,'Training review source changed')
    with tempfile.TemporaryDirectory(prefix='twoFIFO-native-training-recheck-') as directory:
        rebuilt=recheck_review(review['raw']['path'],Path(directory)/'review',review['execution']['path'])
        require(checked(rebuilt)==review,'Native training review must rederive exactly, not self-declare eligibility')
    raw_ref=review['raw'];raw=checked(raw_ref);plan=checked(raw['plan'])
    require(raw.get('backend')=='native' and raw.get('status')=='completed', 'Native complete attempt required')
    rows=raw['request_records'];priming=raw['priming_request_records']
    result,events=request_rows(raw,plan,raw_ref['sha256'])
    sources=strict_references([review_ref,raw_ref,raw['plan'],raw['context'],review['sources'],implementation_sources(__file__)])
    require(all(receipt(r['path'])==r for r in sources),'Source changed during new-schema extraction')
    return dict(schema='gsdeploy.twoFIFO_composable_response_macro_training.v2',
        status='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER',mode='natural',
        capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],
        leaveout_unit='whole_capture_block_all_shards',each_attempt_has_independent_startup=True,
        requests=result,measured_requests=len(rows),macros=sum(len(r['macros']) for r in result),
        history_event_count=len(events),priming_rows_excluded_as_labels=len(priming),
        whole_attempt_history_used_for_visible_prefix=True,case_history_reset=False,
        sources=sources,complete_offered_denominator_preserved=True,physical_SM_service_identified=False,
        fitted_model_qualified=False,main_queue_provider_eligible=False,final_gain_eligible=False,
        GPU_started=False,validation_labels_read=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-review',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();result=extract(a.training_review)
    with Path(a.output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(measured_requests=result['measured_requests'],macros=result['macros'])))
