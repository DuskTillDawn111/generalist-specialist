"""New conditional session-origin lineage; preserves failed median review.

NVTX push precedes host start; host end precedes PHASE emission and NVTX pop.
Long emission tails widen causal brackets without becoming clock error bounds.
No anchor is deleted and no median residual threshold is relaxed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sqlite3

from .native_collect import receipt,save
from .operator_smoke_review_v2 import validate_operator_rows
from .session_clock_audit_v2 import causal_offset_interval
from .smoke_verifier_v3 import check_receipt,validate_request


def eligible_issue(issue):
    return (issue.startswith(('PROCESS_CLOCK_MAPPING_FAILED:','PHASE_OWNER_OR_CLOCK_INVALID:',
                              'REAL_KERNEL_REQUEST_COVERAGE_FAILED:','NO_AFFINE_CALL_WITH_REAL_CUPTI_KERNEL:'))
            or (issue.startswith('INVALID_AFFINE_METADATA_OR_BINDING:') and ':Invalid affine host/CUPTI range binding:' in issue))


def adapt(parsed,offset,pid):
    interval=causal_offset_interval(parsed['phases'],pid)
    if not interval['known'] or not interval['lower_ns']<=offset<=interval['upper_ns']:
        raise ValueError('Session origin must satisfy every source-ordered causal bracket')
    old=copy.deepcopy(parsed['clock_mappings'][str(pid)])
    # Both appearances of each original kernel retain exactly the same genuine
    # CUPTI timestamps, duration and ownership; only host coordinate changes.
    for phase in parsed['phases']:
        if phase['pid']!=pid:raise ValueError('One actual owner PID required')
        phase['clock_mapping_valid']=True
        for kernel in phase['kernels']:
            if kernel['pid']!=pid:raise ValueError('Kernel owner mismatch')
            kernel['start_ns']=kernel['nsys_start_ns']+offset
            kernel['end_ns']=kernel['nsys_end_ns']+offset
    for container in list(parsed.get('nvtx_ranges',[]))+[dict(kernels=parsed.get('unattributed_kernels',[]))]:
        for kernel in container.get('kernels',[]):
            kernel['start_ns']=kernel['nsys_start_ns']+offset
            kernel['end_ns']=kernel['nsys_end_ns']+offset
    parsed['clock_mappings'][str(pid)]=dict(method='sqlite_session_systemClockNs_with_all_source_ordered_causal_brackets',
        offset_ns=offset,valid=True,conditional_constant_offset=True,causal_interval=interval,
        original_median_mapping=old,original_median_review_preserved=True,
        cross_process_precision_qualified=False,device_host_calibration_error_bound_ns=None,drift_bound_ns=None)
    parsed['nsys_clock']='session_relative_plus_explicit_session_systemClockNs_conditional_host_origin'
    return parsed


def review(prior_path,host_path,output):
    prior_path,host_path=Path(prior_path).resolve(),Path(host_path).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    prior=json.loads(prior_path.read_text());host=json.loads(host_path.read_text());issues=[];owners={};audits={}
    if prior.get('schema')!='gsdeploy.startup_training_collection_strict_review.v4':raise ValueError('Actual original strict trace review required')
    if prior.get('status') not in ('PASS','FAIL') or not all(eligible_issue(i) for i in prior['issues']):raise ValueError('Non-clock original issue cannot be rescued by clock lineage')
    if host.get('schema')!='gsdeploy.host_startup_collection_strict_review.v4' or host.get('status')!='PASS':raise ValueError('Independent strict host review PASS required')
    for report in (prior,host):
        for name in ('raw','plan','context'):
            if not check_receipt(report[name]):raise ValueError('Changed actual evidence')
        if not all(check_receipt(r) for r in report['evaluator_sources']):raise ValueError('Changed evaluator source')
    if any(prior[k]!=host[k] for k in ('raw','plan','context')):raise ValueError('Matching actual raw required')
    context=json.loads(Path(prior['context']['path']).read_text());raw=json.loads(Path(prior['raw']['path']).read_text())
    ordering=[s['frozen'] for s in context['execution_sources'] if Path(s['original_path']).name in ('semantic_worker.py','semantic_observer.py','operator_worker_v2.py','operator_observer_v2.py')]
    if len(ordering)!=4 or not all(check_receipt(r) for r in ordering):raise ValueError('Frozen actual source-ordered instrumentation required')
    for alias,owner in prior['owners'].items():
        owner=copy.deepcopy(owner);rows=[r for r in raw['request_records'] if r['model']==alias]
        if not rows:owners[alias]=owner;continue
        for name in ('sqlite','parsed'):
            if not check_receipt(owner[name]):raise ValueError('Changed canonical original '+name)
        parsed=json.loads(Path(owner['parsed']['path']).read_text());pid=raw['ready'][alias]['pid']
        connection=sqlite3.connect('file:'+owner['sqlite']['path']+'?mode=ro',uri=True)
        origins=connection.execute('select systemClockNs from TARGET_INFO_SESSION_START_TIME').fetchall();connection.close()
        if len(origins)!=1 or type(origins[0][0]) is not int:raise ValueError('One genuine session host origin required')
        original_parsed=owner['parsed'];parsed=adapt(parsed,origins[0][0],pid)
        expected=[]
        for row in rows:
            found,anchors=validate_request(row,pid);issues+=found;expected+=anchors
        key=lambda r:(r.get('pid'),r.get('request_id'),r.get('kind'),r.get('segment_id'))
        if Counter(map(key,expected))!=Counter(map(key,parsed['phases'])):issues.append('EXACT_ACTUAL_ANCHOR_COVERAGE_FAILED:'+alias)
        if any(not p['kernels'] for p in parsed['phases'] if p['segment_id'].startswith('handler_')):
            issues.append('HANDLER_WITHOUT_ACTUAL_KERNELS:'+alias)
        if context.get('affine_observation'):
            for row in rows:
                document=row['worker_response'].get('semantic_forward_observations',{}).get('affine_operator_observations',{})
                found,_,_=validate_operator_rows(document,parsed['phases'],request_id=row['request_id'],pid=pid)
                issues.extend(i+':'+alias+':'+row['request_id'] for i in found)
        parsed['sources'].extend([receipt(prior_path),receipt(host_path),receipt(__file__)])
        target=output/('parsed-session-origin-'+alias.replace('#','-')+'.json');save(target,parsed)
        audits[alias]=dict(original_parsed=original_parsed,mapping=parsed['clock_mappings'][str(pid)],
            no_anchors_removed=True,no_kernel_duration_rescaled=True)
        owner['parsed']=receipt(target)
        owner.pop('operator_kernel_index',None)  # Original failed index cannot be reused as a passing index.
        owner['clock_lineage']='new_conditional_session_origin_preserving_failed_median_artifact';owners[alias]=owner
    result=dict(schema='gsdeploy.startup_training_collection_strict_review.v4',status='PASS' if not issues else 'FAIL',issues=issues,
        review_scope='HOST_STRICT_PLUS_NEW_CONDITIONAL_CAUSAL_SESSION_ORIGIN_KERNEL_LINEAGE',owners=owners,
        raw=prior['raw'],plan=prior['plan'],context=prior['context'],capture_run_id=prior['capture_run_id'],
        original_median_review=receipt(prior_path),independent_host_review=receipt(host_path),clock_audits=audits,
        source_ordering_receipts=ordering,evaluator_sources=[receipt(__file__)],
        independent_capture_count=1,minimum_training_capture_requirement_satisfied=False,
        cross_process_CUDA_clock_qualified=False,calibrated_device_host_precision_bound_ns=None,
        training_only=True,validation_started=False,
        interpretation='Conditional per-process source-ordered constant host origin; no calibrated cross-process/SM timing claim. Original median FAIL remains immutable.')
    save(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prior-review',required=True);parser.add_argument('--host-review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();r=review(args.prior_review,args.host_review,args.output)
    print(json.dumps(dict(status=r['status'],issue_counts=dict(Counter(r['issues'])),report=receipt(Path(args.output)/'report.json'))))


if __name__=='__main__':main()
