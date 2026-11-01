"""CPU-only 11-setting, three-capture response replication declaration.

The parameters payload is accepted by the frozen unified-v8_1 job validator.
This writes no execution/capture claims and never prepares or starts a runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from gsdeploy.resources import identity
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .kernel_sequence_audit_v2 import verify
from .native_collect import receipt
from .unified_corun_review_v8_1 import LIGHT, load_review
from .unified_corun_collect_v8_1 import validate_jobs, priming_items

SCHEMA='gsdeploy.kernel_dag_response_replication_plan.v1'


def jobs_for(bindings):
    owners=list(bindings)
    if len(owners)!=4 or any(not b.get('pool_id') or not b.get('task') for b in bindings.values()):
        raise ValueError('Exactly four explicit owner/input/task bindings required')
    jobs=[]
    for width in (1,2,4):
        for members in combinations(owners,width):
            job=dict(name='response-replication-'+str(len(jobs)),context='same-stable-execution-context',
                inputs={owner:bindings[owner]['pool_id'] for owner in members},
                tasks={owner:bindings[owner]['task'] for owner in members},
                start_offsets_ms={owner:0 for owner in members},split='training',
                validation_view='TRAINING_ONLY_NOT_VALIDATION',phase='kernel-response-replication',
                input_stratum='fixed-exact-work')
            job['id']=identity(job);jobs.append(job)
    validate_jobs(jobs)
    return jobs


def prepare(review_path,output):
    strict,plan,raw=load_review(review_path,LIGHT)
    if strict.get('training_only') is not True or strict.get('validation_started') is not False:
        raise ValueError('Explicit training-only source review required')
    bindings={}
    for row in raw['request_records']:
        owner=row['model']
        if owner in bindings:raise ValueError('Select one exact source input per owner explicitly')
        item=next(i for i in plan['inputs'] if i['pool_id']==row['pool_id'] and i['task']==row['task'])
        bindings[owner]=dict(task=row['task'],pool_id=row['pool_id'],exact_input_sha256=item['exact_input_sha256'])
    jobs=jobs_for(bindings);primers=priming_items(dict(inputs=plan['inputs']))
    source_refs=strict_references([receipt(review_path),strict,implementation_sources(__file__)])
    for ref in source_refs:verify(ref)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    parameters=dict(schema=SCHEMA+'.parameters',stages=[dict(id='B',jobs=jobs)],
        training_only=True,validation_labels_read=False)
    with (output/'parameters.json').open('x') as stream:json.dump(parameters,stream,indent=2);stream.write('\n')
    observations=Counter(owner for job in jobs for owner in job['inputs'])
    result=dict(schema=SCHEMA,status='CPU_PLANNED_MATRIX_NOT_CAPTURE_PREPARED_OR_LAUNCHED',
        parameters=receipt(output/'parameters.json'),exact_owner_inputs=bindings,jobs=jobs,
        captures=3,independent_fresh_lifecycles_required=True,jobs_per_capture=11,
        measured_requests_per_capture=20,measured_requests_total=60,
        priming_requests_per_capture=len(primers),priming_requests_total=3*len(primers),
        observations_per_exact_node_per_capture=dict(observations),
        observations_per_exact_node_total={owner:count*3 for owner,count in observations.items()},
        job_width_counts=dict(Counter(len(job['inputs']) for job in jobs)),
        all_four_owners_in_one_job=any(len(job['inputs'])==4 for job in jobs),
        original_bounds=dict(max_jobs=24,max_measured_requests=50,validator_passed=True),
        same_context_protocol='Prepare one frozen execution context/source snapshot, then bind three independently claimed lifecycle attempts to identical full context content; capture IDs/PIDs/raw hashes remain distinct.',
        existing_source_context_is_new_campaign_target=False,
        blockers=['A new source-bound campaign wrapper must prepare one stable context and three exclusive no-retry execution identities; no repeated calls to old prepare with fresh nonces.',
            'Use the original GPU experiment lease/watchdog/source guard; queue only after current baseline and matching campaigns finish.',
            'Extend unified native intake/template adapter to retain multiple request instances of the same exact work within one raw capture; baseline template sharing requires every observed node signature and dependency to match.',
            'Every attempt needs strict original native review, exact SQLite occupancy audit, full measured/outside peer and canonical node denominator; OCR work differences remain separate/UNKNOWN.',
            'Actual six-feature path contrasts and nuisance-adjusted derivative rank, local/root checks and complete capture/block leaveout remain required; this plan does not guarantee rank.',
            'Record actual resource/disk/runtime gates before launch; this declaration does not authorize GPU execution.'],
        offset_extension='All initial offsets are fixed at 0; later predeclared offsets or more work configurations are separate additional training if actual rank/support fails.',
        numerical_fit_run=False,GPU_started=False,capture_claims_created=False,
        validation_labels_read=False,independent_prediction_accuracy_evaluated=False,deployment_qualified=False,
        source_review=receipt(review_path),sources=source_refs)
    for ref in source_refs:verify(ref)
    with (output/'plan.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=prepare(args.review,args.output)
    print(json.dumps({k:result[k] for k in ('status','captures','jobs_per_capture','measured_requests_total','priming_requests_total')}))
