"""Read-only CPU collection audits; never fit or launch workers.

Use ``one`` after both immutable raw and quality annotation exist. ``complete``
requires all four exclusive attempts and verifies their audit receipts. A
collection PASS says nothing about prediction error or a final gain gate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time

from . import flow_profiles, flow_eligibility, flow_metrics, mechanism_audit
from . import h600_selected_validation_run_v2 as runner
from . import h600_selected_validation_v2 as selected
from .flow_crossfit_report import scored_heldout
from .flow_suite_audit import check_structure
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling import digest
from .training_run_audit import verify_quality_annotation

SCHEMA='gsdeploy.H600_independent_collection_audit.v1'


def source_refs(value):
    refs={}
    def visit(item):
        if isinstance(item,dict):
            if set(item)=={'path','sha256'}:
                checked(item);refs[(item['path'],item['sha256'])]=item
            else:
                for child in item.values():visit(child)
        elif isinstance(item,list):
            for child in item:visit(child)
    visit(value)
    return list(refs.values())


def inputs(suite_path):
    """Reuse the completed prospective audit, without recomputing any fit."""
    suite_path=Path(suite_path).resolve();suite=read(suite_path)
    if (suite.get('schema')!=runner.SCHEMA
            or suite['sources']['source']!=receipt(runner.__file__)):
        raise ValueError('Exact selected-v16 suite and runner required')
    prospective_path=suite_path.parent/'prospective-input-audit.json'
    prospective=read(prospective_path)
    if (prospective.get('status')!='PASS' or prospective.get('suite')!=receipt(suite_path)
            or prospective.get('source')!=receipt(runner.__file__)
            or prospective.get('training_verification',{}).get('training_flow_count')!=16):
        raise ValueError('Previously completed prospective full16 PASS audit required')
    sealed=read(checked(suite['sources']['predictions']))
    training=read(checked(suite['sources']['training_audit']))
    plan=read(checked(suite['sources']['plan']))
    qualification=read(checked(suite['sources']['qualification']))
    protocol=read(checked(training['validation_protocol']))
    if (suite['sources']['predictions']!=receipt(Path(suite['sealed_dir'])/'predictions.json')
            or training['predictions']!=suite['sources']['predictions']
            or sealed['sources']['plan']!=suite['sources']['plan']
            or protocol['target_plan']!=suite['sources']['plan']
            or qualification['protocol']!=protocol['capacity_protocol']
            or training['entry_source']!=receipt(selected.__file__)
            or tuple(c['id'] for c in plan['candidates'])!=selected.CANDIDATES
            or plan.get('seed')!=627701 or plan.get('split')!='validation'
            or plan.get('window_seconds')!=600. or plan.get('drain_seconds')!=300.
            or plan.get('warmup_seconds')!=0. or len(plan['trace'])!=1200
            or protocol['validation_thresholds']!=selected.LIMITS):
        raise ValueError('Frozen independent plan, seal or qualification binding changed')
    refs=source_refs([suite,prospective,sealed,training,protocol,qualification])
    refs.extend([receipt(suite_path),receipt(prospective_path),receipt(__file__)])
    check_structure(plan,sealed,suite,{**qualification,'target_seed':627701},suite_path)
    if prospective['audited_ns']<suite['created_ns']:
        raise ValueError('Prospective audit predates suite')
    return suite,plan,sealed,prospective,refs


def inventory(suite,*,complete=False):
    jobs={j['candidate_id']:Path(j['output']).resolve() for j in suite['jobs']}
    roots={p.parent for p in jobs.values()}
    if len(roots)!=1:raise ValueError('Single exclusive suite run root required')
    root=next(iter(roots));present=set()
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir() or child.name not in jobs or child.resolve()!=jobs[child.name]:
                raise ValueError('Extra candidate or attempt in independent run root')
            present.add(child.name)
            if any(p.parent!=child for pattern in ('invocation.json','raw.json','scored-raw.json')
                    for p in child.rglob(pattern)):
                raise ValueError('Nested duplicate attempt or observation is forbidden')
    missing=sorted(set(jobs)-present)
    if complete and missing:raise ValueError('Missing declared independent candidate attempts')
    return dict(declared=sorted(jobs),present=sorted(present),missing=missing,
        exclusive_output_contract=True,unobserved_external_attempts='NOT_IDENTIFIABLE')


def chronology(raw,sealed,suite,prospective):
    startup=raw.get('startup_begin_ns')
    if (type(startup) is not int or startup<=max(sealed['created_ns'],suite['created_ns'],prospective['audited_ns'])
            or not 0<sealed['information_cutoff_ns']<=sealed['created_ns']):
        raise ValueError('Complete forecast seal and prospective audit must precede native startup')
    return dict(information_cutoff_ns=sealed['information_cutoff_ns'],seal_ns=sealed['created_ns'],
        suite_ns=suite['created_ns'],prospective_audit_ns=prospective['audited_ns'],startup_ns=startup,
        seal_and_audit_before_startup=True)


def audit_one(suite_path,raw_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    suite,plan,sealed,prospective,refs=inputs(suite_path)
    inv=inventory(suite);raw_path=Path(raw_path).resolve();native_ref=receipt(raw_path)
    raw,scored_ref=scored_heldout(native_ref)
    cid=raw['candidate_id'];job=next((j for j in suite['jobs'] if j['candidate_id']==cid),None)
    if (job is None or raw_path!=Path(job['output']).resolve()/'raw.json'
            or raw.get('native_raw')!=native_ref or raw['plan']!=suite['sources']['plan']):
        raise ValueError('One exact declared immutable native attempt required')
    time_audit=chronology(raw,sealed,suite,prospective)
    candidate=next(c for c in plan['candidates'] if c['id']==cid)
    if candidate['resource_configuration']['mode']!='shared':raise ValueError('Shared independent collection required')
    mapped=flow_profiles.map_flow_collection(raw_path,allow_validation_diagnostics=True)
    if (len(mapped['traces'])!=1 or mapped['traces'][0].get('split')!='validation'
            or mapped['traces'][0].get('not_for_training') is not True
            or mapped['resource_evidence'].get('split')!='validation'
            or any(r.get('split')!='validation' for r in mapped['traces'][0]['requests'])):
        raise ValueError('Strict validation-only full cohort mapping required')
    catalog=read(checked(plan['catalog']))
    quality=verify_quality_annotation(raw,plan,catalog)
    mechanism=mechanism_audit.audit(raw_path)
    stability=flow_eligibility.native_stability(plan,candidate,raw,catalog)
    score=flow_metrics.score_flow(raw,{**plan,'resident_aliases':list(plan['aliases'])})
    count=len(mapped['traces'][0]['requests'])
    if count!=1200 or quality['count']!=1200 or not score['utility_complete']:
        raise ValueError('All 1200 observations, quality and six actual terms required')
    refs.extend(source_refs([raw['quality_evidence'],mapped['mapping_report'],mechanism]))
    refs.extend(receipt(p) for p in (raw_path,scored_ref['path'],flow_profiles.__file__,
        flow_eligibility.__file__,flow_metrics.__file__,mechanism_audit.__file__,
        Path(__file__).with_name('training_run_audit.py')))
    for ref in refs:checked(ref)
    value=dict(schema=SCHEMA,created_ns=time.perf_counter_ns(),source=receipt(__file__),
        suite=receipt(suite_path),raw=native_ref,scored_raw=scored_ref,plan=raw['plan'],candidate_id=cid,
        status='PASS' if stability['original_stability_passed'] else 'FAIL_ORIGINAL_STABILITY',
        request_count=count,quality=quality,chronology=time_audit,domain_inventory=inv,
        mapping_report=mapped['mapping_report'],semantic_trace_sha256=digest(mapped['traces'][0]),
        mechanism=mechanism,original_stability=stability,actual_six_term_score=score,
        source_receipts=refs,independent_validation=True,not_for_training=True,
        native_observations_modified=False,GPU_started=False,fitting_performed=False,
        prediction_gate_assessed=False,final_gain_eligible=False,
        scope='Collection identity/coverage/quality and original stability only; never predictor PASS')
    output.parent.mkdir(parents=True,exist_ok=True);save(output,value);return value


def complete(suite_path,audit_paths,output):
    if Path(output).exists():raise FileExistsError(output)
    suite,plan,sealed,prospective,refs=inputs(suite_path)
    inv=inventory(suite,complete=True);audits={}
    for path in audit_paths:
        audit=read(path);cid=audit['candidate_id']
        if cid in audits or cid not in inv['declared']:raise ValueError('Unique declared candidate audits required')
        if audit.get('schema')!=SCHEMA or audit.get('source')!=receipt(__file__) or audit.get('suite')!=receipt(suite_path):
            raise ValueError('Exact source-bound collection audit required')
        if (audit.get('request_count')!=1200 or audit.get('quality',{}).get('count')!=1200
                or audit.get('actual_six_term_score',{}).get('utility_complete') is not True
                or audit.get('status') not in ('PASS','FAIL_ORIGINAL_STABILITY')
                or audit.get('status')!=('PASS' if audit['original_stability']['original_stability_passed']
                    else 'FAIL_ORIGINAL_STABILITY')):
            raise ValueError('Complete 1200-request quality/score and original stability audit required')
        for ref in [audit['raw'],audit['scored_raw'],*audit['source_receipts']]:checked(ref)
        expected=next(Path(j['output']).resolve()/'raw.json' for j in suite['jobs'] if j['candidate_id']==cid)
        if audit['raw']!=receipt(expected):raise ValueError('Audit belongs to another attempt')
        chronology(read(expected),sealed,suite,prospective)
        audits[cid]=audit;refs.append(receipt(path))
    if set(audits)!=set(inv['declared']):raise ValueError('All four candidate audits required')
    count=sum(a['request_count'] for a in audits.values())
    value=dict(schema=SCHEMA+'.complete',created_ns=time.perf_counter_ns(),source=receipt(__file__),
        suite=receipt(suite_path),status='PASS' if all(a['status']=='PASS' for a in audits.values()) else 'FAIL_ORIGINAL_STABILITY',
        requested=4800,audited_requests=count,complete=count==4800,domain_inventory=inv,
        candidates=[dict(candidate_id=k,audit=receipt(p)) for p in audit_paths for k in [read(p)['candidate_id']]],
        source_receipts=refs,prediction_gate_assessed=False,GPU_started=False,fitting_performed=False,
        final_gain_eligible=False,scope='Four exclusive collection audits; no prediction or gain claim')
    Path(output).parent.mkdir(parents=True,exist_ok=True);save(output,value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('one');p.add_argument('--raw',required=True)
    q=sub.add_parser('complete');q.add_argument('--audit',action='append',required=True)
    for item in (p,q):
        item.add_argument('--suite',required=True);item.add_argument('--output',required=True)
    args=parser.parse_args()
    result=(audit_one(args.suite,args.raw,args.output) if args.op=='one'
        else complete(args.suite,args.audit,args.output))
    print(result['status'],result.get('request_count',result.get('audited_requests')),'GPU_started=False')


if __name__=='__main__':main()
