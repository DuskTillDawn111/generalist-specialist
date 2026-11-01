"""H600 full16 validation suite entry with the new full-model and prospective audit.

Preparation and audit never open runtime output paths. Running is a distinct
explicit operation, checks all sealed inputs first, and stops on first failure.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time

from .h600_selected_validation_v2 import audit_plan,check_sealed, audit_protocol
from .flow_suite_audit import check_structure
from .h600_capacity_v1 import verify_seal
from .flow_validation import _sources
from .native_flow import checked,execute,receipt,save
from .probe_validation import read

SCHEMA='gsdeploy.independent_H600_selected_numerical_v2_native_flow_suite.v1'


def capacity_qualification(path,training_audit):
    """Adapt the verified capacity seal to the existing pure suite checker."""
    sealed=verify_seal(path)
    protocol=audit_protocol(training_audit['validation_protocol']['path'])
    if (sealed['protocol']!=protocol['capacity_protocol'] or sealed['target_plan']!=protocol['target_plan']):
        raise ValueError('H600 capacity and predictor must bind the same original reserved plan')
    return {**sealed,'target_seed':627701}


def audit_inputs(suite_path):
    """Audit source/qualification/cohort/time contracts without runtime reads."""
    suite=read(suite_path)
    if (suite.get('schema')!=SCHEMA or set(suite['sources'])!={'plan','predictions','training_audit','qualification','source'}
            or suite['sources']['source']!=receipt(__file__)):
        raise ValueError('Exact H600 full16 suite schema and generator source required')
    for ref in suite['sources'].values():checked(ref)
    folder=Path(suite['sealed_dir']);training_audit=check_sealed(folder)
    if (suite['sources']['predictions']!=receipt(folder/'predictions.json')
            or suite['sources']['training_audit']!=receipt(folder/'training-audit.json')):
        raise ValueError('H600 full16 suite inputs differ from sealed directory')
    sealed=read(suite['sources']['predictions']['path'])
    if sealed['sources']['plan']!=suite['sources']['plan']:
        raise ValueError('H600 full16 suite plan differs from sealed predictions')
    plan=audit_plan(checked(suite['sources']['plan']),training_audit['validation_protocol']['path'])
    qualification=capacity_qualification(checked(suite['sources']['qualification']),training_audit)
    refs=[*suite['sources'].values(),*sealed['sources'].values(),*sealed['training_sources'],
        *sealed['resource_sources'],*sealed['execution_sources'],*sealed['quality_sources'],
        *sealed['predictor_sources'],sealed['tool_source'],sealed['metric_source'],*training_audit['source_receipts'],
        *[receipt(Path(__file__).with_name(name)) for name in
            ('flow_suite_audit.py','h600_capacity_v1.py','h600_selected_validation_v2.py')]]
    for ref in refs:checked(ref)
    for name,source_name in (('resources','resource_sources'),('quality','quality_sources')):
        doc=read(sealed['sources'][name]['path']);embedded=_sources(doc)
        if embedded!=sealed[source_name] or doc.get('split','training')!='training':
            raise ValueError('H600 full16 suite omitted or changed training-only '+name+' sources')
        for ref in embedded:checked(ref)
    checks=check_structure(plan,sealed,suite,qualification,suite_path)
    return {'schema':'gsdeploy.independent_h600_flow_prospective_input_audit.v1','status':'PASS',
        'audited_ns':time.perf_counter_ns(),'suite':receipt(suite_path),'source':receipt(__file__),
        'source_receipts':list({(r['path'],r['sha256']):r for r in refs}.values()),
        'checks':checks,'training_verification':training_audit['verification'],
        'runtime_paths_opened':False,'runtime_outputs_or_logs_read':False,'GPU_started_by_audit':False,
        'actual_startup_chronology_assessed':False,'execution_success_assessed':False,
        'physical_capacity_bound_certified':False,'final_gain_eligible':False}


def prepare(plan_path,sealed_dir,qualification_path,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    training_audit=check_sealed(sealed_dir)
    plan=audit_plan(plan_path,training_audit['validation_protocol']['path'])
    sealed=read(Path(sealed_dir)/'predictions.json')
    if sealed['sources']['plan']!=receipt(plan_path):raise ValueError('H600 full16 plan differs from sealed predictions')
    declared={c['id'] for c in plan['candidates']};predicted=[p['candidate_id'] for p in sealed['candidates']]
    if len(predicted)!=len(set(predicted)) or set(predicted)!=declared:
        raise ValueError('Exactly one prediction per declared H600 full16 suite candidate required')
    if any(p['unknown_reason'] or not p['metrics'] or not p['metrics']['six_term_comparison_complete']
            or any(r['e2e_ms'] is None for r in p['requests']) for p in sealed['candidates']):
        raise ValueError('Complete H600 full16 declared-domain predictions required')
    qualification=capacity_qualification(qualification_path,training_audit)
    jobs=[{'candidate_id':c['id'],'output':str(output/'runs'/c['id'])} for c in plan['candidates']]
    random.Random('independent-flow-validation-order:'+str(plan['seed'])).shuffle(jobs)
    value={'schema':SCHEMA,'created_ns':time.perf_counter_ns(),
        'sources':{'plan':receipt(plan_path),'predictions':receipt(Path(sealed_dir)/'predictions.json'),
            'training_audit':receipt(Path(sealed_dir)/'training-audit.json'),
            'qualification':receipt(qualification_path),'source':receipt(__file__)},
        'sealed_dir':str(Path(sealed_dir).resolve()),'jobs':jobs,'no_retries':True,
        'no_seed_replacement':True,'GPU_started':False}
    # Pure complete-domain/time checks before creating the exclusive output.
    check_structure(plan,sealed,value,qualification,output/'suite.json')
    output.mkdir(parents=True,exist_ok=False);save(output/'suite.json',value)
    audit=audit_inputs(output/'suite.json');save(output/'prospective-input-audit.json',audit)
    return value


def run(path):
    # No runtime path is inspected until all prospective checks succeed.
    audit_inputs(path);suite=read(path)
    for index,job in enumerate(suite['jobs']):
        if index:audit_inputs(path)
        print(json.dumps({'event':'starting_independent_h600_validation_run','candidate':job['candidate_id']}),flush=True)
        raw=execute(suite['sources']['plan']['path'],job['candidate_id'],job['output'])
        if raw['status']!='completed' or raw.get('qualification_issues'):
            raise RuntimeError('H600 full16 independent collection failed; stop without retry or replacement')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare')
    for field in ('plan','sealed-dir','qualification','output'):a.add_argument('--'+field,required=True)
    a=sub.add_parser('audit');a.add_argument('--suite',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('run');a.add_argument('--suite',required=True)
    args=p.parse_args()
    if args.op=='prepare':value=prepare(args.plan,args.sealed_dir,args.qualification,args.output)
    elif args.op=='audit':
        if Path(args.output).exists():raise FileExistsError(args.output)
        value=audit_inputs(args.suite);save(args.output,value)
    else:run(args.suite);return
    print(json.dumps({'status':value.get('status','PREPARED'),'GPU_started':False}))


if __name__=='__main__':main()
