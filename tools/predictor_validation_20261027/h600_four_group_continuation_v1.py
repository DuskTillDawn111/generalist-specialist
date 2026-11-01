"""Conditional F8 diagnostic continuation, separately reviewed before run.

Prepare/audit are CPU only. Run waits for original T8/V4 processes and completed
evidence, then invokes frozen F8 seal/run once. No training or V launch exists.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib
import json
from pathlib import Path
import os
import time

from . import admission_training_continuation_v1 as training
from . import h600_four_group_run_v1 as execution
from .native_flow import checked,receipt,save
from .probe_validation import read
from .profiling import digest

BASE=execution.BASE
SCHEMA='gsdeploy.H600_F8_conditional_continuation.v1'
V_SCHEMA='gsdeploy.admission_V4_continuation.v1'
T_MODULE='tools.predictor_validation_20261027.admission_training_continuation_v1'
V_MODULE='tools.predictor_validation_20261027.admission_validation_continuation_v1'
T_PID=85446
POLICY=dict(training_pid=T_PID,final_seed=628301,final_configurations=8,requests=9600,
    independent_blocks=1,order=list(execution.final.ORDER),locks=execution.final.LOCKS,
    upstream='Exact original completed T8 and V4; never launch/restart predecessors',
    actual_evidence='Full V4 4800 actual requests and all actual six terms; prediction UNKNOWN stays UNKNOWN',
    no_retries=True,no_seed_replacement=True,no_training=True,no_independent_launch=True,
    diagnostic=True,statistical_significance_claim=False,final_gain_eligible=False)


def independent():
    return importlib.import_module(V_MODULE)


def sources():
    return execution.capacity.unique_refs([receipt(__file__),receipt(independent().__file__),*execution.sources()])


def runs_module(info,module):
    if info is None:return False
    argv=info['argv']
    return any(a=='-m' and i+1<len(argv) and argv[i+1]==module for i,a in enumerate(argv))


def runs_protocol(info,module,protocol,review):
    if not runs_module(info,module) or info.get('cwd')!=str(execution.final.ROOT):return False
    argv=info['argv'];tail=argv[argv.index('-m')+2:]
    if len(tail)!=5 or tail[0]!='run' or tail.count('--protocol')!=1 or tail.count('--review')!=1:return False
    for option,path in (('--protocol',protocol),('--review',review)):
        position=tail.index(option)
        if position+1>=len(tail) or (Path(info['cwd'])/tail[position+1]).resolve()!=Path(path).resolve():return False
    return True


def matching_processes(module):
    found=[]
    for folder in Path('/proc').iterdir():
        if folder.name.isdigit():
            info=training.process_identity(int(folder.name))
            if runs_module(info,module):found.append(info)
    return found


def process_status(spec):
    current=training.process_identity(spec['pid'])
    if any(p['pid']!=spec['pid'] for p in matching_processes(spec['module'])):
        raise ValueError('Duplicate upstream module process; final blocked')
    if current is not None and current!=spec['process_identity']:
        raise ValueError('Original upstream PID changed/recycled; no replacement')
    checked(spec['source'])
    if Path(spec['completed']).exists():return 'COMPLETED'
    if current is None:raise RuntimeError('Original upstream exited without completed evidence; no restart')
    return 'WAITING'


def _paths():
    return dict(plan=BASE/'four-group-diagnostic-v1/plan.json',
        training_protocol=BASE/'admission-training-continuation-v1/protocol.json',
        validation_protocol=BASE/'admission-validation-continuation-v1/protocol.json',
        role_reservation=BASE/'admission-next-stage-roles-v1.json',
        training_completed=BASE/'admission-training-continuation-v1/completed.json',
        validation_completed=BASE/'admission-validation-continuation-v1/completed.json')


def _inputs(refs):
    paths=_paths()
    for key in ('plan','training_protocol','validation_protocol','role_reservation'):
        if checked(refs[key]).resolve()!=paths[key].resolve():raise ValueError('Exact canonical F8 inputs required')
    plan=execution.final.audit_plan(refs['plan']['path'])
    t=training.audit_protocol(refs['training_protocol']['path'])
    v=independent().audit_protocol(refs['validation_protocol']['path'])
    if (plan['role_reservation']!=refs['role_reservation']
            or t['role_reservation']!=refs['role_reservation']
            or v.get('training_protocol')!=refs['training_protocol']):
        raise ValueError('F8 must follow the same T8/V4/reservation chain')
    return plan,t,v


def prepare(output,validation_pid):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if type(validation_pid) is not int or validation_pid<=0 or validation_pid==T_PID:
        raise ValueError('Actual distinct original V4 PID required before preparation')
    paths=_paths();refs={k:receipt(paths[k]) for k in ('plan','training_protocol','validation_protocol','role_reservation')}
    plan,t,v=_inputs(refs);specs=[]
    for name,pid,module in (('training',T_PID,T_MODULE),('validation',validation_pid,V_MODULE)):
        source=receipt(training.__file__ if name=='training' else independent().__file__)
        info=training.process_identity(pid)
        upstream_protocol=paths[name+'_protocol'];upstream_review=upstream_protocol.with_name('root-review.json')
        if (info is None or not runs_protocol(info,module,upstream_protocol,upstream_review)
                or name=='training' and info!=v.get('training_process_identity')):
            raise ValueError('Bind a live exact original module/protocol/review PID before prepare')
        spec=dict(name=name,pid=pid,module=module,source=source,process_identity=info,
            completed=str(paths[name+'_completed']))
        process_status(spec);specs.append(spec)
    value=dict(schema=SCHEMA+'.protocol',created_ns=time.perf_counter_ns(),source=receipt(__file__),
        policy=POLICY,**refs,upstream=specs,source_receipts=execution.capacity.unique_refs(
            execution.final.training._refs([sources(),refs,plan,t,v])),
        GPU_started=False,final_outcomes_read=False)
    value['protocol_sha256']=digest(value);output.mkdir(parents=True,exist_ok=False)
    save(output/'protocol.json',value);audit_protocol(output/'protocol.json');return value


def audit_protocol(path):
    value=read(path)
    if (value.get('schema')!=SCHEMA+'.protocol' or value.get('source')!=receipt(__file__)
            or value.get('policy')!=POLICY or value.get('GPU_started') is not False
            or value.get('final_outcomes_read') is not False
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Immutable F8 continuation source/policy/protocol required')
    execution.capacity.unique_refs(value['source_receipts']);plan,t,v=_inputs(value);paths=_paths()
    if len(value.get('upstream',[]))!=2:raise ValueError('Exactly original T8 and V4 dependencies required')
    for spec,name,module in zip(value['upstream'],('training','validation'),(T_MODULE,V_MODULE)):
        if (spec.get('name')!=name or spec.get('module')!=module
                or spec.get('completed')!=str(paths[name+'_completed'])
                or type(spec.get('pid')) is not int or spec['pid']<=0
                or (name=='training' and spec['pid']!=T_PID)
                or spec.get('source')!=receipt(training.__file__ if name=='training' else independent().__file__)
                or spec.get('process_identity',{}).get('pid')!=spec['pid']
                or not runs_protocol(spec['process_identity'],module,paths[name+'_protocol'],
                    paths[name+'_protocol'].with_name('root-review.json'))
                or name=='training' and spec['process_identity']!=v.get('training_process_identity')):
            raise ValueError('Exact original process identities and completion paths required')
    # Prepare binds declared source contracts without parsing historical or
    # future raw outcomes. Complete transitive numerical/native evidence is
    # authenticated once after both completed artifacts by audit_upstream.
    required=execution.final.training._refs([sources(),*[value[k] for k in
        ('plan','training_protocol','validation_protocol','role_reservation')],plan,t,v])
    if not {(r['path'],r['sha256']) for r in required}<={(r['path'],r['sha256']) for r in value['source_receipts']}:
        raise ValueError('F8 continuation omitted complete source closure')
    return value


def audit_root_review(protocol_path,review_path):
    protocol=audit_protocol(protocol_path);review=read(review_path)
    if (review.get('schema')!=SCHEMA+'.main_thread_review'
            or review.get('decision')!='APPROVED_FOR_CONDITIONAL_F8_DIAGNOSTIC_ONLY'
            or review.get('reviewer_role')!='main_thread' or review.get('protocol')!=receipt(protocol_path)
            or any(review.get(k)!=protocol[k] for k in
                ('source','plan','training_protocol','validation_protocol','role_reservation'))
            or type(review.get('created_ns')) is not int or review['created_ns']<=protocol['created_ns']
            or review.get('final_gain_eligible') is not False
            or review.get('preserve_prediction_failures_and_unknown') is not True
            or review.get('statistical_significance_claim') is not False):
        raise ValueError('Exact root source-bound conditional F8 diagnostic review required')
    return protocol,review


def wait_upstream(protocol_path,review_path,*,poll_seconds=5.):
    if not 0<poll_seconds<=60:raise ValueError('Bounded positive wait interval required')
    while True:
        protocol,_=audit_root_review(protocol_path,review_path)
        statuses=[process_status(s) for s in protocol['upstream']]
        if statuses==['COMPLETED','COMPLETED']:return
        print(json.dumps(dict(event='WAITING_FOR_ORIGINAL_T8_V4',statuses=statuses)),flush=True)
        time.sleep(poll_seconds)


def audit_upstream(protocol):
    paths=_paths();packets={};refs=[]
    for spec in protocol['upstream']:
        if process_status(spec)!='COMPLETED':raise ValueError('Original T8 and V4 completed artifacts required')
        ref=receipt(spec['completed']);value=read(checked(ref));refs.append(ref)
        expected=training.SCHEMA if spec['name']=='training' else V_SCHEMA
        if (value.get('schema')!=expected+'.completed' or value.get('source')!=spec['source']
                or value.get('protocol')!=protocol[spec['name']+'_protocol']
                or value.get('final_gain_eligible') is not False):
            raise ValueError('Upstream completed source/role/protocol changed')
        packets[spec['name']]=value
    t,v=packets['training'],packets['validation']
    independent().audit_root_review(checked(v['protocol']),checked(v['root_review']))
    if (v.get('training_completed')!=receipt(paths['training_completed'])
            or v.get('not_for_training') is not True or v.get('independent_runs_launched')!=4
            or v.get('native_requested')!=4800 or v.get('native_known')!=4800
            or v.get('final_launched') is not False or v.get('GPU_started') is not True
            or v.get('report')!=receipt(BASE/'admission-validation-v1/report/report.json')
            or v.get('gate')!=receipt(BASE/'admission-validation-v1/report/gate.json')
            or v.get('validation_audit')!=receipt(BASE/'admission-validation-v1/report/validation-audit.json')):
        raise ValueError('Complete fixed independent V4 chain required before F8')
    # Frozen final entry independently recomputes training manifests, native V4
    # report, all actual six terms and original gate; paired prediction gaps do
    # not hide a complete actual cohort or upgrade an incomplete prediction.
    prerequisite=execution._prerequisites(protocol['plan']['path'],v['report']['path'],str(paths['training_completed']))
    if (v.get('status')!='DIAGNOSTIC_COLLECTED' or v.get('collection_error') is not None
            or v.get('independent_gate_status')!=prerequisite['independent_gate_status']
            or read(checked(v['report_verification']))!=prerequisite['independent_report_verification']):
        raise ValueError('V4 completion changed original gate/report verification')
    refs=training.json_source_closure([refs,packets,prerequisite['source_receipts'],sources()])
    return dict(training_completed=receipt(paths['training_completed']),
        validation_completed=receipt(paths['validation_completed']),validation_report=v['report'],
        independent_gate=prerequisite['independent_gate'],independent_gate_status=prerequisite['independent_gate_status'],
        independent_report_verification=prerequisite['independent_report_verification'],source_receipts=refs)


def run(protocol_path,review_path):
    """Root launches once; all downstream collection remains diagnostic."""
    protocol_ref,review_ref=receipt(protocol_path),receipt(review_path)
    protocol,_=audit_root_review(protocol_path,review_path);output=Path(protocol_path).resolve().parent
    with (output/'continuation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('F8 continuation already running') from exc
        claim=Path(protocol['plan']['path']).parent/'F8-continuation-attempt.json'
        save(claim,dict(schema=SCHEMA+'.attempt',created_ns=time.perf_counter_ns(),source=receipt(__file__),
            protocol=protocol_ref,root_review=review_ref,pid=os.getpid(),output=str(output),no_retries=True))
        save(output/'started.json',dict(source=receipt(__file__),protocol=protocol_ref,review=review_ref,
            attempt=receipt(claim),created_ns=time.perf_counter_ns(),GPU_started=False))
        wait_upstream(protocol_path,review_path)
        checked(protocol_ref);checked(review_ref);protocol,_=audit_root_review(protocol_path,review_path)
        upstream=audit_upstream(protocol);save(output/'upstream-audit.json',upstream)
        for ref in upstream['source_receipts']:checked(ref)
        execution.seal(protocol['plan']['path'],upstream['validation_report']['path'],
            upstream['training_completed']['path'],output/'sealed')
        checked(protocol_ref);checked(review_ref);audit_root_review(protocol_path,review_path)
        for ref in upstream['source_receipts']:checked(ref)
        result=execution.run(output/'sealed/seal.json',output/'execution',diagnostic=True)
        checked(protocol_ref);checked(review_ref)
        for ref in upstream['source_receipts']:checked(ref)
        report_ref=result['report'];report=read(checked(report_ref))
        if (result.get('requested')!=9600 or result.get('not_for_training') is not True
                or result.get('final_gain_eligible') is not False
                or report.get('requested')!=9600 or report.get('independent_blocks')!=1
                or report.get('statistical_significance_claim') is not False
                or report.get('original_capacity_qualification')!='UNKNOWN_NOT_ESTABLISHED_BY_THIS_REPORT'):
            raise ValueError('Final diagnostic completion must preserve original finite scope')
        value=dict(schema=SCHEMA+'.completed',created_ns=time.perf_counter_ns(),source=receipt(__file__),
            protocol=protocol_ref,root_review=review_ref,attempt=receipt(claim),upstream_audit=receipt(output/'upstream-audit.json'),
            final_completed=receipt(output/'execution/completed.json'),final_report=report_ref,
            status=result['status'],known=result['known'],requested=9600,execution_error=result['execution_error'],
            independent_gate_status=upstream['independent_gate_status'],
            original_capacity_qualification=report['original_capacity_qualification'],
            descriptive_comparison=report['descriptive_comparison'],actual_audits=report['actual_audits'],
            invalid_attempts=report['invalid_attempts'],primary_value=report['primary_value'],
            source_receipts=training.json_source_closure([protocol_ref,review_ref,receipt(claim),upstream,report,result,sources()]),
            no_training_launched=True,no_independent_launched=True,not_for_training=True,
            independent_blocks=1,statistical_significance_claim=False,final_gain_eligible=False,GPU_started=True)
        save(output/'completed.json',value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True);a.add_argument('--validation-pid',type=int,required=True)
    a=sub.add_parser('audit');a.add_argument('--protocol',required=True)
    a=sub.add_parser('run');a.add_argument('--protocol',required=True);a.add_argument('--review',required=True)
    args=p.parse_args()
    if args.op=='prepare':value=prepare(args.output,args.validation_pid)
    elif args.op=='audit':value=audit_protocol(args.protocol)
    else:value=run(args.protocol,args.review)
    print(json.dumps(dict(schema=value['schema'],GPU_started=args.op=='run')))


if __name__=='__main__':main()
