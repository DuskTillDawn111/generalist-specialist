"""Reviewed finite T8 continuation; prepare/audit are CPU, run is explicit.

Never starts or replaces its upstream processes. Root must separately launch
run with a source-bound conditional review after inspecting this source.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

from . import admission_training_v1 as training
from . import admission_capacity_v1 as capacity
from . import admission_resources_v1 as resources
from . import admission_quality_v1 as quality
from . import h600_selected_validation_v2 as selected
from . import h600_admission_pilot_v1 as pilot
from .native_flow import checked,receipt,save
from .probe_validation import read
from .profiling import digest

BASE=training.ROOT/'artifacts/predictor-gpu-20261027'
SCHEMA='gsdeploy.admission_T8_continuation.v1'
UPSTREAM=(('stage2',79375,'flow-validation-h600-selected-v16','execute-and-assess.py'),
          ('stage3',79980,'h600-admission-pilot-v1','continue-after-independent.py'))
POLICY=dict(jobs=8,requests=9600,training_seeds=list(training.SEEDS),routes=list(training.ROUTES),
    original_upstream_pids=[79375,79980],no_retries=True,no_seed_replacement=True,
    service_refit=False,only_training_launches=True,final_gain_eligible=False)
PROTOCOL_NAMES={'capacity_protocol':'admission-capacity-v1','resource_protocol':'admission-resources-v1',
    'quality_protocol':'admission-quality-v1'}
MODULES={'capacity_protocol':capacity,'resource_protocol':resources,'quality_protocol':quality}


def sources():
    return capacity.unique_refs([receipt(__file__),*training.sources(),*capacity.sources(),
        *resources.sources(),*quality.sources(),receipt(selected.__file__),receipt(pilot.__file__)])


def process_identity(pid):
    """PID plus Linux start tick avoids accepting a recycled PID."""
    path=Path('/proc')/str(pid)
    try:
        stat=(path/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        if fields[0]=='Z':return None
        cmdline=(path/'cmdline').read_bytes().split(b'\0')
        return dict(pid=pid,start_ticks=int(fields[19]),argv=[a.decode() for a in cmdline if a],
            cwd=str((path/'cwd').resolve()))
    except (FileNotFoundError,ProcessLookupError):return None


def _runs_script(info,script):
    wanted=Path(script).resolve()
    for arg in info['argv']:
        if Path(arg).name!=wanted.name:continue
        try:
            if (Path(info['cwd'])/arg).resolve()==wanted:return True
        except OSError:continue
    return False


def matching_processes(script):
    wanted=str(Path(script).resolve());found=[]
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        info=process_identity(int(folder.name))
        if info and _runs_script(info,wanted):found.append(info)
    return found


def upstream_status(spec):
    current=process_identity(spec['pid']);duplicates=matching_processes(spec['script']['path'])
    if any(p['pid']!=spec['pid'] for p in duplicates):raise ValueError('Duplicate upstream process; never launch training')
    if current is not None and current!=spec['process_identity']:
        raise ValueError('Upstream PID identity changed or was recycled')
    checked(spec['script'])
    if Path(spec['completed']).exists():return 'COMPLETED'
    if current is None:raise RuntimeError('Original upstream exited without completed artifact; no retry')
    return 'WAITING'


def _role(path):
    value=read(path)
    if (value.get('schema')!='gsdeploy.H600_admission_next_stage_role_reservation.v1'
            or value.get('training')!=dict(seeds=list(training.SEEDS),mode='admission2',
                routes=['balanced','skewed','spill_low','spill_high'],runs=8,requests=9600)
            or value.get('independent',{}).get('seed')!=628201
            or value.get('final_diagnostic',{}).get('seeds')!=[628301]
            or value.get('existing_shared_failure_preserved') is not True
            or value.get('qualified_positive_final_gain_eligible') is not False
            or value.get('original_scoring_and_thresholds_unchanged') is not True):
        raise ValueError('Original finite training/independent/final role reservation required')
    checked(value['service_hypothesis']['model']);return value


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    campaign_path=BASE/'admission-training-v1/campaign.json';campaign=training.verify(campaign_path)
    refs={key:receipt(BASE/folder/'protocol.json') for key,folder in PROTOCOL_NAMES.items()}
    for key,module in MODULES.items():
        if module.verify_protocol(refs[key]['path'])['campaign']!=receipt(campaign_path):
            raise ValueError('Every mode protocol must bind exact T8 campaign')
    role_path=BASE/'admission-next-stage-roles-v1.json';_role(role_path)
    upstream=[]
    for name,pid,folder,script in UPSTREAM:
        entry=BASE/folder/script;info=process_identity(pid)
        spec=dict(name=name,pid=pid,script=receipt(entry),process_identity=info,
            completed=str(BASE/folder/'completed.json'))
        if info is not None and not _runs_script(info,entry):
            raise ValueError('Original PID must run the exact retained upstream script')
        upstream_status(spec);upstream.append(spec)
    value=dict(schema=SCHEMA+'.protocol',created_ns=time.perf_counter_ns(),source=receipt(__file__),
        policy=POLICY,campaign=receipt(campaign_path),role_reservation=receipt(role_path),
        **refs,upstream=upstream,source_receipts=capacity.unique_refs([*sources(),*campaign['source_receipts'],
            *refs.values(),receipt(role_path),*[s['script'] for s in upstream]]),GPU_started=False)
    value['protocol_sha256']=digest(value);output.mkdir(parents=True,exist_ok=False)
    save(output/'protocol.json',value);audit_protocol(output/'protocol.json');return value


def audit_protocol(path):
    value=read(path)
    if (value.get('schema')!=SCHEMA+'.protocol' or value.get('source')!=receipt(__file__)
            or value.get('policy')!=POLICY or value.get('GPU_started') is not False
            or value.get('protocol_sha256')!=digest({k:v for k,v in value.items() if k!='protocol_sha256'})):
        raise ValueError('Changed continuation policy/source/protocol')
    capacity.unique_refs([*value['source_receipts'],value['campaign'],value['role_reservation']])
    campaign=training.verify(value['campaign']['path']);_role(value['role_reservation']['path'])
    for key,module in MODULES.items():
        if module.verify_protocol(checked(value[key]))['campaign']!=value['campaign']:
            raise ValueError('Mode protocol changed campaign binding')
    expected=[dict(name=n,pid=p,script=receipt(BASE/f/s),completed=str(BASE/f/'completed.json')) for n,p,f,s in UPSTREAM]
    if [{k:v for k,v in s.items() if k!='process_identity'} for s in value['upstream']]!=expected:
        raise ValueError('Exact original upstream dependencies required')
    required=[*sources(),*campaign['source_receipts'],*[value[k] for k in MODULES],value['role_reservation'],
        *[s['script'] for s in value['upstream']]]
    if not {(r['path'],r['sha256']) for r in required}<={(r['path'],r['sha256']) for r in value['source_receipts']}:
        raise ValueError('Continuation omitted required frozen sources')
    return value


def wait_upstream(protocol_path,*,poll_seconds=5.):
    if not 0<poll_seconds<=60:raise ValueError('Bounded positive poll interval required')
    while True:
        protocol=audit_protocol(protocol_path)
        statuses=[upstream_status(s) for s in protocol['upstream']]
        if all(s=='COMPLETED' for s in statuses):return
        print(json.dumps(dict(event='WAITING_FOR_ORIGINAL_UPSTREAM',statuses=statuses)),flush=True)
        time.sleep(poll_seconds)


def json_source_closure(values):
    """Capture immutable transitive JSON receipts once, then hash per job."""
    pending=training._refs(values);found={}
    while pending:
        ref=pending.pop();key=ref['path']
        if key in found:
            if found[key]!=ref:raise ValueError('Conflicting transitive source receipts')
            continue
        path=checked(ref);found[key]=ref
        if path.suffix=='.json':pending.extend(training._refs(read(path)))
    return capacity.unique_refs(list(found.values()))


def audit_upstream(protocol):
    """Recompute the original CPU gate and pilot report; no GPU or refit."""
    refs=[];packets={}
    for spec in protocol['upstream']:
        if upstream_status(spec)!='COMPLETED':raise ValueError('Both upstream completed artifacts required')
        ref=receipt(spec['completed']);packet=read(checked(ref))
        if packet.get('source')!=spec['script'] or packet.get('final_gain_eligible') is not False:
            raise ValueError('Upstream completed source/role mismatch')
        refs.extend(training._refs([ref,packet]));packets[spec['name']]=packet
    stage2=packets['stage2'];stage3=packets['stage3'];report2=read(checked(stage2['report']))
    gate=read(checked(stage2['gate']));report3=read(checked(stage3['report']))
    if (stage2['report']!=receipt(BASE/'flow-validation-h600-selected-v16/report/report.json')
            or stage2['gate']!=receipt(BASE/'flow-validation-h600-selected-v16/gate.json')
            or gate.get('report')!=stage2['report'] or gate.get('status') not in ('PASS','FAIL')
            or report2.get('candidate_coverage')!={'requested':4,'observed':4}
            or len(report2.get('candidates',[]))!=4 or any(c.get('known')!=1200 for c in report2['candidates'])
            or stage3['report']!=receipt(BASE/'h600-admission-pilot-v1/report/report.json')
            or report3.get('schema')!=pilot.SCHEMA+'.report'
            or {c.get('candidate_id') for c in report3.get('candidates',[])}!=set(pilot.IDS)
            or len(report3['candidates'])!=3
            or report3.get('status') not in ('DIAGNOSTIC_COMPLETE','DIAGNOSTIC_INCOMPLETE')):
        raise ValueError('Complete original stage2 and explicit all-three-mode stage3 reports required')
    with tempfile.TemporaryDirectory(prefix='admission-continuation-cpu-') as temp:
        recomputed=selected.gate(BASE/'flow-validation-h600-selected-v16/protocol.json',
            stage2['report']['path'],Path(temp)/'gate.json')
        if recomputed!=gate:raise ValueError('Stage2 gate differs from original audited recomputation')
        seal_path=checked(stage3['seal']);pilot.check_sealed(seal_path.parent)
        state=read(checked(stage3['run_state']))
        if (state.get('candidate_order')!=list(pilot.IDS) or state.get('status') not in
                ('COMPLETED','INCOMPLETE_DECLARED_MODES_SKIPPED','INCOMPLETE_NO_RETRY')
                or state.get('seal')!=stage3['seal'] or state.get('entry_source')!=receipt(pilot.__file__)
                or state.get('final_gain_eligible') is not False
                or len(state.get('completed',[]))!=len(set(state.get('completed',[])))
                or not set(state.get('completed',[]))<=set(pilot.IDS)):
            raise ValueError('Original finite pilot run-state required')
        paths=[BASE/'h600-admission-pilot-v1/runs'/cid/'raw.json' for cid in state['completed']]
        recomputed3=pilot.report(seal_path.parent,paths,Path(temp)/'pilot-report')
        if {k:v for k,v in recomputed3.items() if k!='created_ns'}!={k:v for k,v in report3.items() if k!='created_ns'}:
            raise ValueError('Stage3 report differs from native/source-bound recomputation')
    return dict(stage2_report=stage2['report'],stage2_gate=stage2['gate'],stage3_report=stage3['report'],
        stage2_gate_status=gate['status'],stage3_status=report3['status'],
        stage3_conclusion=report3['complete_three_mode_mechanism_conclusion'],
        insufficiencies=[dict(candidate_id=c['candidate_id'],complete=c['complete'],
            evidence_error=c['evidence_error'],activation=c['activation'],original_stability=c['original_stability'])
            for c in report3['candidates']],source_receipts=json_source_closure(refs))


def audit_root_review(protocol_path,review_path):
    protocol=audit_protocol(protocol_path);review=read(review_path)
    required={k:protocol[k] for k in ('campaign','capacity_protocol','resource_protocol','quality_protocol','role_reservation','source')}
    if (review.get('schema')!=SCHEMA+'.main_thread_review'
            or review.get('decision')!='APPROVED_FOR_CONDITIONAL_TRAINING_ONLY'
            or review.get('reviewer_role')!='main_thread' or review.get('protocol')!=receipt(protocol_path)
            or any(review.get(k)!=v for k,v in required.items())
            or type(review.get('created_ns')) is not int or review['created_ns']<=protocol['created_ns']
            or review.get('final_gain_eligible') is not False
            or not isinstance(review.get('stage3_incomplete_training_reason'),str)
            or not review['stage3_incomplete_training_reason'].strip()):
        raise ValueError('Root source-bound conditional review required; no implicit launch authorization')
    return protocol,review


def _fits(protocol,output):
    manifests={}
    cpu_env={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    original={k:os.environ.get(k) for k in cpu_env}
    try:
        os.environ.update(cpu_env)
        for key,module in MODULES.items():
            folder=output/(key.removesuffix('_protocol')+'-fit')
            module.fit(protocol[key]['path'],folder)
            module.verify_manifest(folder/'manifest.json')
            manifests[key]=receipt(folder/'manifest.json')
    finally:
        for key,value in original.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    return manifests


def run(protocol_path,review_path):
    """Explicit root launch only. One exclusive invocation; no retries/resume."""
    protocol_ref,root_review_ref=receipt(protocol_path),receipt(review_path)
    protocol,review=audit_root_review(protocol_path,review_path)
    output=Path(protocol_path).resolve().parent
    with (output/'continuation.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Continuation already running; no duplicate launch') from exc
        save(output/'started.json',dict(source=receipt(__file__),protocol=receipt(protocol_path),
            root_review=receipt(review_path),pid=os.getpid(),created_ns=time.perf_counter_ns(),GPU_started=False))
        wait_upstream(protocol_path)
        checked(protocol_ref);checked(root_review_ref)
        protocol,review=audit_root_review(protocol_path,review_path)
        upstream=audit_upstream(protocol)
        save(output/'upstream-audit.json',upstream)
        materialized=dict(schema=training.SCHEMA+'.main_thread_review',campaign=protocol['campaign'],
            decision='APPROVED_FOR_TRAINING_ONLY',reviewer_role='main_thread',created_ns=time.perf_counter_ns(),
            final_gain_eligible=False,root_authorization=receipt(review_path),continuation_protocol=receipt(protocol_path),
            continuation_source=receipt(__file__),stage3_incomplete_training_reason=review['stage3_incomplete_training_reason'],
            **{k:upstream[k] for k in ('stage2_report','stage2_gate','stage3_report')},
            **{k:protocol[k] for k in ('capacity_protocol','resource_protocol','quality_protocol','role_reservation')})
        training_review=output/'training-review.json';save(training_review,materialized)
        training_review_ref=receipt(training_review)
        training._review(protocol['campaign']['path'],training_review)
        campaign=training.verify(protocol['campaign']['path']);done=[]
        for job in campaign['jobs']:
            checked(protocol_ref);checked(root_review_ref)
            checked(training_review_ref)
            audit_root_review(protocol_path,review_path)
            capacity.unique_refs(upstream['source_receipts'])
            # Frozen execute independently rechecks every earlier job and its
            # native mapping before this next exclusive attempt.
            training.execute(protocol['campaign']['path'],job['job_id'],training_review)
            done.append(dict(job_id=job['job_id'],completion=receipt(Path(job['output'])/'training-complete.json')))
            save(output/f'job-{len(done):02d}-complete.json',done[-1])
        audit_root_review(protocol_path,review_path)
        checked(training_review_ref);capacity.unique_refs(upstream['source_receipts'])
        manifests=_fits(protocol,output)
        checked(protocol_ref);checked(root_review_ref)
        checked(training_review_ref);capacity.unique_refs(upstream['source_receipts'])
        audit_root_review(protocol_path,review_path)
        refs=capacity.unique_refs([receipt(protocol_path),receipt(review_path),receipt(training_review),
            receipt(output/'upstream-audit.json'),*upstream['source_receipts'],*protocol['source_receipts'],
            *manifests.values(),*[d['completion'] for d in done]])
        value=dict(schema=SCHEMA+'.completed',created_ns=time.perf_counter_ns(),source=receipt(__file__),
            protocol=receipt(protocol_path),root_review=receipt(review_path),upstream_audit=receipt(output/'upstream-audit.json'),
            training_review=receipt(training_review),jobs=done,job_count=len(done),manifests=manifests,
            upstream_statuses={k:upstream[k] for k in ('stage2_gate_status','stage3_status','stage3_conclusion')},
            source_receipts=refs,service_refitted=False,independent_or_final_launched=False,
            GPU_started=True,final_gain_eligible=False)
        save(output/'completed.json',value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='op',required=True)
    a=s.add_parser('prepare');a.add_argument('--output',required=True)
    a=s.add_parser('audit');a.add_argument('--protocol',required=True)
    a=s.add_parser('run');a.add_argument('--protocol',required=True);a.add_argument('--review',required=True)
    args=p.parse_args()
    if args.op=='prepare':value=prepare(args.output)
    elif args.op=='audit':value=audit_protocol(args.protocol)
    else:value=run(args.protocol,args.review)
    print(json.dumps(dict(schema=value['schema'],GPU_started=args.op=='run')))


if __name__=='__main__':main()
