"""Fixed independent V4 launcher; explicit run is GPU work, audit is CPU only.

The frozen validation module owns prediction and actual scoring. This wrapper
only reserves one finite run, preserves UNKNOWN, and stops on collection failure.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import time

from . import admission_validation_v1 as validation
from .native_flow import checked,receipt,save
from .probe_validation import read

SCHEMA='gsdeploy.admission2_independent_validation_run.v1'
ORDER=validation.capacity.ROUTES


def checkpoint(path,value):
    """Atomically update only the mutable state; evidence uses exclusive save."""
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.state-',suffix='.json',delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def closure(refs):
    """Hash every transitive receipt once; reject conflicting path identities.

    Frozen execution snapshots also name their original Python sources as path
    strings. Bind those originals to the same hash as their frozen copies.
    """
    pending=list(refs);found={}
    while pending:
        ref=pending.pop();checked(ref)
        if ref['path'] in found:
            if ref!=found[ref['path']]:raise ValueError('Conflicting recursive source identity')
            continue
        found[ref['path']]=ref
        if Path(ref['path']).suffix=='.json':
            value=read(checked(ref));pending.extend(validation.training._refs(value))
            def originals(item):
                if isinstance(item,dict):
                    if 'original_path' in item and isinstance(item.get('frozen'),dict):
                        original=receipt(item['original_path'])
                        if original['sha256']!=item['frozen']['sha256']:
                            raise ValueError('Changed original execution snapshot source')
                        pending.append(original)
                    for child in item.values():originals(child)
                elif isinstance(item,list):
                    for child in item:originals(child)
            originals(value)
    return list(found.values())


def recheck(refs):
    for ref in refs:checked(ref)


def audit(sealed_dir):
    """Full fit/forecast proof once; callers rehash the complete closure per job."""
    folder=Path(sealed_dir).resolve()
    initial=[receipt(folder/'seal-audit.json'),receipt(folder/'predictions.json'),receipt(__file__)]
    value=validation.check_sealed(folder)
    protocol=validation.audit_protocol(checked(value['protocol']))
    sealed=read(checked(initial[1]));plan=read(checked(protocol['target_plan']))
    if tuple(c['id'] for c in plan['candidates'])!=ORDER:
        raise ValueError('Exact fixed independent four-route order required')
    refs=closure(initial+validation.training._refs([value,protocol,sealed,plan]))
    recheck(refs)
    statuses=[]
    for item in sealed['candidates']:
        complete=(item.get('unknown_reason') is None and len(item.get('requests',[]))==1200
            and all(r.get('finish_ns') is not None for r in item.get('requests',[]))
            and item.get('metrics') is not None
            and item['metrics'].get('absolute_utility_complete') is True)
        statuses.append(dict(candidate_id=item['candidate_id'],requested=1200,
            prediction_status='PREDICTED_COMPLETE' if complete else 'UNKNOWN',
            unknown_reason=item.get('unknown_reason')))
    return dict(protocol=protocol,sealed=sealed,plan=plan,source_closure=refs,prediction_statuses=statuses)


def run(sealed_dir,output,*,diagnostic=False):
    """Run all four fixed routes; UNKNOWN requires explicit diagnostic intent.

    Existing output is a permanent reservation. A failed attempt cannot resume,
    retry or replace a seed. Native collection keeps the original bounded drain.
    """
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Independent V4 reservation already exists; no retry')
    packet=audit(sealed_dir)
    claim=Path(packet['protocol']['target_plan']['path']).resolve().parent/'independent-V4-attempt.json'
    if claim.exists():raise FileExistsError('This public independent V4 already has an attempt; no retry under another output or seal')
    unknown=any(s['prediction_status']=='UNKNOWN' for s in packet['prediction_statuses'])
    if unknown and not diagnostic:
        raise ValueError('Prediction UNKNOWN: explicit diagnostic execution required; no qualification PASS')
    recheck(packet['source_closure'])
    output.mkdir(parents=True,exist_ok=False)
    save(claim,dict(schema=SCHEMA+'.attempt',created_ns=time.perf_counter_ns(),
        source=receipt(__file__),seal=receipt(Path(sealed_dir)/'seal-audit.json'),
        output=str(output),diagnostic=bool(diagnostic),candidate_order=list(ORDER),
        requested_requests=4800,not_for_training=True,final_gain_eligible=False))
    state=dict(schema=SCHEMA,created_ns=time.perf_counter_ns(),entry_source=receipt(__file__),
        seal=receipt(Path(sealed_dir)/'seal-audit.json'),plan=packet['protocol']['target_plan'],
        attempt_claim=receipt(claim),
        candidate_order=list(ORDER),requested_candidates=4,requested_requests=4800,
        prediction_statuses=packet['prediction_statuses'],diagnostic=bool(diagnostic),
        prediction_gate_status='UNKNOWN' if unknown else 'NOT_ASSESSED_UNTIL_INDEPENDENT_REPORT',
        source_closure=packet['source_closure'],attempted=[],completed=[],remaining=list(ORDER),
        status='RESERVED',GPU_started=False,no_retries=True,no_seed_replacement=True,
        not_for_training=True,final_gain_eligible=False)
    save(output/'reservation.json',copy.deepcopy(state))
    checkpoint(output/'run-state.json',state)
    try:
        for cid in ORDER:
            recheck(packet['source_closure'])
            state['attempted'].append(cid);state['status']='COLLECTING';state['GPU_started']=True
            save(output/('attempt-'+cid+'.json'),copy.deepcopy(state));checkpoint(output/'run-state.json',state)
            validation.training.native_flow.execute(packet['protocol']['target_plan']['path'],cid,output/cid)
            actual=validation.actual_evidence(output/cid/'raw.json',packet['protocol'],packet['sealed'])
            recheck(packet['source_closure'])
            state['completed'].append(dict(candidate_id=cid,native_raw=actual['native_raw'],
                scored_raw=actual['scored_raw'],known=actual['known'],requested=actual['requested'],
                original_stability=actual['original_stability']))
            state['remaining'].remove(cid)
            save(output/('complete-'+cid+'.json'),copy.deepcopy(state));checkpoint(output/'run-state.json',state)
        recheck(packet['source_closure']);state['status']='COLLECTED_DIAGNOSTIC' if diagnostic else 'COLLECTED_UNASSESSED'
    except BaseException as exc:
        state.update(status='INCOMPLETE_NO_RETRY',error=type(exc).__name__+':'+str(exc))
        checkpoint(output/'run-state.json',state)
        raise
    checkpoint(output/'run-state.json',state);return state


def report(sealed_dir,run_dir,output):
    """Report only independently complete actuals; declared V4 denominator stays."""
    run_dir=Path(run_dir).resolve();state=read(run_dir/'run-state.json')
    if (state.get('schema')!=SCHEMA or state.get('entry_source')!=receipt(__file__)
            or state.get('seal')!=receipt(Path(sealed_dir)/'seal-audit.json')
            or state.get('candidate_order')!=list(ORDER) or state.get('requested_requests')!=4800
            or state.get('requested_candidates')!=4 or state.get('final_gain_eligible') is not False):
        raise ValueError('Original finite independent run state required')
    claim=read(checked(state['attempt_claim']))
    if (state['attempt_claim']!=receipt(Path(state['plan']['path']).parent/'independent-V4-attempt.json')
            or claim.get('schema')!=SCHEMA+'.attempt' or claim.get('source')!=receipt(__file__)
            or claim.get('seal')!=state['seal'] or claim.get('output')!=str(run_dir)
            or claim.get('candidate_order')!=list(ORDER) or claim.get('requested_requests')!=4800):
        raise ValueError('Exact seal-bound exclusive independent attempt claim required')
    recheck(state['source_closure'])
    cids=[item['candidate_id'] for item in state['completed']]
    if cids!=list(ORDER[:len(cids)]):raise ValueError('Independent completed prefix/order required')
    paths=[]
    for item in state['completed']:
        path=run_dir/item['candidate_id']/'raw.json'
        if receipt(path)!=item['native_raw'] or receipt(path.with_name('scored-raw.json'))!=item['scored_raw']:
            raise ValueError('Changed independent completed observation')
        paths.append(path)
    value=validation.report(sealed_dir,paths,output)
    validation.verify_report(Path(output)/'report.json')
    save(Path(output)/'finite-run-status.json',dict(schema=SCHEMA+'.report_status',
        run_state=receipt(run_dir/'run-state.json'),reservation=receipt(run_dir/'reservation.json'),
        report=receipt(Path(output)/'report.json'),status=state['status'],requested_candidates=4,
        requested_requests=4800,attempted=state['attempted'],completed=cids,remaining=state['remaining'],
        prediction_gate_status=state['prediction_gate_status'],final_gain_eligible=False,not_for_training=True))
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    a=sub.add_parser('audit');a.add_argument('--sealed-dir',required=True)
    a=sub.add_parser('run');a.add_argument('--sealed-dir',required=True);a.add_argument('--output',required=True);a.add_argument('--diagnostic',action='store_true')
    a=sub.add_parser('report');a.add_argument('--sealed-dir',required=True);a.add_argument('--run-dir',required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.op=='audit':
        packet=audit(args.sealed_dir);value=dict(schema=SCHEMA+'.audit',source_closure_count=len(packet['source_closure']),
            prediction_statuses=packet['prediction_statuses'],GPU_started=False)
    elif args.op=='run':value=run(args.sealed_dir,args.output,diagnostic=args.diagnostic)
    else:value=report(args.sealed_dir,args.run_dir,args.output)
    print(json.dumps(value))


if __name__=='__main__':main()
