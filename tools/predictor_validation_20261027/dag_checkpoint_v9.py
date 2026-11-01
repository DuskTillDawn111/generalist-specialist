"""v9 checkpoints bind request-start supply in addition to the complete v7 DAG.

The v7 structural/resource validator remains authoritative; v9 adds its source,
the start-provider callable/configuration/evidence, and started-bound checks.
No observed CUDA progress or fabricated cutoff edge enters this contract.
"""
from __future__ import annotations

import copy
import inspect
import math

from gsdeploy.forecast_errors import UnknownEvidence
from . import dag_checkpoint_v7 as base

STATE_FIELDS=base.STATE_FIELDS
decode=base.decode
encode=base.encode
receipt=base.receipt
SCHEMA='gsdeploy.simulated_dag_atomic_checkpoint.v9'


def request_start_bound(engine,row,*,owner_ready_ns=None):
    provider=engine.request_start_provider
    if provider is None:
        return max(row['arrival_ns'],owner_ready_ns if owner_ready_ns is not None else row['arrival_ns'])
    if owner_ready_ns is not None and hasattr(provider,'from_owner_ready'):
        value=provider.from_owner_ready(copy.deepcopy(row),owner_ready_ns=owner_ready_ns)
    else:
        value=provider(copy.deepcopy(row))
    if type(value) is not int or value<row['arrival_ns']:
        raise UnknownEvidence('TRAINED_OR_OBSERVED_ABSOLUTE_REQUEST_START_SUPPLY_REQUIRED:'+row['request_id'])
    return max(value,owner_ready_ns if owner_ready_ns is not None else value)


def contract(engine,requests,now_ns,deadline_ns,drain_deadline_ns):
    result=base.contract(engine,requests,now_ns,deadline_ns,drain_deadline_ns)
    sources={r['path']:r for r in result['sources']}
    ref=receipt(__file__);sources[ref['path']]=ref
    provider=engine.request_start_provider
    if provider is not None:
        target=provider if inspect.isfunction(provider) or inspect.ismethod(provider) else type(provider)
        path=inspect.getsourcefile(target)
        if path is None:
            raise ValueError('Request-start provider implementation source required')
        ref=receipt(path);sources[ref['path']]=ref
    result['sources']=[sources[p] for p in sorted(sources)]
    result['providers']['request_start']=base.callable_identity(provider)
    result['providers']['request_start_from_owner_ready']=(
        base.callable_identity(provider.from_owner_ready) if provider is not None and hasattr(provider,'from_owner_ready') else None)
    result['request_start_contract']='Pure absolute not-before START; original public arrival and owner FIFO unchanged'
    return result


def export_checkpoint(state,diagnostic,binding):
    result=base.export_checkpoint(state,diagnostic,binding)
    result['schema']=SCHEMA
    result['checkpoint_sha256']=base.digest({k:v for k,v in result.items() if k!='checkpoint_sha256'})
    return result


def import_checkpoint(checkpoint,engine,binding):
    if (checkpoint.get('schema')!=SCHEMA or checkpoint.get('checkpoint_sha256')!=
            base.digest({k:v for k,v in checkpoint.items() if k!='checkpoint_sha256'})):
        raise ValueError('Changed/foreign v9 request-start checkpoint')
    projected=copy.deepcopy(checkpoint);projected['schema']=base.SCHEMA
    projected['checkpoint_sha256']=base.digest({k:v for k,v in projected.items() if k!='checkpoint_sha256'})
    state=base.import_checkpoint(projected,engine,binding)
    for field in ('finishes','receipt_pending'):
        for rid,record in state[field].items():
            original=state['cohort'][rid]
            delays=(dict(parent_receipt_ms=0.,worker_next_ready_ms=0.) if engine.lifecycle_provider is None
                else engine.lifecycle_provider.delays(copy.deepcopy(original),engine.aliases,engine.scenario))
            if engine.lifecycle_provider is not None and delays.get('known') is not True:
                raise ValueError('Checkpoint requires known bound joint lifecycle delays')
            for source,target in (('parent_receipt_ms','finish_ns'),('worker_next_ready_ms','worker_next_ready_ns')):
                value=delays.get(source)
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError('Checkpoint requires finite nonnegative bound lifecycle delays')
                if record[target]!=record['handler_end_ns']+round(value*1e6):
                    raise ValueError('Checkpoint joint RESULT/READY differs from bound lifecycle provider')
    ready_by_request={rid:r['worker_next_ready_ns'] for field in ('finishes','receipt_pending')
                      for rid,r in state[field].items()}
    order={rid:i for i,rid in enumerate(state['cohort'])}
    for owner in engine.aliases:
        previous_ready=binding['original_now_ns']
        rows=sorted((at,order[rid],rid) for rid,at in state['starts'].items() if state['cohort'][rid]['alias']==owner)
        expected=sorted((r['arrival_ns'],order[rid],rid) for rid,r in state['cohort'].items() if r['alias']==owner)
        if [rid for _,_,rid in rows]!=[rid for _,_,rid in expected[:len(rows)]]:
            raise ValueError('Checkpoint historical owner START order differs from original FIFO')
        for index,(at,_,rid) in enumerate(rows):
            if at<request_start_bound(engine,state['cohort'][rid],owner_ready_ns=previous_ready):
                raise ValueError('Checkpoint START precedes bound request-start host supply')
            if index+1<len(rows) and rid not in ready_by_request:
                raise ValueError('Later owner START lacks its predecessor explicit predicted READY')
            previous_ready=ready_by_request.get(rid,previous_ready)
    return state
