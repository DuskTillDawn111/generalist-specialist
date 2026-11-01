"""Free fitted family/phase response slopes with frozen joint protocol vectors.

Preserves exact public input support and full request residual vectors; no
context-label lookup. Shares v3's explicitly unqualified protocol assumptions.
"""
from fractions import Fraction
import hashlib
import json
import math

from . import phase_heterogeneous_response_training_v1 as fitmodel
from . import joint_request_response_transport_candidate_v2 as original
from . import joint_request_protocol_transport_candidate_v3 as protocol
from .local_joint_macro_candidate_v1 import checked


def rate(owner,phase,state,fit):
    features=fitmodel.vector(owner,phase,state,fit['features'])
    actual=math.exp(-sum(x*b for x,b in zip(features,fit['coefficients'])))
    return Fraction(max(1,round(actual*original.RATE_DENOMINATOR)),original.RATE_DENOMINATOR)


def progress(macro,at,fit):
    original.require(macro['left_ns']<=at<=macro['right_ns'],'Mark outside actual occurrence macro')
    return sum((max(0,min(at,p['right_ns'])-p['left_ns'])*rate(macro['owner'],macro['semantic_phase'],p['state'],fit)
        for p in macro['causal_covariate_path']),Fraction(0))


def rate_table(fit):
    table={}
    for owner in ('text#0','text#1','generalist_2#0','ocr#0'):
        phases=('entry','ocr_inference','ready_tail') if owner=='ocr#0' else ('entry','prefill','decode','ready_tail')
        for phase in phases:
            for n in range(4):
                for same in range(n+1):table[owner,phase,n,same]=rate(owner,phase,dict(
                    peer_visible_macro_owner_count=n,peer_same_visible_macro_phase_owner_count=same),fit)
    return table


def build_law(documents,fit,*,original_phase_fit):
    original.require(fit.get('numerical_qualification') is True,'Full-rank numerically qualified heterogeneous fit required')
    original.require(set(fit['captures'])==set(original_phase_fit['captures']),
        'Heterogeneous response and joint donor law must use same whole captures')
    law=protocol.build_law(documents,original_phase_fit)
    docs={d['requests'][0]['macros'][0]['capture_sha256']:d for d in documents};raws={}
    for capture,doc in docs.items():
        review=checked(doc['sources'][0]);raw=checked(review['inputs']['raw.json']);raws[capture]={r['request_id']:r for r in raw['request_records']}
    baselines={tuple(b['group']):b['log_reference_ms'] for b in fit['baselines']}
    for vector,source in zip(law['vectors'],law['provenance_not_used_for_lookup']):
        capture=source['capture_sha256'];rid=source['request_id'];raw=raws[capture][rid]
        request=next(r for r in docs[capture]['requests'] if r['request_id']==rid)
        for node,macro in zip(vector['nodes'],request['macros']):
            original.require(node['macro_id']==macro['macro_id'],'Exact full macro node identity required')
            work=progress(macro,macro['right_ns'],fit);node['reference_work_ns']=original.fraction_record(work)
            shape=macro['forward_shape'] or {}
            group=(macro['owner'],macro['pool_id'],macro['task'],macro['macro_id'],macro['semantic_phase'],macro['kind'],
                *(shape.get(k) for k in ('input_sequence_length','input_batch_size','forward_index')))
            original.require(group in baselines,'Exact public input/shape baseline support required')
            node['baseline_reference_ms']=math.exp(baselines[group]);node['log_reference_work_residual']=math.log(float(work)/1e6)-baselines[group]
            for mark in node['marks']:
                if mark['identity']=='RESULT':at=raw['worker_response']['end_perf_ns']
                elif mark['identity'].startswith('event:'):at=raw['execution_events'][int(mark['identity'].split(':')[1])]['perf_ns']
                else:raise ValueError('Unqualified source protocol mark')
                mark['reference_work_at']=original.fraction_record(progress(macro,at,fit))
        vector['vector_id']=hashlib.sha256(json.dumps({k:v for k,v in vector.items() if k!='vector_id'},sort_keys=True).encode()).hexdigest()
        source['vector_id']=vector['vector_id']
    original_phase_beta=law.pop('beta')
    law.update(schema='gsdeploy.phase_heterogeneous_joint_protocol_candidate.v1',
        status='TRAINING_FAMILY_PHASE_RESPONSE_JOINT_PROTOCOL_CANDIDATE_ONLY',
        response_features=fit['features'],response_coefficients=fit['coefficients'],
        original_phase_beta_for_source_protocol_derivation_only=original_phase_beta,
        dispatch_contract='after_READY_processing reference hypothesis; busy runtime unqualified',
        candidate=fit['candidate'],context_or_job_lookup_used=False)
    return law
