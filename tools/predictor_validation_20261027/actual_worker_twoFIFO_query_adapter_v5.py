"""Bind a v5 world view to the complete public cohort and original boundaries."""
from fractions import Fraction
import hashlib
import json

from .actual_worker_twoFIFO_reference_v5 import SCHEMA,resource_timeline_view


def clock(value):
    if type(value) is dict and set(value)=={'numerator','denominator'}:
        if any(type(v) is not int for v in value.values()) or value['denominator']<=0:
            raise ValueError('Exact public boundary clock required')
        value=Fraction(value['numerator'],value['denominator'])
    if type(value) not in (int,Fraction) or value<0:
        raise ValueError('Exact public boundary clock required')
    return Fraction(value)


def canonical_requests(rows):
    result=[];ids=set()
    for row in rows:
        if (type(row) is not dict or set(row)!={'request_id','owner','task','pool_id','arrival_ns'}
                or any(type(row[k]) is not str or not row[k] for k in ('request_id','owner','task','pool_id'))
                or row['request_id'] in ids):
            raise ValueError('Complete distinct public cohort required')
        ids.add(row['request_id']);arrival=clock(row['arrival_ns'])
        result.append(dict(row,arrival_ns=[arrival.numerator,arrival.denominator]))
    return result


def query_bound_view(reference,public_requests,*,window_start_ns,window_end_ns,drain_end_ns):
    if reference.get('schema')!=SCHEMA:
        raise ValueError('Explicit v5 two-FIFO world required')
    public=canonical_requests(public_requests)
    expected=canonical_requests(reference['public_offered_requests'])
    if public!=expected or len(public)!=reference['offered_requests']:
        raise ValueError('Public cohort/order/route/input/arrival differs from generated world')
    supplied=list(map(clock,(window_start_ns,window_end_ns,drain_end_ns)))
    actual=list(map(clock,(*reference['original_window'],reference['original_drain_end_ns'])))
    if supplied!=actual:
        raise ValueError('Original public window/drain differs from generated world')
    fingerprint=dict(public_requests=public,
        original_window_and_drain_ns=[[v.numerator,v.denominator] for v in supplied])
    digest=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    view=dict(reference)
    view.update(query_fingerprint_sha256=digest,public_query_identity_verified=True,
        source_FIFO_order_match_only=True,query_identity_confers_native_qualification=False)
    return view


def query_bound_resource_view(reference,public_requests,**boundaries):
    return resource_timeline_view(query_bound_view(reference,public_requests,**boundaries))
