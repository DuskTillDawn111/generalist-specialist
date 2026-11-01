"""Causal token-bucket exposure bound around the local queue routing policy.

The bound is a long-run offered routing rate with an explicit finite burst.
It does not prove physical GPU stability. No request is dropped or delayed:
when the limited path has no token, other resident candidates are compared.
"""
from __future__ import annotations

import copy
import math


class RateLimitedPolicy:
    def __init__(self, base_policy, guard):
        expected={'schema','task','family','rate_rps','burst'}
        if set(guard)!=expected or guard['schema']!='queue_aware_token_bucket_v1':
            raise ValueError('Explicit complete token bucket specification required')
        for name in ('rate_rps','burst'):
            value=guard[name]
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('Positive finite bucket rate and burst required')
        if guard['burst']<1:
            raise ValueError('Bucket must accommodate one request')
        self.base_policy=base_policy
        self.aliases=base_policy.aliases
        self.guard=copy.deepcopy(guard)
        self.reset()

    def reset(self):
        self.tokens=float(self.guard['burst'])
        self.last_refill_ns=None
        self.consumed_requests=0

    def choose(self,task,pool_id,now_ns,pending_snapshot):
        if type(now_ns) is not int or now_ns<0:
            raise ValueError('Nonnegative integer host time required')
        if self.last_refill_ns is not None and now_ns<self.last_refill_ns:
            raise ValueError('Token bucket clock must be monotonic')
        before=self.tokens;last=self.last_refill_ns
        elapsed=0. if last is None else (now_ns-last)/1e9
        refill=elapsed*self.guard['rate_rps']
        available=min(self.guard['burst'],before+refill)
        allowed=available>=1.
        decision=copy.deepcopy(self.base_policy.choose(task,pool_id,now_ns,pending_snapshot))
        for candidate in decision['audit']:
            limited=task==self.guard['task'] and candidate['family']==self.guard['family']
            candidate['rate_guard_applies']=limited
            if limited and not allowed and candidate.get('score') is not None:
                candidate.update(pre_guard_score=candidate['score'],score=None,status='RATE_LIMITED_CANDIDATE')
        comparable=[c for c in decision['audit'] if c.get('score') is not None]
        chosen=min(comparable,key=lambda c:(c['score'],c['predicted_completion_ms'],c['alias'])) if comparable else None
        consume=chosen is not None and task==self.guard['task'] and chosen['family']==self.guard['family']
        self.tokens=available-int(consume);self.last_refill_ns=now_ns
        self.consumed_requests+=int(consume)
        decision.update(alias=chosen['alias'] if chosen else None,
            status='CHOSEN' if chosen else 'UNKNOWN_NO_COMPARABLE_RESIDENT_ALIAS',
            trigger='MINIMUM_LOCAL_SURROGATE_WITH_RATE_GUARD' if chosen else 'NO_KNOWN_CANDIDATE_WITH_RATE_GUARD',
            rate_guard=dict(spec=copy.deepcopy(self.guard),bucket_before_tokens=before,
                last_refill_ns=last,elapsed_seconds=elapsed,refill_tokens=refill,
                available_tokens=available,allowed=allowed,consume=consume,
                bucket_after_tokens=self.tokens,consumed_requests=self.consumed_requests,
                fallbackaliases=[c['alias'] for c in comparable if c['family']!=self.guard['family']],
                exposure_bound='N_limited(s,t] <= burst + rate_rps*(t-s); long-run bound, not physical stability',
                requests_dropped=0,requests_delayed_by_guard=0))
        return decision
