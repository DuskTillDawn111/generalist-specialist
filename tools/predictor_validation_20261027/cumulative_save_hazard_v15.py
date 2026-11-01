"""v14 hazard arithmetic with fixed-size state and full external empty audit.

The append-only ledger is an audit side channel, never a prediction feature.
Each forecast has an explicit audit scope, so repeated restore keys do not mix
candidate histories. Frozen v14 solver code remains the numerical implementation.
"""
from __future__ import annotations

import copy

from .cumulative_save_hazard_v14 import CumulativeSaveHazardController as V14Controller
from .native_collect import receipt
from .profiling import digest

SCHEMA='gsdeploy.cumulative_SAVE_hazard_budget_candidate.v15'
AUDIT_SCHEMA='gsdeploy.scoped_empty_hazard_audit.v15'


class CumulativeSaveHazardController(V14Controller):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        baseline=copy.deepcopy(self._configuration)
        self._v14_budget_identity=self._identity
        self._sources=[*self._sources,receipt(__file__)]
        self._configuration=dict(schema=SCHEMA,sources=copy.deepcopy(self._sources),
            base_v14_configuration=baseline,base_v14_controller_identity=self._v14_budget_identity,
            seeded_budget_random_stream_identity=self._v14_budget_identity,
            empty_history='Complete append-only scoped ledger; fixed-size count/total/hashchain/last interval in state',
            audit_history_used_for_prediction=False,physical_execution_gate=False)
        self._identity=digest(self._configuration)
        self._audit_events=[];self._audit_scopes={};self._current_scope=None
        self._scope_ordinal=0;self._ledger_hash=digest(dict(schema=AUDIT_SCHEMA,controller_identity=self.identity))
        self.assert_sources_unchanged()

    def _budget(self,restore_key):
        # Preserve v14 seeded realizations as well as median arithmetic. Solver
        # provenance changes must not silently redraw the same declared sample.
        if self._budget_policy=='MEDIAN_LOG2':
            import math
            return math.log(2.)
        import math
        encoded=digest(dict(seed=self._seed,restore_key=restore_key,controller_identity=self._v14_budget_identity))
        return -math.log((int(encoded[:13],16)+.5)/2**52)

    def begin_audit_scope(self,scope_key):
        if not isinstance(scope_key,str) or not scope_key:raise ValueError('Explicit forecast audit scope key required')
        if self._current_scope is not None:
            self._audit_scopes[self._current_scope]['end_offset']=len(self._audit_events)
        ordinal=self._scope_ordinal;self._scope_ordinal+=1
        scope=digest(dict(controller_identity=self.identity,scope_key=scope_key,ordinal=ordinal))
        body=dict(schema=AUDIT_SCHEMA,controller_identity=self.identity,scope_id=scope,
                  scope_key=scope_key,scope_ordinal=ordinal,start_offset=len(self._audit_events))
        cursor=dict(body,cursor_sha256=digest(body))
        self._audit_scopes[scope]=dict(cursor=cursor,end_offset=None,hash_before=self._ledger_hash)
        self._current_scope=scope
        return copy.deepcopy(cursor)

    def export_audit_since(self,cursor):
        body={k:v for k,v in cursor.items() if k!='cursor_sha256'}
        scope=cursor.get('scope_id');metadata=self._audit_scopes.get(scope)
        if (cursor.get('cursor_sha256')!=digest(body) or metadata is None
                or cursor!=metadata['cursor'] or cursor.get('controller_identity')!=self.identity):
            raise ValueError('Source-bound audit cursor from this controller required')
        begin=cursor['start_offset'];end=metadata['end_offset']
        if end is None:end=len(self._audit_events)
        events=self._audit_events[begin:end]
        if any(r['scope_id']!=scope for r in events):raise ValueError('Audit scopes mixed')
        return dict(schema=AUDIT_SCHEMA,controller_identity=self.identity,cursor=copy.deepcopy(cursor),
            start_offset=begin,end_offset=end,empty_interval_count=len(events),
            total_empty_interval_ns=sum(r['end_ns']-r['start_ns'] for r in events),
            ledger_hash_before=metadata['hash_before'],
            ledger_hash_after=events[-1]['ledger_entry_hash'] if events else metadata['hash_before'],
            events=copy.deepcopy(events),complete_scope_export=True,audit_used_for_prediction=False)

    def _ensure_scope(self):
        if self._current_scope is None:self.begin_audit_scope('UNSCOPED_EXPLICIT_CONTROLLER_DEFAULT')

    def _check(self,state):
        if (state.get('schema')!=SCHEMA or state.get('controller_identity')!=self.identity
                or state.get('state_sha256')!=digest({k:v for k,v in state.items() if k!='state_sha256'})):
            raise ValueError('Sealed cumulative state bound to fixed v14/v15 source and solver required')
        if state.get('audit_scope_id')!=self._current_scope:
            raise ValueError('State belongs to a different forecast audit scope')
        return copy.deepcopy({k:v for k,v in state.items() if k!='state_sha256'})

    def _initial_summary(self,state):
        return dict(count=0,total_ns=0,hashchain=digest(dict(controller_identity=self.identity,
            scope_id=self._current_scope,restore_key=state['restore_key'],origin_restore_ns=state['restore_ns'],
            observation_ns=state['observation_ns'])),last_interval=None)

    def _append_empty_audit(self,state,interval):
        summary=state['empty_audit_summary'];previous=summary['hashchain']
        current=digest(dict(previous_hash=previous,interval=interval))
        entry=dict(ledger_offset=len(self._audit_events),scope_id=state['audit_scope_id'],
            scope_key=self._audit_scopes[state['audit_scope_id']]['cursor']['scope_key'],
            restore_key=state['restore_key'],held_owner=state['held_owner'],
            origin_restore_ns=state['restore_ns'],observation_ns=state['observation_ns'],
            start_ns=interval['start_ns'],end_ns=interval['end_ns'],policy=interval['policy'],
            state_interval_count=summary['count']+1,state_previous_hash=previous,state_interval_hash=current,
            ledger_previous_hash=self._ledger_hash)
        entry['ledger_entry_hash']=digest(entry);self._ledger_hash=entry['ledger_entry_hash']
        self._audit_events.append(entry)
        state['empty_audit_summary']=dict(count=summary['count']+1,
            total_ns=summary['total_ns']+interval['end_ns']-interval['start_ns'],
            hashchain=current,last_interval=copy.deepcopy(interval))

    def _integrate(self,state,end_ns):
        # v14 sees an ephemeral, initially empty list for this one update. Its
        # arithmetic, boundary handling and no-query semantics are unchanged.
        state['empty_intervals']=[]
        result=super()._integrate(state,end_ns)
        intervals=state.pop('empty_intervals')
        for interval in intervals:self._append_empty_audit(state,interval)
        return result

    def _result(self,state,old_review=None):
        self._ensure_scope()
        intervals=state.pop('empty_intervals',[])
        if intervals:raise ValueError('Unexported legacy interval history cannot be silently compacted')
        state['schema']=SCHEMA
        state.setdefault('audit_scope_id',self._current_scope)
        state.setdefault('empty_audit_summary',self._initial_summary(state))
        result=super()._result(state,old_review)
        result['schema']=SCHEMA
        result['empty_audit_summary']=copy.deepcopy(result['state']['empty_audit_summary'])
        result['audit_scope_id']=result['state']['audit_scope_id']
        result['complete_empty_audit_external']=True
        return result
