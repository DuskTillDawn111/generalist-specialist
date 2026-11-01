import copy
import json

from tests import test_cumulative_save_hazard_v14 as previous
from tools.predictor_validation_20261027.cumulative_save_hazard_v14 import CumulativeSaveHazardController as OldController
from tools.predictor_validation_20261027.cumulative_save_hazard_v15 import CumulativeSaveHazardController
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.profiling import digest

A=previous.previous.A
B=previous.previous.B
OMIT={'schema','controller_identity','state_sha256','empty_intervals','empty_audit_summary',
      'audit_scope_id','complete_empty_audit_external'}


def numerical(value):
    if isinstance(value,dict):return {k:numerical(v) for k,v in value.items() if k not in OMIT}
    if isinstance(value,list):return [numerical(v) for v in value]
    return value


class LedgerControllerTests(previous.ConditionalEntryHazardTests):
    def controller(self,**kwargs):
        return CumulativeSaveHazardController(receipt(self.path),allow_unvalidated_approximation=True,**kwargs)

    def pair(self,**kwargs):
        return (OldController(receipt(self.path),allow_unvalidated_approximation=True,**kwargs),self.controller(**kwargs))

    def replay(self,updates,*,frontier=A,restore=0,observation=0,**kwargs):
        old,new=self.pair(**kwargs);cursor=new.begin_audit_scope('equivalence')
        left=old.start_restore('holder',restore,frontier,'r',observation_ns=observation)
        right=new.start_restore('holder',restore,frontier,'r',observation_ns=observation)
        self.assertEqual(numerical(left),numerical(right))
        for clock,frontier in updates:
            left=old.update(left['state'],clock,frontier);right=new.update(right['state'],clock,frontier)
            self.assertEqual(numerical(left),numerical(right))
        exported=new.export_audit_since(cursor)
        self.assertEqual([{k:r[k] for k in ('start_ns','end_ns','policy')} for r in exported['events']],left['state']['empty_intervals'])
        self.assertEqual(right['state']['empty_audit_summary']['count'],len(left['state']['empty_intervals']))
        self.assertEqual(right['state']['empty_audit_summary']['total_ns'],sum(r['end_ns']-r['start_ns'] for r in left['state']['empty_intervals']))
        self.assertNotIn('empty_intervals',right['state'])
        return left,right,new,cursor

    def test_exact_v14_equivalence_across_age_bins_changes_and_terminal(self):
        self.replay([(1000,A),(2000,B),(5000,A),(10000,B),(20000,A),
                     (50000,B),(100000,A),(200000,B),(1000000,A),(2000000,[])])

    def test_exact_v14_equivalence_empty_pause_and_host_refresh(self):
        self.replay([(i,[] if i%2 else A) for i in range(1,101)],empty_frontier_policy='NO_QUERY_PAUSE')

    def test_exact_v14_equivalence_unknown_empty_history(self):
        self.replay([(1,[]),(2,[]),(3,A),(4,B),(5,[]),(100,A)],empty_frontier_policy='UNKNOWN')

    def test_exact_v14_equivalence_conditional_late_entry(self):
        self.replay([(88_000_001,[]),(88_000_002,B),(88_000_003,A),(89_000_000,A)],
                    observation=88_000_000,empty_frontier_policy='NO_QUERY_PAUSE')

    def test_seeded_uniform_draws_are_identical_to_v14(self):
        self.replay([(10,B),(100,A),(1000,[]),(2000,A)],budget_policy='SEEDED_UNIFORM',seed=42,
                    empty_frontier_policy='NO_QUERY_PAUSE')

    def test_scopes_isolate_repeated_restore_keys_and_cursors(self):
        controller=self.controller(empty_frontier_policy='NO_QUERY_PAUSE')
        first=controller.begin_audit_scope('candidate');r=controller.start_restore('holder',0,[],'same-key')
        r=controller.update(r['state'],10,[])
        second=controller.begin_audit_scope('candidate');q=controller.start_restore('holder',0,[],'same-key')
        q=controller.update(q['state'],20,[])
        one=controller.export_audit_since(first);two=controller.export_audit_since(second)
        self.assertEqual(one['empty_interval_count'],1);self.assertEqual(two['empty_interval_count'],1)
        self.assertEqual(one['total_empty_interval_ns'],10);self.assertEqual(two['total_empty_interval_ns'],20)
        self.assertNotEqual(one['events'][0]['scope_id'],two['events'][0]['scope_id'])
        with self.assertRaises(ValueError):controller.update(r['state'],30,[])
        broken=copy.deepcopy(second);broken['start_offset']=0
        with self.assertRaises(ValueError):controller.export_audit_since(broken)
        one['events'][0]['end_ns']=999
        self.assertEqual(controller.export_audit_since(first)['events'][0]['end_ns'],10)

    def test_full_append_only_audit_chain_and_fixed_state_field_shape(self):
        controller=self.controller(empty_frontier_policy='NO_QUERY_PAUSE')
        cursor=controller.begin_audit_scope('audit');r=controller.start_restore('holder',0,[],'r');shapes=[]
        for clock in range(1,301):
            r=controller.update(r['state'],clock,[])
            shapes.append((len(r['state']),len(r['state']['empty_audit_summary'])))
        self.assertEqual(len(set(shapes)),1)
        events=controller.export_audit_since(cursor)['events']
        self.assertEqual(len(events),300)
        for i,row in enumerate(events):
            self.assertEqual(row['ledger_entry_hash'],digest({k:v for k,v in row.items() if k!='ledger_entry_hash'}))
            if i:
                self.assertEqual(row['ledger_previous_hash'],events[i-1]['ledger_entry_hash'])
                self.assertEqual(row['state_previous_hash'],events[i-1]['state_interval_hash'])
        self.assertEqual(r['empty_audit_summary']['hashchain'],events[-1]['state_interval_hash'])
        json.dumps(controller.export_audit_since(cursor),allow_nan=False)

    def test_audit_export_and_scope_names_cannot_change_prediction_numbers(self):
        left=self.controller(empty_frontier_policy='NO_QUERY_PAUSE');right=self.controller(empty_frontier_policy='NO_QUERY_PAUSE')
        a=left.begin_audit_scope('candidate-a');right.begin_audit_scope('different-name')
        l=left.start_restore('holder',0,A,'r');r=right.start_restore('holder',0,A,'r')
        for clock in range(1,101):
            f=[] if clock%2 else B
            l=left.update(l['state'],clock,f);left.export_audit_since(a)
            r=right.update(r['state'],clock,f)
            self.assertEqual(numerical(l),numerical(r))
