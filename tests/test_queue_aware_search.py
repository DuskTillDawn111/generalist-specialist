import io
import unittest
from tools.queue_aware_20261023.search import search,identity,measured_table_search,DiagnosticOracle,candidates

class SearchTests(unittest.TestCase):
    def test_safe_pruning_equals_full_dfs_and_reduces_calls(self):
        items=[{'id':i} for i in range(4)];values={0:[-.1,-.1],1:[-2.,-.3],2:[-.4,-.4],3:[-.7,-.7]}
        def evaluate(c,s):return dict(U=values[c['id']][s],eligible=True)
        dfs=search(items,[0,1],evaluate,lambda c:0.)
        bounded=search(items,[0,1],evaluate,lambda c:-.1 if c['id']==0 else (-.2 if c['id']==2 else 0.),algorithm='bounded')
        self.assertEqual(dfs['winner_id'],bounded['winner_id']);self.assertEqual(dfs['objective'],bounded['objective'])
        self.assertEqual(dfs['oracle_calls'],8);self.assertLess(bounded['oracle_calls'],8)
    def test_ties_and_unknown_do_not_certify_full_domain(self):
        items=[{'id':i} for i in range(3)]
        def evaluate(c,s):return dict(U=-.5,eligible=True) if c['id']<2 else dict(U=None,eligible=None)
        for algo in ('plain_dfs','bounded'):
            result=search(items,[1,2],evaluate,lambda c:0.,algorithm=algo)
            self.assertEqual(len(result['tie_ids']),2);self.assertFalse(result['whole_declared_domain_certified'])
    def test_false_eligibility_is_unknown_without_proof_in_measured_table(self):
        c={'id':1};doc=dict(candidates=[c],seeds=[1],records=[dict(candidate_id=identity(c),seed=1,utility=dict(eligible=False,U=-.4))])
        result=measured_table_search(doc);self.assertEqual(result['unknown_ids'],[identity(c)])
        self.assertIsNone(result['formal_U'])
    def test_complete_measured_six_term_required(self):
        c={'id':1};v={t+'_norm':.1 for t in 'QTLPME'};v.update(U=-.1,eligible=True,utility_complete=True,cohort_verified=True,constraints_verified=True,energy_verified=True,memory_verified=True,quality_identity_verified=True)
        doc=dict(candidates=[c],seeds=[1],records=[dict(candidate_id=identity(c),seed=1,utility=v)])
        self.assertEqual(measured_table_search(doc)['formal_U'],-.1)
        del v['E_norm'];self.assertIsNone(measured_table_search(doc)['formal_U'])
    def test_measured_tampered_utility_and_missing_identity_are_unknown(self):
        c={'id':1};v={t+'_norm':.1 for t in 'QTLPME'};v.update(U=-.1,eligible=True,utility_complete=True,cohort_verified=True,constraints_verified=True,energy_verified=True,memory_verified=True,quality_identity_verified=True)
        doc=dict(candidates=[c],seeds=[1],records=[dict(candidate_id=identity(c),seed=1,utility=v)])
        v['U']=-.2;self.assertIsNone(measured_table_search(doc)['formal_U'])
        v['U']=-.1;v['quality_identity_verified']=False;self.assertIsNone(measured_table_search(doc)['formal_U'])
    def test_real_domain_and_dynamic_bound(self):
        o=DiagnosticOracle([107,108]);items=list(candidates(o.o.catalog,o.o.config['workload']))
        self.assertEqual(len(items),7628)
        self.assertTrue(any(c['deployment']=={'generalist_2':1} for c in items))
        self.assertTrue(any('generalist_2' not in c['deployment'] for c in items))
        for strategy in ('queue_specialist_only','queue_quality','queue_latency'):
            c=next(c for c in items if c['kind']=='causal_queue' and c['deployment']=={'generalist_2':1,'text':2,'ocr':1} and c['strategy']==strategy)
            values=[o.evaluate(c,s) for s in (107,108)];self.assertTrue(all(v['U'] is not None for v in values))
            self.assertGreaterEqual(o.upper(c)+1e-12,sum(v['U'] for v in values)/2)
            self.assertIsNone(values[0]['formal_U'])
            if strategy!='queue_specialist_only':self.assertEqual(c['spill_guard']['rate_rps'],1.85/8)

if __name__=='__main__':unittest.main()
