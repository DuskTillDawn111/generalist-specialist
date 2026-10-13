"""CPU-only contract tests using the already committed preflight traces."""
import unittest
from .common import *
from .variants import make, flags_dict
from .report import classify_factor
from deployment_core.replicas import replica_key


class Contracts(unittest.TestCase):
    def test_flags_and_global_order(self):
        for w in read(F/'heldout_workloads.json')['workloads']:
            for method in NEW:
                tr=read(OUT/'preflight'/w['id']/method/'cached_only_trace.json')
                for iteration in tr['iterations']:
                    cs=iteration['candidates']
                    if method=='M01':self.assertFalse(any(c['mutation']['operation']=='SWAP' for c in cs))
                    else:
                        ranked=[c for c in cs if c['rank'] is not None]
                        expected=sorted(ranked,key=lambda c:(-c['delta_z'],c['mutation']['model']))
                        actual=sorted(ranked,key=lambda c:c['rank'])
                        self.assertEqual([c['mutation']['deployment'] for c in expected],[c['mutation']['deployment'] for c in actual])

    def test_native_code_local_binding(self):
        from method_v2.engine import RevisedGuidedSearch
        a=make('M10',None);b=make('M01',None)
        loopa=next(c.__dict__['search'] for c in type(a).mro() if c.__name__=='LocallyBoundNativeLoop')
        loopb=next(c.__dict__['search'] for c in type(b).mro() if c.__name__=='LocallyBoundNativeLoop')
        self.assertIs(loopa.__code__,RevisedGuidedSearch.search.__code__)
        self.assertIs(loopb.__code__,RevisedGuidedSearch.search.__code__)
        self.assertIsNot(loopa.__globals__,loopb.__globals__)

    def test_sharing_role_partition(self):
        rows=list(old_donors().values());r=next(r for r in rows if r['phase']=='search')
        self.assertEqual(identity(r['identity']['config'],r['workload_sha256']),r['identity'])
        self.assertNotEqual(digest(identity(r['identity']['config'],r['workload_sha256'])),digest(identity(r['identity']['config'],r['workload_sha256'],'confirmation_final')))

    def test_frozen_claim_rule(self):
        row=lambda status:{'outcome':status,'stability_regression':False,'severe_mean_regression':False}
        self.assertEqual(classify_factor([row('WIN')]*2+[row('TIE')]*10,True,True),'SUPPORTED')
        self.assertEqual(classify_factor([row('TIE')]*12,True,True),'NOT_SUPPORTED')
        self.assertEqual(classify_factor([row('WIN')]*2,True,False),'INSUFFICIENT')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    save_json(OUT/'adapter_validation.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'GPU_runs':0})
    if not result.wasSuccessful():raise SystemExit(1)
