"""CPU copy/memset resolver contract; fixture targets never supply predictions."""
import copy
import unittest

from tools.predictor_validation_20261027.copy_memset_baseline_v5 import ExactSourceCopyMemsetHypothesisResolver
from tools.predictor_validation_20261027.kernel_semantic_pooling_v5 import fit_group


def resolver():
    r=ExactSourceCopyMemsetHypothesisResolver.__new__(ExactSourceCopyMemsetHypothesisResolver)
    r.fit_source=dict(path='/cpu_fixture/fit.json',sha256='f'*64);r.cache={}
    rows=[dict(run_id=run,features={'actual_bytes':b},duration_ns=100+b) for run in ('cpu_fixture_1','cpu_fixture_2') for b in (10,20)]
    model=fit_group(rows,['actual_bytes']);model['fit_accepted']=False;model['reasons']=['TRAINING_RESIDUAL_EXCEEDS_25_PERCENT'];r.models={'fixture':model}
    source=dict(path='/cpu_fixture/source.sqlite',sha256='a'*64)
    identity=dict(rowid=1,kind='copy',pid=2,device=0,context_id=3,stream=4,correlation_id=5,start_ns=100,end_ns=99999999)
    memory=dict(bytes=10,copyKind=1,srcKind=0,dstKind=2)
    r.index={(source['sha256'],'copy',1):dict(source_sqlite=source,identity=identity,memory_metadata=memory,
        runtime_api='fixtureMemcpy',group_id='fixture',features={'actual_bytes':10},id='cpu_fixture_copy')}
    node=dict(id='copy:1',kind='copy',stream_key=[2,0,3,4],correlation_id=5,observed_start_ns=100,
        observed_end_ns=99999999,api_name='fixtureMemcpy',memory_metadata=memory,duration_ns=99999899)
    return r,node,source


class CopyResolverTests(unittest.TestCase):
    def test_numeric_uses_coefficients_not_query_measured_duration(self):
        r,node,source=resolver();p=r(node,source)
        self.assertAlmostEqual(p['duration_ns'],110);self.assertFalse(p['known']);self.assertTrue(p['numeric_hypothesis_available'])
        self.assertFalse(p['measured_target_fallback_used'])

    def test_kind_direction_and_bytes_cannot_be_changed(self):
        r,node,source=resolver();node=copy.deepcopy(node);node['memory_metadata']['copyKind']=2
        self.assertFalse(r(node,source)['numeric_hypothesis_available'])

    def test_source_identity_and_stream_checked(self):
        r,node,source=resolver();self.assertFalse(r(node,dict(source,path='/cpu_fixture/other.sqlite'))['numeric_hypothesis_available'])
        node['stream_key']=[2,0,3,5];self.assertFalse(r(node,source)['numeric_hypothesis_available'])

    def test_insufficient_training_rows_remain_unknown(self):
        r,node,source=resolver();r.models['fixture']={'reasons':['INSUFFICIENT_OBSERVATIONS_OR_RUNS']}
        p=r(node,source);self.assertFalse(p['numeric_hypothesis_available']);self.assertNotIn('duration_ns',p)

    def test_prediction_mutation_does_not_change_later_strict_evidence(self):
        r,node,source=resolver();p=r(node,source)
        p['prediction']['reasons'].clear()
        self.assertEqual(r(node,source)['strict_failure_reasons'],['TRAINING_RESIDUAL_EXCEEDS_25_PERCENT'])


if __name__=='__main__':unittest.main()
