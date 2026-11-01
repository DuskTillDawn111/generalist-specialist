"""CPU exact-source resolver fixtures; no target durations in query index."""
import copy
import unittest

from tools.predictor_validation_20261027.kernel_semantic_query_v5 import ExactSourceSemanticHypothesisResolver
from tools.predictor_validation_20261027.kernel_semantic_pooling_v5 import fit_group


def resolver():
    obj=ExactSourceSemanticHypothesisResolver.__new__(ExactSourceSemanticHypothesisResolver)
    obj.fit_source=dict(path='/cpu_fixture/fit.json',sha256='f'*64);obj._prediction_cache={}
    rows=[dict(run_id=run,features={'x':x},duration_ns=100+10*x) for run in ('cpu_fixture_1','cpu_fixture_2') for x in (1,2)]
    model=fit_group(rows,['x']);model['fit_accepted']=False;model['reasons']=['TRAINING_RESIDUAL_EXCEEDS_25_PERCENT']
    obj.models={'fixture_group':model};obj.groups={}
    signature=dict(name='fixture',block=[128,1,1],grid=[1,1,1],registers_per_thread=16,static_shared_bytes=0,dynamic_shared_bytes=0,cluster=None)
    identity=dict(pid=1,context_id=2,correlation_id=3,nsys_start_ns=4,nsys_end_ns=99999999,stream=5)
    source=dict(path='/cpu_fixture/trace.sqlite',sha256='a'*64)
    obj.index={(source['sha256'],3,4):dict(source=source,signature=signature,identity=identity,
        group_id='fixture_group',features={'x':1},observation_id='cpu_fixture_launch',feature_sources=[])}
    kernel=dict(signature,**identity,source_sqlite=source)
    return obj,kernel


class SemanticQueryTests(unittest.TestCase):
    def test_numeric_failed_fit_never_promoted_to_known_or_trace_duration(self):
        obj,kernel=resolver();r=obj(kernel)
        self.assertTrue(r['numeric_hypothesis_available']);self.assertFalse(r['known']);self.assertFalse(r['strict_known'])
        self.assertAlmostEqual(r['duration_ns'],110);self.assertFalse(r['measured_target_fallback_used'])
        self.assertIn('TRAINING_RESIDUAL_EXCEEDS_25_PERCENT',r['strict_failure_reasons'])

    def test_changed_receipt_and_static_geometry_rejected(self):
        obj,kernel=resolver();changed=copy.deepcopy(kernel);changed['source_sqlite']['path']='/cpu_fixture/other.sqlite'
        self.assertFalse(obj(changed)['numeric_hypothesis_available'])
        changed=copy.deepcopy(kernel);changed['grid']=[2,1,1]
        self.assertFalse(obj(changed)['numeric_hypothesis_available'])

    def test_query_without_name_recovers_exact_bound_source_name(self):
        obj,kernel=resolver();del kernel['name'];self.assertEqual(obj(kernel)['kernel_name'],'fixture')

    def test_missing_model_numeric_result_stays_unknown(self):
        obj,kernel=resolver();obj.models['fixture_group']={'reasons':['INSUFFICIENT_OBSERVATIONS_OR_RUNS']}
        r=obj(kernel);self.assertFalse(r['numeric_hypothesis_available']);self.assertNotIn('duration_ns',r)

    def test_ordered_segment_identity_maps_exactly(self):
        obj,kernel=resolver();segment=dict(kernel=kernel,timing_evidence=dict(source=kernel['source_sqlite'],identity=[4,99999999,1,5,3,0,2]))
        self.assertAlmostEqual(obj(segment)['duration_ns'],110)

    def test_cache_is_copied_and_full_feature_specific(self):
        obj,kernel=resolver();r=obj(kernel);r['prediction']['reasons'].append('mutation')
        self.assertNotIn('mutation',obj(kernel)['prediction']['reasons'])
        entry=next(iter(obj.index.values()));entry['features']['x']=2
        self.assertAlmostEqual(obj(kernel)['duration_ns'],120)


if __name__=='__main__':unittest.main()
