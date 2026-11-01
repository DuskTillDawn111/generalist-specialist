"""CPU fixtures only; no model evidence or source timing fallback."""
import copy
import unittest

from gsdeploy.forecasting.kernel_baseline import GROUP_FIELDS
from tests.test_kernel_baseline_v2 import observation,estimator
from tools.predictor_validation_20261027.kernel_baseline_intake_v3 import json_ready
from tools.predictor_validation_20261027.kernel_baseline_query_v3 import hydrate_estimator,ExactSourceBaselineResolver


class QueryBridgeTests(unittest.TestCase):
    def setUp(self):
        original=estimator();groups=[]
        for key,models in original.models.items():
            groups.append(dict(group=dict(zip(GROUP_FIELDS,key[:-1])),perturbation_flags=list(key[-1]),models=json_ready(models)))
        self.fit=hydrate_estimator(dict(schema='gsdeploy.real_kernel_baseline_fit.v3',training_sha256=original.training_sha256,groups=groups))
        self.fit.allow_cpu_fixture=True
        for key in self.fit.groups:self.fit.groups[key]=[dict(evidence_kind='cpu_fixture')]
        self.row=observation(1);self.source=self.row['measurement_source']
        self.kernel=dict(source_sqlite=self.source,pid=100,context_id=1,correlation_id=2,nsys_start_ns=10,nsys_end_ns=99,
            name='fixture',block=[128,1,1],grid=[1,1,1],registers_per_thread=8,static_shared_bytes=0,dynamic_shared_bytes=0,cluster=None)
        launch=dict(launch={k:self.kernel[k] for k in ('pid','context_id','correlation_id','nsys_start_ns','nsys_end_ns')},
                    signature={k:self.kernel[k] for k in ('name','block','grid','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','cluster')})
        self.row['operator_to_kernel_evidence']=self.source
        self.resolver=ExactSourceBaselineResolver.__new__(ExactSourceBaselineResolver);self.resolver.fit=self.fit
        self.resolver.index={(self.source['sha256'],2,10):dict(query=self.row,launch=launch,source=self.source)}

    def test_hydrated_frozen_predictor_preserves_prediction(self):
        self.assertAlmostEqual(self.fit.predict(self.row)['targets']['duration_ns']['value'],140)

    def test_measurement_target_is_ignored(self):
        self.row['duration_ns']=999999
        result=self.resolver(self.kernel)
        self.assertAlmostEqual(result['duration_ns'],140)
        self.assertFalse(result['measured_target_fallback_used'])

    def test_missing_metadata_binding_unknown(self):
        self.kernel['correlation_id']=200
        self.assertFalse(self.resolver(self.kernel)['known'])

    def test_static_or_actual_launch_identity_change_unknown(self):
        for field,value in [('grid',[2,1,1]),('pid',101),('nsys_end_ns',100),('name','changed')]:
            kernel=copy.deepcopy(self.kernel);kernel[field]=value
            self.assertFalse(self.resolver(kernel)['known'])

    def test_omitted_name_is_read_from_exact_source_binding(self):
        self.kernel.pop('name');result=self.resolver(self.kernel)
        self.assertTrue(result['known']);self.assertEqual(result['kernel_name'],'fixture')

    def test_ordered_segment_identity_supported(self):
        static={k:v for k,v in self.kernel.items() if k not in ('source_sqlite','nsys_start_ns','nsys_end_ns','pid','correlation_id','context_id','name')}
        result=self.resolver(dict(kernel=static,timing_evidence=dict(source=self.source,identity=[10,99,100,7,2,0,1])))
        self.assertTrue(result['known'])


if __name__=='__main__':unittest.main()
