"""CPU-only adapter edge cases; no profiler/export/CUDA or numerical fit."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import unified_kernel_dag_intake_v1 as m


class UnifiedIntakeTests(unittest.TestCase):
    def fixture(self):
        kernels=[dict(nsys_start_ns=a,nsys_end_ns=b,pid=7,stream=s,correlation_id=i,
                      device=0,context_id=1) for i,(a,b,s) in enumerate([(1,4,7),(2,3,13)])]
        raw=dict(ready={'ocr':{'pid':7}},request_records=[dict(request_id='r',model='ocr')])
        parsed=dict(clock_mappings={'7':{'valid':True,'offset_ns':999999999}},
            nvtx_ranges=[dict(stage=['r','gpu','infer'],kernels=kernels)],unattributed_kernels=[])
        return raw,parsed,{'path':'native.sqlite','sha256':'s'*64}

    def test_canonical_same_owner_multistream_retained_without_mapped_timestamps(self):
        raw,parsed,ref=self.fixture();requests,inventory=m.canonical_requests(raw,parsed,ref)
        self.assertEqual([k['stream'] for k in requests['r']['kernels']],[7,13])
        self.assertEqual([k['nsys_start_ns'] for k in requests['r']['kernels']],[1,2])
        self.assertEqual(inventory,dict(full_session_kernels=2,outside_measured_kernels=0,
                                       outside_measured_overlap_wall_ns=0))

    def test_duplicate_native_kernel_cannot_double_weight_phases(self):
        raw,parsed,ref=self.fixture();parsed['unattributed_kernels']=[copy.deepcopy(parsed['nvtx_ranges'][0]['kernels'][0])]
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.canonical_requests(raw,parsed,ref)

    def test_missing_owner_wrong_pid_and_duplicate_offered_registry_are_rejected(self):
        for mode in ('missing','pid','duplicate'):
            raw,parsed,ref=self.fixture()
            if mode=='missing':parsed['nvtx_ranges'][0]['stage'][0]='foreign'
            elif mode=='pid':parsed['nvtx_ranges'][0]['kernels'][0]['pid']=8
            else:raw['request_records']*=2
            with self.subTest(mode=mode),self.assertRaises(ValueError):m.canonical_requests(raw,parsed,ref)

    def test_unmeasured_kernel_peer_overlap_uses_half_open_native_domain(self):
        for start,end,expected in ((0,1,0),(4,5,0),(0,2,1),(3,5,1),(2,3,1)):
            raw,parsed,ref=self.fixture()
            other=dict(parsed['nvtx_ranges'][0]['kernels'][0],nsys_start_ns=start,nsys_end_ns=end,correlation_id=55)
            parsed['unattributed_kernels']=[other]
            _,inventory=m.canonical_requests(raw,parsed,ref)
            self.assertEqual(inventory['outside_measured_overlap_wall_ns'],expected)

    def test_invalid_native_intervals_cannot_cancel_outside_peer_overlap(self):
        for start,end in ((3,2),(2,2),(-1,2),(True,2),(1,2.5)):
            for target in ('measured','outside'):
                raw,parsed,ref=self.fixture()
                invalid=dict(parsed['nvtx_ranges'][0]['kernels'][0],nsys_start_ns=start,
                             nsys_end_ns=end,correlation_id=56)
                if target=='outside':
                    parsed['unattributed_kernels']=[dict(invalid,nsys_start_ns=1,nsys_end_ns=4,correlation_id=55),invalid]
                else:parsed['nvtx_ranges'][0]['kernels'][0]=invalid
                with self.subTest(target=target,start=start,end=end),self.assertRaisesRegex(ValueError,'positive exact integer'):
                    m.canonical_requests(raw,parsed,ref)

    def static_fixture(self):
        kernel=dict(source_sqlite={'sha256':'exact'},nsys_start_ns=1,nsys_end_ns=2,pid=7,
                    context_id=1,correlation_id=55,name='exact-name',block=[32,1,1],grid=[1,1,1],
                    registers_per_thread=8,static_shared_bytes=0,dynamic_shared_bytes=0,cluster=None)
        return [dict(nodes=[dict(kind='gpu',kernel=kernel)])],{'hardware_id':'device'}

    def test_static_missing_or_different_source_never_borrows_configuration(self):
        for source in (None,'different','exact'):
            profiles,hardware=self.static_fixture()
            resolver=SimpleNamespace(hardware=hardware,source_receipts=[],index={})
            if source:
                resolver.index[(source,1,2,7,1,55)]=None if source=='exact' else {'allocated_shared_bytes':1}
            with patch.object(m,'AuditedOccupancyResolver',return_value=resolver):
                counts,_=m.bind_static(profiles,hardware,'audit')
            self.assertEqual(counts,{'UNKNOWN_EXACT_STATIC_ALLOCATION':1})
            self.assertNotIn('allocated_shared_bytes',profiles[0]['nodes'][0]['kernel'])

    def test_exact_static_member_keeps_name_and_rejects_mismatched_launch_or_hardware(self):
        for mode in ('correct','grid','hardware'):
            profiles,hardware=self.static_fixture();kernel=profiles[0]['nodes'][0]['kernel']
            patch_kernel={k:kernel[k] for k in ('block','grid','registers_per_thread','static_shared_bytes','dynamic_shared_bytes','cluster')}
            patch_kernel.update(allocated_registers_per_block=256,allocated_shared_bytes=1024)
            if mode=='grid':patch_kernel['grid']=[2,1,1]
            resolver=SimpleNamespace(hardware=hardware if mode!='hardware' else {'hardware_id':'different'},
                source_receipts=[],index={('exact',1,2,7,1,55):patch_kernel})
            with patch.object(m,'AuditedOccupancyResolver',return_value=resolver):
                if mode=='correct':
                    counts,_=m.bind_static(profiles,hardware,'audit')
                    self.assertEqual(counts,{'KNOWN_EXACT_STATIC_ALLOCATION':1});self.assertEqual(kernel['name'],'exact-name')
                else:
                    with self.assertRaises(ValueError):m.bind_static(profiles,hardware,'audit')

    def test_singleton_baselines_never_qualify_even_when_path_rank_is_six(self):
        prepared=dict(rejected_rows=[],features=list(range(6)),rows=[dict(group=('node',i),
            ordered_matrix=np.asarray([np.zeros(6),np.eye(6)[i]])) for i in range(6)])
        result=m.structural_diagnostics(prepared)
        self.assertEqual(result['within_node_path_covariate_rank'],6)
        self.assertEqual(result['nuisance_adjusted_derivative_rank'],0)
        self.assertEqual(result['reason'],'ONE_OBSERVATION_PER_EXACT_NODE_BASELINE_NOT_IDENTIFIABLE')
        self.assertFalse(result['numerical_qualification']);self.assertFalse(result['fit_started'])

    def test_rejected_cohort_does_not_enter_even_structural_rank(self):
        with patch.object(m.np.linalg,'matrix_rank',side_effect=AssertionError('no incomplete rank')):
            result=m.structural_diagnostics({'rejected_rows':[{}]})
        self.assertEqual(result['reason'],'INCOMPLETE_SUPPORTED_NODE_COHORT')

    def test_nontraining_review_role_rejected_before_native_loader(self):
        for change in ({'training_only':False},{'validation_started':True},{'model_fitted':True}):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path=root/'review.json';path.write_text(json.dumps(dict(
                    schema=m.REVIEW_SCHEMA,status='PASS',issues=[],training_only=True,
                    validation_started=False,model_fitted=False,**{} )|change))
                with patch.object(m,'load_review',side_effect=AssertionError('native labels read')):
                    with self.assertRaisesRegex(ValueError,'training-only'):m.intake(path,'unused',root/'output')


if __name__=='__main__':unittest.main()
