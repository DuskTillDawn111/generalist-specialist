"""Independent CPU checks of nuisance identifiability and wait admission."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import kernel_dag_response_numerical_pipeline_v1 as numerical
from tools.predictor_validation_20261027 import kernel_dag_response_numerical_continuation_v1 as continuation


class IndependentNumericalTests(unittest.TestCase):
    def test_full_within_path_rank_does_not_identify_singleton_baselines(self):
        rows=[]
        for index in range(60):
            rows.append(dict(group=('distinct-baseline',index),
                ordered_matrix=np.vstack([np.zeros(6),np.eye(6)])))
        self.assertEqual(numerical.structural_summary(dict(rows=rows))['path_contrast_rank'],6)
        derivatives=np.random.default_rng(8).normal(size=(60,6))
        result=numerical.likelihood.nuisance_certificate(rows,derivatives)
        self.assertEqual(result['projected_derivative_rank'],0)
        self.assertFalse(result['full_column_rank'])
        self.assertEqual(result['parameters'],66)

    def test_conservative_certificate_never_qualifies_dense_rank_failure(self):
        rng=np.random.default_rng(32)
        groups=np.repeat(np.arange(4),[3,7,11,15])
        rows=[dict(group=int(g)) for g in groups]
        intercepts=np.eye(4)[groups]
        for mode in ('full','duplicate','nuisance-confounded'):
            derivative=rng.normal(size=(len(rows),6))
            if mode=='duplicate':derivative[:,-1]=derivative[:,0]
            if mode=='nuisance-confounded':derivative[:,-1]=intercepts@np.arange(4.)
            certificate=numerical.likelihood.nuisance_certificate(rows,derivative)
            dense=np.column_stack([intercepts,derivative])
            dense_full=np.linalg.matrix_rank(dense)==dense.shape[1]
            self.assertEqual(dense_full,mode=='full')
            if certificate['full_column_rank']:self.assertTrue(dense_full)
            if mode=='full':self.assertTrue(certificate['full_column_rank'])
            else:self.assertFalse(certificate['full_column_rank'])

    def test_foreign_plan_review_is_rejected_before_claim_or_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest_ref=continuation.write(root/'manifest.json',{})
            plan_ref=continuation.write(root/'plan.json',{})
            review_ref=continuation.write(root/'review.json',dict(
                schema=continuation.SCHEMA+'.independent_review',
                status='PASS_CPU_WAIT_SEAL_FIT_INTERFACES_NOT_NUMERICAL_QUALIFICATION',
                manifest=manifest_ref,numerical_plan=dict(plan_ref,sha256='0'*64),
                unresolved_blocking_findings=[],test_summary=dict(status='PASS')))
            manifest=dict(numerical_plan=plan_ref,output=str(root/'run'))
            with patch.object(continuation,'audit',return_value=(manifest,{})), \
                 patch.object(continuation.subprocess,'run',side_effect=AssertionError('must not start')):
                with self.assertRaisesRegex(ValueError,'Exact root independent'):
                    continuation.run(manifest_ref['path'],review_ref['path'])
            self.assertFalse((root/'run').exists())


if __name__=='__main__':unittest.main()
