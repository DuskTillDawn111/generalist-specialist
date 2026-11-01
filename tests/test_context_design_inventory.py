"""Design rank and input identity gates, without opening any research results."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

PATH=Path(__file__).resolve().parents[1]/'tools/spillover_context_20261022/build_context_design.py'
tree=ast.parse(PATH.read_text())
ns={'np':np,'hashlib':hashlib,'json':json}
for name in ('design_rank','identity_match','digest','identity_with_receipt'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(PATH),'exec'),ns)
AUDIT=PATH.with_name('audit_context_identity.py')
atree=ast.parse(AUDIT.read_text())
for name in ('protocol_only_equivalent','contract_equal'):
    node=next(n for n in atree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(AUDIT),'exec'),ns)


class ContextDesignTests(unittest.TestCase):
    def test_repeated_requests_do_not_resolve_missing_directions(self):
        row=[1.,0.,.3,0.]
        r=ns['design_rank']([row]*100,4)
        self.assertEqual(r['rank'],1);self.assertEqual(r['nullity'],3)
        self.assertIsNone(r['condition_number']);self.assertEqual(r['zero_columns'],[1,3])
        full=ns['design_rank'](np.eye(4).tolist(),4)
        self.assertEqual(full['rank'],4);self.assertEqual(full['condition_number'],1.)

    def test_missing_or_different_identity_never_matches_by_fallback(self):
        b=dict(request_sha256=['r'],protocol_sha256=['p'],effective_generation=[{'sampling':False}])
        response=dict(request_sha256='r',protocol_sha256='p',effective_generation={'sampling':False})
        self.assertTrue(ns['identity_match'](response,b)[0])
        for key in b:
            missing=dict(response);missing.pop(key)
            self.assertFalse(ns['identity_match'](missing,b)[0])
        mismatch=dict(response,request_sha256='shared-worker-other-hash')
        self.assertFalse(ns['identity_match'](mismatch,b)[0])
        self.assertFalse(ns['identity_match'](response,{})[0])

    def test_protocol_equivalence_cannot_hide_native_control_or_media_change(self):
        old=dict(protocol_sha256='old',generation={'tokens':20},media={'sha':'fixed'})
        new=dict(old,protocol_sha256='shared')
        self.assertTrue(ns['protocol_only_equivalent'](old,new))
        self.assertFalse(ns['protocol_only_equivalent'](old,dict(new,generation={'tokens':21})))
        self.assertFalse(ns['protocol_only_equivalent'](old,dict(new,media={'sha':'other'})))

    def test_scoped_receipt_rejects_reuse_or_observation_tampering(self):
        response=dict(request_sha256='new',protocol_sha256='shared',effective_generation={'tokens':20})
        baseline=dict(request_sha256=['old'],protocol_sha256=['oldp'],effective_generation=[{'tokens':20}])
        scope=dict(raw_sha256='raw',request_id='req',family='generalist_2',task='text',n=1,sample_id='a')
        entry=dict(**scope,response_digest=ns['digest'](response),baseline_digest=ns['digest'](baseline),proof_kind='PROTOCOL_ONLY_EXACT_NATIVE_REQUEST_EQUIVALENCE')
        self.assertTrue(ns['identity_with_receipt'](response,baseline,entry,scope)[0])
        self.assertFalse(ns['identity_with_receipt'](dict(response,service_ms=10),baseline,entry,scope)[0])
        self.assertFalse(ns['identity_with_receipt'](response,baseline,entry,dict(scope,sample_id='b'))[0])
        image_scope=dict(scope,family='image');image_entry=dict(entry,**image_scope)
        self.assertFalse(ns['identity_with_receipt'](response,baseline,image_entry,image_scope)[0])

    def test_ocr_contract_change_never_qualifies_by_missing_fields(self):
        original=dict(candidate={'slot':'ocr'},adapter='ocr',execution={'dtype':'float32'},resources={'checkpoint':'fixed','request_output':'old'})
        runtime=dict(original,task_candidates={'ocr':{'slot':'ocr'}},resources={'checkpoint':'fixed','request_output':'new'})
        self.assertTrue(ns['contract_equal'](original,runtime,'ocr'))
        self.assertFalse(ns['contract_equal'](original,dict(runtime,execution={'dtype':'float16'}),'ocr'))
        self.assertFalse(ns['contract_equal'](original,dict(runtime,resources={'checkpoint':'changed'}),'ocr'))


if __name__=='__main__':unittest.main()
