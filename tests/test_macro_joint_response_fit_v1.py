import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import macro_joint_response_fit_v1 as module


def document(beta=.3):
    requests=[]
    for count in range(3):
        for sign in (-1,1):
            duration=round(10*math.exp(beta*count+sign*.08)*1e6)
            state=dict(cutoff_processing_ns=100,peer_visible_macro_owner_count=count,
                peer_same_visible_macro_phase_owner_count=0)
            macro=dict(owner='owner',pool_id='input',task='text',macro_id='prefill_0000',
                semantic_phase='prefill',kind='language_forward_macro',forward_shape=None,
                left_ns=100,right_ns=100+duration,response_wall_ns=duration,
                response_domain=module.DOMAIN,additional_stream_elapsed_or_sync_wait_may_be_added=False,
                true_SM_service_identified=False,capture_sha256='capture',request_id=str(len(requests)),
                causal_covariate_path=[dict(left_ns=100,right_ns=100+duration,duration_ns=duration,
                    state=state,each_state_uses_only_its_processed_prefix=True)])
            requests.append(dict(macros=[macro],worker_start_to_ready_ns=duration))
    return dict(schema=module.MACRO_SCHEMA,mode='natural',validation_labels_read=False,requests=requests)


class MacroFitTests(unittest.TestCase):
    def test_free_positive_negative_recovery_joint_baseline(self):
        for beta in (.3,-.3):
            rows=module.prepared([document(beta)],module.CANDIDATES['visible_count'])
            result,b,mu,var=module.fit(rows,'visible_count')
            self.assertAlmostEqual(b[0],beta,places=5)
            self.assertAlmostEqual(next(iter(mu.values())),math.log(10),places=5)
            self.assertAlmostEqual(math.sqrt(var),.08,places=5)
            self.assertTrue(result['joint_response_jacobian']['full_column_rank'])

    def test_endpoint_jacobian_and_meanNLL_gradient(self):
        rows=module.prepared([document()],module.CANDIDATES['visible_count'])
        beta=np.array([.17]);nll,_,_,_,_=module.profile(beta,rows)
        value,gradient=module.objective(beta,rows)
        self.assertAlmostEqual(value,nll/len(rows))
        step=1e-6
        numerical=(module.objective(beta+step,rows)[0]-module.objective(beta-step,rows)[0])/(2*step)
        self.assertAlmostEqual(gradient[0],numerical,places=6)
        value,mu,var,_,_=module.profile(np.array([.3]),rows)
        expected=sum(math.log(r['response_ms'])+.5*math.log(2*math.pi*var)+.5 for r in rows)
        self.assertAlmostEqual(value,expected,places=5)

    def test_dynamic_integral_aggregation_retains_endpoint_and_order(self):
        doc=document();m=doc['requests'][0]['macros'][0]
        start=m['left_ns'];end=m['right_ns'];path=m['causal_covariate_path'];piece=path[0]
        piece.update(right_ns=start+4000000,duration_ns=4000000)
        path.append(dict(left_ns=start+4000000,right_ns=end,duration_ns=end-start-4000000,
            state=dict(piece['state'],cutoff_processing_ns=start+4000000,peer_visible_macro_owner_count=1),
            each_state_uses_only_its_processed_prefix=True))
        row=module.prepared([doc],module.FEATURES[:1])[0]
        work=4+(m['response_wall_ns']/1e6-4)/2
        predicted,extended=module.conditional_median(row,np.array([math.log(2)]),work)
        self.assertAlmostEqual(predicted,row['response_ms']);self.assertFalse(extended)
        predicted,extended=module.conditional_median(row,np.array([math.log(2)]),work+2)
        self.assertTrue(extended);self.assertAlmostEqual(predicted,row['response_ms']+4)
        rows=module.prepared([doc],module.FEATURES[:1]);beta=np.array([.17]);step=1e-6
        _,gradient=module.objective(beta,rows)
        numerical=(module.objective(beta+step,rows)[0]-module.objective(beta-step,rows)[0])/(2*step)
        self.assertAlmostEqual(gradient[0],numerical,places=6)

    def test_nuisance_confounding_detected(self):
        doc=document()
        for r in doc['requests']:
            m=r['macros'][0];m['pool_id']=str(m['causal_covariate_path'][0]['state'][module.FEATURES[0]])
        result,*_=module.fit(module.prepared([doc],module.FEATURES[:1]),'visible_count')
        self.assertFalse(result['joint_response_jacobian']['full_column_rank'])

    def test_every_kind_retained_and_unseen_support_unknown(self):
        doc=document()
        for r,kind in zip(doc['requests'],['host_entry_response','host_ready_response','ocr_macro',
                'language_forward_macro','language_forward_macro','language_forward_macro']):
            r['macros'][0]['kind']=kind
        rows=module.prepared([doc],())
        self.assertEqual(len(rows),6);self.assertEqual(len({r['kind'] for r in rows}),4)
        result=module.evaluate(rows,np.empty(0),{},1.)
        self.assertEqual(result['summary']['unknown_rows'],6)
        self.assertTrue(all('conditional_median_ms' not in r for r in result['records']))

    def test_wrong_domain_noncausal_gap_and_incomplete_coverage_rejected(self):
        for field,value in [('response_domain','natural_host_call_wall_ns'),('left_ns',99)]:
            doc=document();doc['requests'][0]['macros'][0][field]=value
            with self.assertRaises(ValueError):module.prepared([doc],())
        doc=document();doc['requests'][0]['macros'][0]['causal_covariate_path'][0]['state']['cutoff_processing_ns']=101
        with self.assertRaises(ValueError):module.prepared([doc],())
        doc=document();doc['mode']='kernel'
        with self.assertRaises(ValueError):module.prepared([doc],())

    def test_exact_rederive_rejects_rehashed_label_and_changed_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            def save(name,value):
                path=base/name;path.write_text(json.dumps(value));return module.receipt(path)
            source=base/'source.py';source.write_text('pass\n');source_ref=module.receipt(source)
            raw_review_ref=save('raw-review.json',dict(sources=[source_ref]))
            review=dict(training_feature_extraction_eligible=True,validation_labels_read=False,
                raw_review=raw_review_ref,inputs={'raw.json':dict(path=str(base/'raw.json'),sha256='x')},
                execution=dict(path=str(base/'execution.json'),sha256='x'),sources=[source_ref])
            # These referenced files are checked before invoking mock review.
            (base/'raw.json').write_text('{}');review['inputs']['raw.json']=module.receipt(base/'raw.json')
            (base/'execution.json').write_text('{}');review['execution']=module.receipt(base/'execution.json')
            review_ref=save('review.json',review)
            doc=document();doc.update(status='DISJOINT_TRAINING_RESPONSE_MACROS_NOT_FITTED_PROVIDER',sources=[review_ref])
            macro_ref=save('macro.json',doc)
            with patch.object(module,'recheck_raw',return_value=raw_review_ref),\
                    patch.object(module,'recheck_review',return_value=review_ref),\
                    patch.object(module,'extract_macros',return_value=copy.deepcopy(doc)):
                module.verified_document(macro_ref['path'])
                altered=copy.deepcopy(doc);altered['requests'][0]['worker_start_to_ready_ns']+=1
                save('macro.json',altered)
                with self.assertRaisesRegex(ValueError,'rederive exactly'):module.verified_document(macro_ref['path'])
                save('macro.json',doc);source.write_text('changed\n')
                with self.assertRaisesRegex(ValueError,'closure changed'):module.verified_document(macro_ref['path'])

    def test_leaveout_is_complete_capture_and_never_in_training(self):
        docs={}
        for index in range(3):
            doc=document(.2)
            for request in doc['requests']:
                request['macros'][0]['capture_sha256']=str(index)
            docs[str(index)]=doc
        with patch.object(module,'verified_document',side_effect=lambda path:(docs[path],[])):
            result=module.run(list(docs))
        self.assertFalse(result['validation_labels_read']);self.assertFalse(result['joint_provider_complete'])
        for candidate in result['candidates']:
            self.assertEqual(candidate['pooled_leaveout']['rows'],18)
            for fold in candidate['whole_capture_leaveout']:
                self.assertNotIn(fold['heldout_capture_sha256'],fold['training_fit']['captures'])
                self.assertEqual(fold['training_fit']['macro_rows'],12)
                self.assertEqual(fold['evaluation']['summary']['rows'],6)


if __name__=='__main__':unittest.main()
