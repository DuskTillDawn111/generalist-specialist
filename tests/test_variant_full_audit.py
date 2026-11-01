import copy
import unittest

from tools.predictor_validation_20261027 import variant_full_audit as module
from tools.predictor_validation_20261027.background_duration_variant import fit_traces
from tools.predictor_validation_20261027.profiling_analysis import digest


class StrictFullVariantAuditTests(unittest.TestCase):
    def model(self, variant):
        # Genuine fitted numerical models, not a fixture merely changing labels.
        requests=[]
        for index,duration in enumerate((20.,24.,35.,39.)):
            start=index*100_000_000
            requests.append({'request_id':'r'+str(index),'alias':'ocr#0','family':'ocr',
                'task':'ocr','pool_id':'input','handler_start_ns':start,
                'handler_end_ns':start+round(duration*1e6),
                'segments':[{'phase':'ocr_native','start_ns':start,'end_ns':start+round(duration*1e6)}]})
        traces=[{'id':'training','split':'training','execution_context_sha256':'ctx','requests':requests}]
        value=fit_traces(traces,variant=variant)
        value['variant']=variant
        value['training_policy']=copy.deepcopy(module.original.FULL_POLICY)
        value.pop('model_sha256')
        value['model_sha256']=digest(value)
        return value

    def protocol(self):
        return {'policy':copy.deepcopy(module.original.FULL_POLICY),
                'selection_verification':{'selected_variant':'ocr_handler'}}

    def test_actual_fitted_ocr_handler_matches_selected_full_policy(self):
        self.assertTrue(module.verify_identity(self.model('ocr_handler'),self.protocol(),
                        {'selected_variant':'ocr_handler'}))

    def test_actual_ridge1_numerics_with_ocr_policy_are_rejected(self):
        ridge=self.model('phase_ridge1')
        # The legacy numerical-definition check accepts these valid ridge1
        # coefficients despite copied OCR training_policy, so the wrapper must
        # bind them to the selected identity explicitly.
        module.original._verify_duration_definition(ridge)
        with self.assertRaisesRegex(ValueError,'actual full/base numerical identity'):
            module.verify_identity(ridge,self.protocol(),{'selected_variant':'ocr_handler'})

    def test_relabeling_ridge1_as_ocr_without_changing_numerics_is_rejected(self):
        ridge=self.model('phase_ridge1');ridge['variant']='ocr_handler'
        with self.assertRaisesRegex(ValueError,'numerical identity'):
            module.verify_identity(ridge,self.protocol(),{'selected_variant':'ocr_handler'})

    def test_fitted_ocr_phase_speed_tamper_is_rejected(self):
        value=self.model('ocr_handler')
        phase=value['base_model']['semantic_models'][0]
        key=next(iter(phase['progress']))
        phase['progress'][key]*=1.2
        with self.assertRaisesRegex(ValueError,'exactly tied'):
            module.verify_identity(value,self.protocol(),{'selected_variant':'ocr_handler'})

    def test_full_protocol_winner_cannot_differ_from_selection(self):
        protocol=self.protocol();protocol['selection_verification']['selected_variant']='phase_ridge1'
        with self.assertRaisesRegex(ValueError,'winner'):
            module.verify_identity(self.model('ocr_handler'),protocol,{'selected_variant':'ocr_handler'})


if __name__=='__main__':
    unittest.main()
