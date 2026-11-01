"""Fixed variants keep targets/handler fits and only change declared layers."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from tests import test_duration_profiles as fixtures
from tools.predictor_validation_20261027.background_duration_variant import fit_traces
from tools.predictor_validation_20261027.background_duration import fit_traces as baseline_fit
from tools.predictor_validation_20261027.background_statistical import BackgroundStatisticalProvider, family_task
from tools.predictor_validation_20261027.duration_variant_profiles import bounded_ridge


class VariantTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.DurationProviderTests();f.setUp();self.traces=copy.deepcopy(f.traces)
        # Add repeat OCR solo and co-runs, preserving mixed two-part OCR wall time.
        for i in range(4):
            start=0;end=80_000_000 if i<2 else 150_000_000
            ocr={'request_id':'o','alias':'o','family':'ocr','task':'ocr','pool_id':'ocr-input',
                'handler_start_ns':start,'handler_end_ns':end,
                'segments':[{'phase':'ocr_a','start_ns':0,'end_ns':end//3},{'phase':'ocr_b','start_ns':end//3,'end_ns':end}]}
            rows=[ocr]
            if i>=2:
                other=copy.deepcopy(self.traces[0]['requests'][0]);other['handler_end_ns']=end
                other['segments']=[{'phase':'prefill','start_ns':0,'end_ns':end//2},{'phase':'decode','start_ns':end//2,'end_ns':end}]
                rows.append(other)
            self.traces.append({'id':'ocr-'+str(i),'split':'training','execution_context_sha256':'v7-fixture','requests':rows})

    def test_phase_ridge1_only_changes_semantic_layer_and_uses_exact_alpha(self):
        old=baseline_fit(self.traces);new=fit_traces(self.traces,variant='phase_ridge1')
        self.assertEqual(old['base_model']['handler_models'],new['base_model']['handler_models'])
        self.assertEqual(old['base_model']['solo_baselines'],new['base_model']['solo_baselines'])
        self.assertEqual(new['definition']['base_numerics']['semantic_alpha'],1.)
        matrix=np.array([[1.],[2.],[5.]]);p=.7
        speed,_=bounded_ridge(matrix,[p],alpha=1.)
        self.assertAlmostEqual(speed[0],(1+p)/(matrix.mean()+1.))

    def test_ocr_handler_ties_every_seen_phase_speed_without_fitting_ocr_phase(self):
        new=fit_traces(self.traces,variant='ocr_handler');base=new['base_model']
        ocr=next(p for p in base['profiles'] if p['task']=='ocr')
        self.assertEqual([s['id'] for s in ocr['segments']],['ocr_native'])
        self.assertAlmostEqual(ocr['segments'][0]['duration_ms'],80.)
        handler=next(m for m in base['handler_models'] if m['target'][1]=='ocr')
        semantic=next(m for m in base['semantic_models'] if m['target'][1]=='ocr')
        self.assertEqual(semantic['fit_diagnostics']['phase_parameters_fitted'],0)
        for key,speed in semantic['progress'].items():
            self.assertEqual(speed,handler['progress'][base['phase_to_family'][key]])
        baseline=baseline_fit(self.traces)
        self.assertEqual([m for m in baseline['base_model']['handler_models'] if m['target'][1]!='ocr'],
                         [m for m in base['handler_models'] if m['target'][1]!='ocr'])
        self.assertTrue(all([s['id'] for s in p['segments']]==['prefill','decode'] for p in base['profiles'] if p['task']!='ocr'))

    def test_frozen_provider_uses_exact_ocr_handler_and_keeps_unknown_mixtures(self):
        new=fit_traces(self.traces,variant='ocr_handler');provider=BackgroundStatisticalProvider(new,'v7-fixture')
        aliases={'s':{'family':'text','supported_tasks':['text']},'o':{'family':'ocr','supported_tasks':['ocr']},'g':{'family':'generalist_2','supported_tasks':['asr']}}
        def active(alias,task,pool,segment):
            p=provider.lookup[(aliases[alias]['family'],task,pool)]
            return {'alias':alias,'task':task,'pool_id':pool,'profile_id':p['id'],'segment_id':segment}
        o=active('o','ocr','ocr-input','ocr_native');s=active('s','text','input','decode')
        a=provider.service(o,{'o':o,'s':s},aliases,'mean')
        s['segment_id']='unseen_predicted_phase'
        b=provider.service(o,{'o':o,'s':s},aliases,'mean')
        self.assertTrue(a['known']);self.assertTrue(b['known']);self.assertEqual(a['service_ms'],b['service_ms'])
        g=active('g','asr','input','prefill');unknown=provider.service(o,{'o':o,'g':g},aliases,'mean')
        self.assertFalse(unknown['known'])


if __name__=='__main__':unittest.main()
