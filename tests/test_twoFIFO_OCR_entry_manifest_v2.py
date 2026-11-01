import unittest

from tools.predictor_validation_20261027.twoFIFO_OCR_entry_manifest_v2 import build_cases


class OCRGridTests(unittest.TestCase):
    def test_every_capture_and_exact_input_repeats_all_offset_conditions(self):
        base=dict(inputs={'text':{'long':{'pool_id':'language-long'}},
                          'ocr':{'short':{'pool_id':'OCR-short'},'long':{'pool_id':'OCR-long'}}},
                  capture_blocks=[dict(capture_id='capture'+str(i)) for i in range(3)])
        blocks=build_cases(base)
        for block in blocks:
            self.assertEqual(len(block['cases']),34)
            self.assertEqual(sum(case['offered'] for case in block['cases']),82)
            for pool in ('OCR-short','OCR-long'):
                cases=[case for case in block['cases'] if case['OCR_pool_id']==pool]
                self.assertEqual(len(cases),17)
                self.assertEqual({c['planned_offset_ms'] for c in cases},{0,1,20,50,100})
                self.assertTrue(any(not c['peer_owners'] for c in cases))
                self.assertEqual({len(c['peer_owners']) for c in cases},{0,1,2})
                self.assertEqual({c['planned_direction'] for c in cases},{'own','peer_first','OCR_first'})
                self.assertTrue(all(c['expected_phase_is_label'] is False for c in cases))


if __name__=='__main__':unittest.main()
