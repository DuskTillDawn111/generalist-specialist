"""OCR boundary semantics cannot lose time or turn host return into GPU completion."""
import unittest
from unittest.mock import patch
from experiments.activation_completion_bands import completion_band_layout
from gsdeploy.forecast_errors import UnknownEvidence


class CompletionBandTests(unittest.TestCase):
    def spans(self):
        return [dict(id='setup',kind='host',start_ns=0,end_ns=10),
            dict(id='det',kind='gpu',start_ns=10,end_ns=30,native_root_id='native_ocr',native_band_role='detection'),
            dict(id='rec',kind='gpu',start_ns=30,end_ns=100,native_root_id='native_ocr',native_band_role='recognition_and_tail'),
            dict(id='tail',kind='host',start_ns=100,end_ns=110)]

    def row(self,boundary):
        return dict(worker_response=dict(semantic_phase_events=[dict(component='ocr_recognition',host_start_ns=boundary)]))

    def test_async_detection_completion_gap_stays_in_first_band(self):
        spans=self.spans()
        with patch('experiments.activation_completion_bands.original_layout',return_value=spans):
            result=completion_band_layout(self.row(65),'ocr')
        self.assertEqual([(r['start_ns'],r['end_ns']) for r in result],[(0,10),(10,65),(65,100),(100,110)])
        self.assertEqual(spans[1]['end_ns'],30)
        self.assertFalse(result[1]['pure_GPU_time'])
        self.assertFalse(result[1]['extra_synchronization_added'])

    def test_bad_boundary_is_rejected_instead_of_clipped(self):
        for boundary in (20,100,120):
            with patch('experiments.activation_completion_bands.original_layout',return_value=self.spans()):
                with self.assertRaises(UnknownEvidence):completion_band_layout(self.row(boundary),'ocr')

    def test_other_family_layout_unchanged(self):
        spans=self.spans()
        with patch('experiments.activation_completion_bands.original_layout',return_value=spans):
            self.assertIs(completion_band_layout({},'text'),spans)


if __name__=='__main__':unittest.main()
