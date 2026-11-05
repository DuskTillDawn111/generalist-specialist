"""Explicit OCR consumption-boundary candidate, using the shared FIFO engine.

The original phase-category provider remains frozen. This adapter requires the
new boundary policy and rejects accidental reuse of the old OCR template.
Neither adapter grants independent accuracy or physical-model qualification.
"""
import json

from experiments.activation_completion_bands import SCHEMA as BOUNDARY_SCHEMA
from experiments.activation_semantic_bands import BandContextProvider


class CompletionBandContextProvider(BandContextProvider):
    def __init__(self, document, context, *, allow_cpu_fixture=False):
        spec = document.get('derived_host_call_band_model', {})
        if spec.get('boundary_policy') != BOUNDARY_SCHEMA:
            raise ValueError('Explicit completion-boundary policy required')
        expected = ('detection_through_first_crop', 'recognition_from_first_forward')
        for profile in document['profiles']:
            if profile['family'] != 'ocr':
                continue
            spans = [s for s in profile['segments'] if s['kind'] == 'gpu']
            if (tuple(s.get('native_band_role') for s in spans) != expected
                    or any(s.get('boundary_schema') != BOUNDARY_SCHEMA
                           or s.get('pure_GPU_time') is not False
                           or s.get('extra_synchronization_added') is not False
                           or s['id'] != s['native_root_id'] + '.completion_band.' + s['native_band_role']
                           for s in spans)
                    or spans[0]['native_root_id'] != spans[1]['native_root_id']):
                raise ValueError('OCR template differs from consumed-detection boundary policy')
        super().__init__(document, context, allow_cpu_fixture=allow_cpu_fixture)

    def _context(self, active, aliases, scenario):
        values, evidence = super()._context(active, aliases, scenario)
        details = json.loads(evidence)
        details['boundary_policy'] = BOUNDARY_SCHEMA
        return values, json.dumps(details, sort_keys=True)
