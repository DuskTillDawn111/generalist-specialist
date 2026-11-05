"""Candidate OCR boundary after detection consumption, without extra sync.

The retained pipeline consumes detector results on CPU, creates ordered crops,
and submits the first recognition input before entering rec.forward. Its
observed host entry is a causal boundary after detector-result consumption,
unlike det.forward's asynchronous host return. Both bands still contain mixed
host/transfers/CUDA work. These are derived training observations only.
"""
from experiments.activation_semantic_bands import band_layout as original_layout
from gsdeploy.forecast_errors import UnknownEvidence

SCHEMA='gsdeploy.ocr_consumed_detection_bands.v1'


def completion_band_layout(row, family):
    original=original_layout(row,family)
    if family!='ocr':return original
    indices=[i for i,s in enumerate(original) if s['kind']=='gpu']
    if len(indices)!=2 or indices[1]!=indices[0]+1:
        raise UnknownEvidence('EXPECTED_TWO_CONTIGUOUS_OCR_DERIVED_BANDS')
    first,second=(original[i] for i in indices)
    if (first.get('native_band_role')!='detection' or second.get('native_band_role')!='recognition_and_tail'
            or first['native_root_id']!=second['native_root_id']):
        raise UnknownEvidence('EXPECTED_PINNED_OCR_BAND_STRUCTURE')
    events=row['worker_response']['semantic_phase_events']
    recognition=[e for e in events if e['component']=='ocr_recognition']
    if not recognition:raise UnknownEvidence('MISSING_OBSERVED_RECOGNITION_ENTRY')
    boundary=min(e['host_start_ns'] for e in recognition)
    if not first['end_ns']<=boundary<second['end_ns']:
        raise UnknownEvidence('INVALID_DETECTION_CONSUMPTION_BOUNDARY')
    root=first['native_root_id']
    result=[dict(s) for s in original]
    for index,old,left,right,role in [
            (indices[0],first,first['start_ns'],boundary,'detection_through_first_crop'),
            (indices[1],second,boundary,second['end_ns'],'recognition_from_first_forward')]:
        result[index]=dict(old,id=root+'.completion_band.'+role,start_ns=left,end_ns=right,
            native_band_role=role,boundary_schema=SCHEMA,
            boundary_evidence='first ocr_recognition host entry after retained CPU detector-result consumption',
            pure_GPU_time=False,extra_synchronization_added=False)
    return result
