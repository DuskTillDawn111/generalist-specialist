"""Repair the heldout scorer's id/segment_id input adapter, preserving its frozen model.

The original failed scorer and protocol stay byte-for-byte intact. This wrapper
changes only the mechanical span key, and records its post-capture correction.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import validate_activation_energy as frozen


def normalize_span_keys(spans):
    result=[]
    for span in spans:
        if 'segment_id' in span:
            if 'id' in span and span['id']!=span['segment_id']:
                raise ValueError('Conflicting semantic stage identifiers')
            result.append(dict(span))
        else:
            result.append(dict(span,segment_id=span['id']))
    return result


def main():
    if len(sys.argv)!=4 or sys.argv[1:3]!=['score','--output']:
        raise ValueError('Expected score --output DIRECTORY')
    output=Path(sys.argv[3]).resolve()
    receipt=dict(schema='gsdeploy.conditional_energy_scorer_schema_repair.v1',
                 source=str(Path(__file__).resolve()),
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 original_protocol_sha256=hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest(),
                 reason='stage_layout returns id; activity_windows requires segment_id',
                 correction='copy each observed span and add segment_id equal to its existing id',
                 frozen_model_changed=False,frozen_criteria_changed=False,
                 correction_made_after_capture=True)
    with (output/'score-schema-repair-v1.json').open('x') as out:json.dump(receipt,out,indent=2);out.write('\n')
    original=frozen.activity_windows
    frozen.activity_windows=lambda profiles,spans,start,end:original(profiles,normalize_span_keys(spans),start,end)
    frozen.main()
    path=output/'validation.json'
    report=json.loads(path.read_text());report['scorer_schema_repair']=receipt
    path.rename(output/'validation-before-repair-provenance-v1.json')
    with path.open('x') as out:json.dump(report,out,ensure_ascii=False,indent=2);out.write('\n')


if __name__=='__main__':main()
