"""Strict selected-variant identity around the frozen full-fit source verifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from . import variant_full_fit as original
from .background_duration_variant import background_definition
from .duration_variant_profiles import variant_definition
from .native_flow import checked, receipt, save
from .probe_validation import read
from .profiling_analysis import digest


def verify_identity(document, full_protocol, selection):
    """Winner label, numerical definitions and fitted OCR ties must agree."""
    selected = selection.get('selected_variant')
    expected_base = variant_definition('ocr_handler')
    expected = background_definition('ocr_handler')
    base = document['base_model']
    if (selected != 'ocr_handler' or document.get('variant') != selected or
            full_protocol.get('policy') != original.FULL_POLICY or
            full_protocol['policy']['numerical_engine'] != expected_base or
            document.get('training_policy') != full_protocol['policy'] or
            document.get('definition') != expected or document.get('definition_sha256') != digest(expected) or
            base.get('definition') != expected_base or base.get('definition_sha256') != digest(expected_base)):
        raise ValueError('Selected OCR variant and actual full/base numerical identity must agree')
    # This verifies actual OCR model structure, exact full-context speed ties,
    # zero fitted OCR phase parameters, not only a variant metadata label.
    original._verify_duration_definition(document)
    if full_protocol.get('selection_verification', {}).get('selected_variant') != selected:
        raise ValueError('Full protocol selection verification disagrees with winner')
    return True


def verify_variant_full_model(path):
    sources = [receipt(Path(__file__).with_name(name)) for name in (
        'variant_full_audit.py', 'variant_full_fit.py', 'duration_variant_fit.py',
        'background_duration_variant.py', 'duration_variant_profiles.py', 'variant_selection.py',
        'duration_selection.py')]
    model_ref = receipt(path)
    document = read(checked(model_ref))
    full_ref = document['training_evidence_tree']['full_fit_protocol']
    full_protocol = read(checked(full_ref))
    selection_ref = full_protocol['selection']
    selection = read(checked(selection_ref))
    verify_identity(document, full_protocol, selection)
    verification = original.verify_variant_full_model(path)
    if verification.get('variant') != 'ocr_handler':
        raise ValueError('Recursive verifier returned another numerical variant')
    for ref in [model_ref, full_ref, selection_ref, *sources]:
        checked(ref)
    return {**verification, 'strict_variant_full_identity_verified':True,
            'selected_variant':'ocr_handler', 'actual_numerical_variant':'ocr_handler',
            'full_fit_protocol':full_ref, 'training_selection':selection_ref,
            'strict_audit_source_receipts':sources,
            'independent_validation':False, 'final_gain_eligible':False}


def audit(path, output):
    if Path(output).exists():
        raise FileExistsError(output)
    verification = verify_variant_full_model(path)
    value = {'schema':'gsdeploy.strict_selected_variant_full_model_audit.v1',
             'created_ns':time.perf_counter_ns(), 'verification':verification,
             'source_receipts':[receipt(path), *verification['strict_audit_source_receipts']],
             'read_only':True, 'fitting_performed':False, 'GPU_started':False,
             'independent_results_read':False, 'independent_validation':False,
             'original_failures_preserved':True, 'final_gain_eligible':False}
    save(output,value)
    return value


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True); parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = audit(args.model,args.output)
    print(json.dumps({'verified':result['verification']['verified'],
                      'strict_variant_full_identity_verified':True, 'GPU_started':False}))
