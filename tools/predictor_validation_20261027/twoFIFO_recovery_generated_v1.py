"""Recovery-producer training diagnostics using the unchanged queue engine.

The original stopped run is not accepted here. A saved fit must authenticate
the explicit recovery intake. Pure full-cohort forecast and scoring functions
remain the same; no numerical fitting or GPU execution occurs in this module.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

from . import twoFIFO_recovery_phase12_v1 as response
from . import twoFIFO_source_rebound_generated_v1 as numerical
from .implementation_source_closure_v27 import implementation_sources, strict_references

SCHEMA = 'gsdeploy.twoFIFO_recovery_generated_training.v1'
generated = numerical.generated
require, receipt, checked, save = response.require, response.receipt, response.checked, numerical.save


def prepare(fit_manifest_path, scoring_plan_path, quality_path, output):
    manifest_ref = receipt(fit_manifest_path)
    manifest = response.audit(fit_manifest_path)
    require(not (Path(fit_manifest_path).resolve().parent / 'phase12-fit-claim.json').exists(),
            'Draw and score design must precede the permitted recovery fit')
    score_ref, quality_ref = receipt(scoring_plan_path), receipt(quality_path)
    generated.whole_score(dict(original_window=generated.encode([Fraction(0), Fraction(generated.MEASUREMENT_NS)]),
        original_drain_end_ns=generated.encode(Fraction(generated.MEASUREMENT_NS + generated.DRAIN_NS)),
        cutoff_ns=generated.encode(Fraction(generated.MEASUREMENT_NS + generated.DRAIN_NS)),
        RESULT_receipts={}, unfinished_request_ids=[], offered_requests=0), [], checked(score_ref), checked(quality_ref))
    design = dict(schema=SCHEMA + '.design', status='CPU_PREDECLARED_BEFORE_RECOVERY_FIT',
        fit_manifest=manifest_ref, scoring_plan=score_ref, quality=quality_ref,
        recovery_manifest=manifest['recovery_manifest'],
        scheme=generated.SCHEME, scheme_sha256=generated.digest(generated.SCHEME),
        expected_cases=150, expected_measured_requests=546, expected_priming_history_requests=216,
        heldout_unit='whole_capture_block_all_four_shards', queue_envelope_attached=True,
        queue_bounds_used_for_utility=False, six_term_weights='original_equal_QTLPME',
        numerical_fit_run=False, GPU_started=False, validation_labels_read=False,
        sources=strict_references([manifest_ref, manifest['sources'], score_ref, quality_ref,
                                   implementation_sources(__file__)]))
    for ref in design['sources']:
        response.verify(ref)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return save(output / 'design.json', design)


def audit(path):
    design = checked(receipt(path))
    expected = dict(schema=SCHEMA + '.design', status='CPU_PREDECLARED_BEFORE_RECOVERY_FIT',
        scheme=generated.SCHEME, scheme_sha256=generated.digest(generated.SCHEME),
        expected_cases=150, expected_measured_requests=546, expected_priming_history_requests=216,
        heldout_unit='whole_capture_block_all_four_shards', queue_envelope_attached=True,
        queue_bounds_used_for_utility=False, six_term_weights='original_equal_QTLPME',
        numerical_fit_run=False, GPU_started=False, validation_labels_read=False)
    require(all(design.get(k) == v for k, v in expected.items()),
            'Original complete-cohort draw and score contract with recovery producer required')
    manifest = response.audit(design['fit_manifest']['path'])
    require(design['recovery_manifest'] == manifest['recovery_manifest'], 'Recovery producer changed')
    sources = strict_references([design['fit_manifest'], manifest['sources'], design['scoring_plan'],
                                design['quality'], implementation_sources(__file__)])
    require(design['sources'] == sources, 'Recovery generated implementation closure changed')
    for ref in sources:
        response.verify(ref)
    return design


def run(design_path, fit_report_path, output):
    implementation_start = implementation_sources(__file__)
    design_ref = receipt(design_path)
    design = audit(design_path)
    fit_audit = response.audit_fit(fit_report_path)
    fit = fit_audit['report']
    require(fit['manifest'] == design['fit_manifest'], 'Saved recovery fit belongs to another design')

    def evaluate(bundle, intake_document):
        require(intake_document == fit['intake_evidence'], 'Recovery fit intake did not rederive exactly')
        response.check_intake(intake_document, response.audit(design['fit_manifest']['path']))
        result = numerical.evaluate_bundle(bundle, fit['fit'], checked(design['scoring_plan']),
                                           checked(design['quality']), output)
        result.update(schema=SCHEMA, numerical_interface_schema=numerical.SCHEMA,
            upstream_intake_schema=intake_document['schema'], recovery_manifest=design['recovery_manifest'],
            design=design_ref, response_fit=fit_audit['receipt'],
            sources=strict_references([design_ref, design['sources'], fit_audit['receipt'],
                fit['sources'], intake_document['sources'], implementation_start]))
        return result

    result = response.with_verified_bundle(design['fit_manifest']['path'], fit['continuation']['path'], evaluate)
    audit(design_path)
    require(implementation_sources(__file__) == implementation_start, 'Recovery generated source changed during run')
    for ref in result['sources']:
        response.verify(ref)
    return save(Path(output) / 'report.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--fit-manifest', required=True)
    prep.add_argument('--scoring-plan', required=True)
    prep.add_argument('--quality', required=True)
    prep.add_argument('--output', required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--design', required=True)
    execute.add_argument('--fit-report', required=True)
    execute.add_argument('--output', required=True)
    args = parser.parse_args()
    print(prepare(args.fit_manifest, args.scoring_plan, args.quality, args.output)
          if args.command == 'prepare' else run(args.design, args.fit_report, args.output))
