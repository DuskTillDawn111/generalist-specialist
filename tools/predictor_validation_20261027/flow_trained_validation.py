"""Independent-flow entry with transitive continuous-training source audits.

The frozen flow reporter and thresholds stay unchanged. This entry adds the
strict raw -> prospectively declared training plan -> mapped trace checks that
the older sealer cannot perform for continuous-flow evidence wrappers.
"""
import argparse
import json
from pathlib import Path

from . import flow_validation, flow_gate
from .native_flow import checked, receipt, save
from .probe_validation import read


def audit_model(path):
    document = read(path)
    if document.get('definition', {}).get('estimator') == 'duration_weighted_relative_work_WLS.v1':
        from .duration_full_fit import verify_duration_full_model
        result = verify_duration_full_model(path)
    else:
        from .flow_fit import verify_training_model
        result = verify_training_model(path)
    if (result.get('verified') is not True or result.get('training_flow_count') != 8 or
            result.get('omitted_flow_ids') or result.get('omitted_seed') is not None):
        raise ValueError('Full eight-flow training model required for independent validation')
    return result


def seal(plan_path, model_path, resources_path, quality_path, output):
    from . import flow_fit
    if read(model_path).get('definition', {}).get('estimator') == 'duration_weighted_relative_work_WLS.v1':
        from . import duration_full_fit
        verifier_source = receipt(duration_full_fit.__file__)
    else:
        verifier_source = receipt(flow_fit.__file__)
    verification = audit_model(model_path)
    if read(plan_path).get('phase') != 'predictor_validation':
        raise ValueError('Prospective independent validation plan required')
    value = flow_validation.seal(plan_path, model_path, resources_path, quality_path, output)
    audit_model(model_path)
    save(Path(output) / 'training-audit.json', {
        'schema': 'gsdeploy.independent_flow_transitive_training_audit.v1',
        'predictions': receipt(Path(output) / 'predictions.json'), 'model': receipt(model_path),
        'verification': verification, 'verifier_source': verifier_source,
        'entry_source': receipt(__file__), 'validation_observations_used_for_fitting': False})
    return value


def check_sealed(sealed_dir):
    folder = Path(sealed_dir)
    audit = read(folder / 'training-audit.json')
    for key in ('predictions', 'model', 'verifier_source', 'entry_source'):
        checked(audit[key])
    if audit['predictions'] != receipt(folder / 'predictions.json'):
        raise ValueError('Training audit binds a different sealed forecast')
    sealed = read(folder / 'predictions.json')
    if sealed['sources']['model'] != audit['model']:
        raise ValueError('Training audit and sealed model differ')
    audit_model(audit['model']['path'])
    return audit


def report(sealed_dir, raw_paths, output):
    check_sealed(sealed_dir)
    value = flow_validation.report(sealed_dir, raw_paths, output)
    check_sealed(sealed_dir)
    save(Path(output) / 'training-audit.json', {
        'schema': 'gsdeploy.independent_flow_report_training_audit.v1',
        'sealed_audit': receipt(Path(sealed_dir) / 'training-audit.json'),
        'report': receipt(Path(output) / 'report.json'), 'entry_source': receipt(__file__)})
    return value


def gate(plan_path, report_path, output):
    audit = read(Path(report_path).parent / 'training-audit.json')
    for key in ('sealed_audit', 'report', 'entry_source'):
        checked(audit[key])
    if audit['report'] != receipt(report_path):
        raise ValueError('Gate transitive audit binds a different report')
    check_sealed(Path(audit['sealed_audit']['path']).parent)
    return flow_gate.run(plan_path, report_path, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    a = sub.add_parser('seal')
    for field in ('plan', 'model', 'resources', 'quality', 'output'):
        a.add_argument('--' + field, required=True)
    a = sub.add_parser('report')
    for field in ('sealed-dir', 'output'):
        a.add_argument('--' + field, required=True)
    a.add_argument('--raw', action='append', required=True)
    a = sub.add_parser('gate')
    for field in ('plan', 'report', 'output'):
        a.add_argument('--' + field, required=True)
    args = parser.parse_args()
    if args.operation == 'seal':
        value = seal(args.plan, args.model, args.resources, args.quality, args.output)
    elif args.operation == 'report':
        value = report(args.sealed_dir, args.raw, args.output)
    else:
        value = gate(args.plan, args.report, args.output)
    print(json.dumps({'status': value.get('status', 'SEALED'), 'output': args.output, 'GPU_started': False}))
