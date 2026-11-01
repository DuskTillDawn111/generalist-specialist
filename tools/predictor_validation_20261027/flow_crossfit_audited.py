"""Crossfit report with exact omitted-source identities and full fold coverage."""
import argparse
import inspect
import json
from pathlib import Path

from gsdeploy.forecasting import FullHorizonModel
from . import flow_crossfit_report as core
from .flow_fit import verify_training_model
from .native_flow import checked, receipt, save
from .probe_validation import read


def audit_manifest(path):
    manifest = read(path)
    protocol = read(checked(manifest['protocol']))
    jobs = {j['flow_id']: j for j in protocol['flow_jobs']}
    seed_cv = manifest.get('schema') == 'gsdeploy.continuous_seed_crossfit_manifest.v1'
    if not jobs or len(jobs) != protocol['expected_flow_count']:
        raise ValueError('Complete protocol flow units required')
    folds = manifest['folds']
    if not folds:
        raise ValueError('Empty crossfit manifest')
    seen, seed_units, contracts, refs = set(), set(), {}, [receipt(path), manifest['protocol']]
    for fold in folds:
        model = read(checked(fold['model']))
        verification = verify_training_model(fold['model']['path'])
        if model['training_evidence_tree']['protocol'] != manifest['protocol']:
            raise ValueError('Crossfit model binds another protocol')
        heldouts = fold['heldout_flows'] if seed_cv else [fold]
        if seed_cv:
            seed = fold['heldout_seed']
            if seed in seed_units:
                raise ValueError('Duplicate seed fold')
            seed_units.add(seed)
            expected = {fid for fid, job in jobs.items() if job['seed'] == seed}
            if {h['flow_id'] for h in heldouts} != expected or len(heldouts) != len(expected):
                raise ValueError('Incomplete heldout seed routes')
            if set(verification['omitted_flow_ids']) != expected:
                raise ValueError('Model does not exclude the whole seed')
        for heldout in heldouts:
            fid = heldout.get('flow_id', heldout.get('fold_id'))
            if fid not in jobs or fid in seen:
                raise ValueError('Unknown or duplicate heldout flow')
            seen.add(fid); job = jobs[fid]
            plan = read(checked(heldout['heldout_plan']))
            native_ref = receipt(job['raw_path'])
            if (heldout['heldout_raw'] != native_ref or heldout['heldout_plan'] != job['plan'] or
                    heldout['candidate_id'] != job['candidate_id'] or
                    fid != 'seed-' + str(plan['seed']) + '__' + heldout['candidate_id'] or
                    heldout['heldout_trace_id'] != 'flow:' + native_ref['sha256'] or
                    native_ref in verification['training_raw_receipts']):
                raise ValueError('Heldout identity does not bind the omitted immutable native source')
            if seed_cv and plan['seed'] != fold['heldout_seed']:
                raise ValueError('Heldout raw belongs to another seed')
            raw = read(checked(native_ref))
            if raw['plan'] != job['plan'] or raw['candidate_id'] != job['candidate_id']:
                raise ValueError('Native raw/plan/candidate mismatch')
            contract = {'plan': job['plan'], 'trace': plan['trace'],
                'candidate_ids': sorted(c['id'] for c in plan['candidates']),
                'baseline': plan['baseline_candidate_id'], 'tie_tolerance': plan.get('utility_tie_tolerance', 0.)}
            previous = contracts.setdefault(plan['seed'], contract)
            if previous != contract:
                raise ValueError('Routes within one seed do not share one frozen comparison contract')
            refs.extend([fold['model'], native_ref, job['plan']])
    if seen != set(jobs):
        raise ValueError('Crossfit manifest omits declared flow units')
    if seed_cv and seed_units != {j['seed'] for j in jobs.values()}:
        raise ValueError('Crossfit manifest omits a seed')
    for seed, contract in contracts.items():
        actual = {j['candidate_id'] for j in jobs.values() if j['seed'] == seed}
        if actual != set(contract['candidate_ids']):
            raise ValueError('Incomplete candidate set within a seed')
    return {'contracts': contracts, 'source_receipts': refs, 'seed_cv': seed_cv}


def run(manifest_path, resources_path, quality_path, output):
    audit = audit_manifest(manifest_path)
    paths = set(Path(inspect.getfile(FullHorizonModel)).parents[1].rglob('*.py'))
    paths.update(Path(__file__).parent / name for name in (
        'flow_crossfit_audited.py', 'flow_crossfit_report.py', 'flow_validation.py',
        'flow_metrics.py', 'background_resources.py', 'empirical_resources.py'))
    predictor_sources = [receipt(p) for p in sorted(paths)]
    value = core.run(manifest_path, resources_path, quality_path, output)
    # The frozen core's current V10 baseline/tolerance match this contract.
    # Recompute explicitly so this audit does not depend on that coincidence.
    for seed, contract in audit['contracts'].items():
        rows = [r for r in value['candidates'] if r['seed'] == seed]
        value['by_seed'][str(seed)] = core.ranking(rows, contract['baseline'],
                                                  tie_tolerance=contract['tie_tolerance'])
    for ref in predictor_sources + audit['source_receipts']:
        checked(ref)
    # Keep core output immutable; the audited report binds it and adds evidence.
    value['original_core_report'] = receipt(Path(output) / 'report.json')
    value['predictor_source_receipts'] = predictor_sources
    value['heldout_identity_audit'] = {
        'whole_declared_domain_complete': True, 'omitted_raw_identity_verified': True,
        'baseline_and_ties_from_plan': True, 'audit_sources': audit['source_receipts']}
    save(Path(output) / 'audited-report.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'resources', 'quality', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.resources, args.quality, args.output)
    print(json.dumps({'known': result['known'], 'requested': result['requested'],
                      'independent_validation': False, 'heldout_identity_verified': True}))
