"""Fresh whole-seed duration-model rollouts on training CV, never validation."""
from __future__ import annotations

import argparse
import collections
import inspect
import json
import math
from pathlib import Path
import time

from gsdeploy.forecasting import FullHorizonModel
from . import flow_crossfit_report as core
from .native_flow import checked, receipt, save
from .probe_validation import read


ROUTES = {'shared_balanced', 'shared_skewed', 'shared_spill_low', 'shared_spill_high'}
MANIFEST_SCHEMA = 'gsdeploy.continuous_seed_crossfit_manifest.v1'
ESTIMATOR = 'duration_weighted_relative_work_WLS.v1'


def verify_duration_model(path):
    # Keep model verification in the independent duration training implementation.
    from .duration_fit import verify_duration_model as verify
    return verify(path)


def audit_manifest(path):
    manifest = read(path)
    if (manifest.get('schema') != MANIFEST_SCHEMA or
            manifest.get('estimator') != ESTIMATOR or
            manifest.get('retrospective_training_cross_validation') is not True or
            manifest.get('no_independent_validation') is not True):
        raise ValueError('Explicit duration whole-seed training CV manifest required')
    verifier = manifest.get('training_verifier', {})
    if verifier.get('name') != 'verify_duration_model':
        raise ValueError('Explicit duration model verifier required')
    checked(verifier['source'])
    protocol = read(checked(manifest['protocol']))
    jobs_list = protocol['flow_jobs']
    jobs = {j['flow_id']: j for j in jobs_list}
    seeds = {j['seed'] for j in jobs_list}
    if (len(jobs_list) != 8 or len(jobs) != 8 or protocol['expected_flow_count'] != 8 or len(seeds) != 2):
        raise ValueError('Complete two-seed/eight-flow protocol required')
    for seed in seeds:
        ids = [j['candidate_id'] for j in jobs_list if j['seed'] == seed]
        if len(ids) != 4 or set(ids) != ROUTES:
            raise ValueError('Complete four-route protocol required within each seed')
    folds = manifest['folds']
    if len(folds) != 2:
        raise ValueError('Exactly two whole-seed folds required')
    refs = [receipt(path), manifest['protocol'], verifier['source'], *manifest.get('source_receipts', [])]
    cases, contracts, seen, seen_seeds, verifications = [], {}, set(), set(), {}
    for fold in folds:
        seed = fold['heldout_seed']
        if (fold.get('unit') != 'seed' or seed not in seeds or seed in seen_seeds or
                fold['fold_id'] != 'seed-' + str(seed)):
            raise ValueError('Unknown, duplicate, or incorrectly identified whole-seed fold')
        seen_seeds.add(seed)
        model = read(checked(fold['model']))
        verification = verify_duration_model(fold['model']['path'])
        verifications[str(seed)] = verification
        omitted = {fid for fid, job in jobs.items() if job['seed'] == seed}
        training = set(jobs) - omitted
        declared_training = fold['training_flow_ids']
        if (model.get('training_evidence_tree', {}).get('protocol') != manifest['protocol'] or
                verification.get('verified') is not True or verification.get('omitted_seed') != seed or
                verification.get('duration_model_verified') is not True or
                set(verification.get('omitted_flow_ids', [])) != omitted or
                set(verification.get('training_flow_ids', [])) != training or
                verification.get('training_flow_count') != len(training) or
                len(declared_training) != len(training) or set(declared_training) != training):
            raise ValueError('Model does not exclude exactly the complete heldout seed')
        heldouts = fold['heldout_flows']
        if len(heldouts) != 4 or {h['flow_id'] for h in heldouts} != omitted:
            raise ValueError('Incomplete or duplicate heldout seed routes')
        training_raws = verification.get('training_raw_receipts', [])
        training_paths = {r['path'] for r in training_raws}
        training_hashes = {r['sha256'] for r in training_raws}
        for heldout in heldouts:
            fid = heldout['flow_id']
            if fid in seen:
                raise ValueError('Duplicate heldout flow')
            seen.add(fid)
            job = jobs[fid]
            plan = read(checked(heldout['heldout_plan']))
            native_ref = receipt(job['raw_path'])
            if (heldout['heldout_raw'] != native_ref or heldout['heldout_plan'] != job['plan'] or
                    heldout['candidate_id'] != job['candidate_id'] or plan['seed'] != seed or
                    job['seed'] != seed or fid != 'seed-' + str(seed) + '__' + job['candidate_id'] or
                    heldout['heldout_trace_id'] != 'flow:' + native_ref['sha256'] or
                    native_ref['path'] in training_paths or native_ref['sha256'] in training_hashes or
                    heldout['heldout_trace_id'] in verification.get('training_trace_ids', [])):
                raise ValueError('Heldout identity does not bind the omitted immutable native source')
            if plan.get('phase') != 'predictor_training' or plan.get('split') != 'training':
                raise ValueError('Heldout plan must remain training; independent validation forbidden')
            raw = read(checked(native_ref))
            if raw['plan'] != job['plan'] or raw['candidate_id'] != job['candidate_id']:
                raise ValueError('Native raw/plan/candidate mismatch')
            candidate_ids = [c['id'] for c in plan['candidates']]
            tolerance = plan.get('utility_tie_tolerance', 0.)
            if (len(candidate_ids) != 4 or set(candidate_ids) != ROUTES or
                    plan['baseline_candidate_id'] not in ROUTES or
                    not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0):
                raise ValueError('Complete plan candidate set and valid baseline/tie tolerance required')
            contract = {'plan': job['plan'], 'trace': plan['trace'],
                        'candidate_ids': sorted(candidate_ids), 'baseline': plan['baseline_candidate_id'],
                        'tie_tolerance': tolerance}
            if contracts.setdefault(seed, contract) != contract:
                raise ValueError('Routes within a seed have inconsistent comparison contracts')
            candidates = [c for c in plan['candidates'] if c['id'] == job['candidate_id']]
            if len(candidates) != 1 or candidates[0]['execution_context_sha256'] != verification.get('execution_context_sha256'):
                raise ValueError('Candidate/model execution identity mismatch')
            cases.append({'seed': seed, 'flow_id': fid, 'fold_id': fold['fold_id'] + '__' + job['candidate_id'],
                          'model': fold['model'], 'heldout_raw': native_ref, 'heldout_plan': job['plan'],
                          'candidate_id': job['candidate_id'], 'heldout_trace_id': heldout['heldout_trace_id']})
            refs.extend([fold['model'], native_ref, job['plan']])
    if seen != set(jobs) or seen_seeds != seeds:
        raise ValueError('Whole declared domain omitted from crossfit manifest')
    for ref in refs:
        checked(ref)
    return {'cases': cases, 'contracts': contracts, 'source_receipts': refs,
            'model_verifications': verifications}


def predictor_sources():
    paths = set(Path(inspect.getfile(FullHorizonModel)).parents[1].rglob('*.py'))
    names = ('duration_crossfit_report.py', 'duration_fit.py', 'background_duration.py',
             'duration_profiles.py', 'flow_crossfit_report.py', 'flow_validation.py', 'flow_metrics.py',
             'background_resources.py', 'empirical_resources.py', 'background_statistical.py',
             'statistical_profiles.py', 'profiling_analysis.py', 'native_flow.py', 'probe_validation.py',
             'validation.py', 'flow_fit.py', 'flow_profiles.py', 'background_profiles.py',
             'semantic_profiles.py', 'profiling.py', 'background_collect.py')
    paths.update(Path(__file__).parent / name for name in names)
    return [receipt(p) for p in sorted(paths)]


def verifier_source():
    return receipt(Path(__file__).with_name('duration_fit.py'))


def run(manifest_path, resources_path, quality_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    sources = predictor_sources()
    if read(manifest_path).get('training_verifier', {}).get('source') != verifier_source():
        raise ValueError('Manifest verifier does not bind actual duration verification implementation')
    audit = audit_manifest(manifest_path)
    manifest = read(checked(receipt(manifest_path)))
    duration_protocol_ref = manifest['duration_protocol']
    duration_protocol = read(checked(duration_protocol_ref))
    resources, quality_document = read(resources_path), read(quality_path)
    quality = quality_document.get('loss_by_family_task', quality_document)
    refs = [*audit['source_receipts'], *sources, receipt(resources_path), receipt(quality_path),
            duration_protocol_ref, duration_protocol['selection_protocol'], duration_protocol['threshold_addendum']]
    refs.extend(quality_document.get('source_receipts', []))
    refs.extend(resources.get('source_receipts', resources.get('provenance', {}).get('sources', [])))
    rows, by_seed = [], collections.defaultdict(list)
    for case in audit['cases']:
        plan = read(checked(case['heldout_plan']))
        candidate = next(c for c in plan['candidates'] if c['id'] == case['candidate_id'])
        model = read(checked(case['model']))
        raw, scored_ref = core.scored_heldout(case['heldout_raw'])
        refs.append(scored_ref)
        prediction = core.predict(plan, candidate, model, resources, quality)
        row = core.compare(plan, candidate, prediction, raw)
        row.update(**case, heldout_scored=scored_ref, independent_validation=False)
        rows.append(row)
        by_seed[case['seed']].append(row)
    rankings = {str(seed): core.ranking(items, audit['contracts'][seed]['baseline'],
                                      tie_tolerance=audit['contracts'][seed]['tie_tolerance'])
                for seed, items in sorted(by_seed.items())}
    for ref in refs:
        checked(ref)
    value = {'schema': 'gsdeploy.duration_training_seed_crossfit_error_report.v1',
             'created_ns': time.perf_counter_ns(),
             'scope': 'Retrospective whole-seed training development CV; not independent validation',
             'manifest': receipt(manifest_path), 'candidates': rows, 'by_seed': rankings,
             'duration_protocol': duration_protocol_ref,
             'selection_protocol': duration_protocol['selection_protocol'],
             'threshold_addendum': duration_protocol['threshold_addendum'],
             'requested': sum(r['requested'] for r in rows), 'known': sum(r['known'] for r in rows),
             'complete_prediction_coverage': all(r['known'] == r['requested'] for r in rows),
             'source_receipts': refs, 'predictor_source_receipts': sources,
             'model_verifications': audit['model_verifications'],
             'heldout_identity_audit': {'whole_declared_domain_complete': True,
                'omitted_raw_identity_verified': True, 'whole_seed_excluded': True,
                'baseline_and_ties_from_plan': True, 'contracts': audit['contracts']},
             'retrospective_training_cross_validation': True, 'independent_validation': False,
             'final_gain_eligible': False, 'thresholds_tuned': False,
             'predictive_fitting_performed': False, 'GPU_started': False}
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'report.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'resources', 'quality', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    value = run(args.manifest, args.resources, args.quality, args.output)
    print(json.dumps({k: value[k] for k in ('known', 'requested', 'independent_validation', 'GPU_started')}))
