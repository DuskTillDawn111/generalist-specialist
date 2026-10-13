import json
from pathlib import Path
from module8e3_heldout_validation.common import ROOT, F, read, ref, utc, immutable, inputs, verify_preregistration, verify_manifest as verify_e3
from deployment_search.runtime_cache import digest, file_hash, save_json

E3 = ROOT / 'results/module8e3_heldout_validation'
OUT = ROOT / 'results/module8e4_revision_ablation'
CONTEXT = E3 / 'runtime_context.json'
METHODS = ['M00', 'M10', 'M01', 'M11']
NEW = ['M10', 'M01']


def verify_inputs():
    verify_preregistration(); verify_e3()
    assert read(E3/'final_validation.json')['status'] == 'PASS'
    assert read(E3/'campaign_status.json')['status'] == 'complete'
    checks = []
    for group in ['artifacts']:
        for name, r in read(E3/'report_manifest.json')[group].items():
            actual = file_hash(r['path']); assert actual == r['sha256'], r['path']
            checks.append({'name': name, **r, 'match': True})
    for r in read(E3/'physical_measurements.json')['runs']:
        assert r['valid'] and file_hash(r['raw_path']) == r['raw_sha256']
        assert digest(r['identity']) == r['measurement_identity_hash']
    return checks


def verify_execution():
    verify_e3()
    from module8e2_freeze import verify_sources
    verify_sources()
    manifest = read(OUT/'campaign_manifest.json')
    for group in ['sources','protected_inputs']:
        for r in manifest[group].values(): assert file_hash(r['path']) == r['sha256'], r['path']
    return manifest


def identity(config, workload_hash, role='search'):
    return {'config': config, 'runtime_context_sha256': file_hash(CONTEXT), 'workload_sha256': workload_hash,
        'phase_contract': role, 'protocol_sha256': file_hash(F/'method_comparison_freeze.json'),
        'sharing_policy_sha256': file_hash(F/'measurement_sharing_policy.json')}


def old_donors():
    rows = read(E3/'physical_measurements.json')['runs']; donors = {}
    for r in rows:
        key = digest(r['identity'])
        if key not in donors:
            donors[key] = dict(r, physical_raw_id='e3_'+r['physical_raw_id'], source='8E-3')
    return donors
