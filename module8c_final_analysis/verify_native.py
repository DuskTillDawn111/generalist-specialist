"""Post-campaign source verification; no GPU work or runtime measurements."""
from pathlib import Path
from module8c_final_real_search.common import OUT, ROOT, read, ref, verify_freeze
from deployment_search.runtime_cache import file_hash, save_json


def main():
    assert read(OUT / 'campaign_status.json')['status'] == 'complete'
    verify_freeze()
    source = ROOT / 'results/integer_replica_search_real_pilot/before_hashes.json'
    expected = read(source)['native_models_and_adapters']
    observations = {}
    errors = []
    for name, record in expected.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        actual = file_hash(path)
        matches = actual == record['sha256']
        observations[name] = {'path': str(path), 'expected_sha256': record['sha256'],
                              'actual_sha256': actual, 'matches': matches}
        if not matches:
            errors.append(name)
    save_json(OUT / 'final_native_source_validation.json', {
        'status': 'PASS' if not errors else 'FAIL', 'source_manifest': ref(source),
        'verification_source': ref(Path(__file__)), 'files_checked': len(observations),
        'all_original_native_sources_unchanged': not errors,
        'new_gpu_runs': 0, 'observations': observations, 'errors': errors,
    })
    assert not errors, errors
    print(f'Native source verification PASS: {len(observations)} files unchanged')


if __name__ == '__main__':
    main()
