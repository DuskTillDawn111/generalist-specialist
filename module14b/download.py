"""Only four frozen v1.0.0 release artifacts, pinned by publisher hash prefix."""
import hashlib, requests
from module14b.common import *
from module13b.download import ranges


def main():
    entries = read(OUT / 'ocr_gapfill_shortlist.json')['entries']
    manifest = {'at_utc': now(), 'models': [], 'completed': False,
                'order': [e['model_id'] for e in entries], 'checkpoint_substitution': False}
    for e in entries:
        p = Path(e['weight_path']); errors = []
        print('DOWNLOAD', e['model_id'], e['checkpoint'], flush=True)
        try:
            if not p.exists():
                # Same frozen release asset, via its public REST asset ID. This
                # avoids intermittent github.com browser-endpoint timeouts.
                # Never persist or print the signed official redirect URL.
                resolver = requests.Session(); resolver.trust_env = False
                api_url = 'https://api.github.com/repos/baudm/parseq/releases/assets/' + str(e['release_asset_id'])
                with resolver.get(api_url, headers={'Accept': 'application/octet-stream',
                                                    'Range': 'bytes=0-0'},
                                  stream=True, timeout=(20, 30)) as response:
                    response.raise_for_status()
                    assert response.status_code == 206
                    assert response.headers.get('Content-Range') == 'bytes 0-0/' + str(e['observable_checkpoint_bytes'])
                    transport_url = response.url
                try:
                    part = ranges(transport_url, p, e['observable_checkpoint_bytes'])
                except Exception as err:
                    errors.append({'transport': 'ranged', 'error': str(err).split('?')[0][:500]})
                    session = requests.Session(); session.trust_env = False
                    part = Path(str(p) + '.stream.partial')
                    with session.get(transport_url, stream=True, timeout=(25, 120)) as r:
                        r.raise_for_status()
                        with part.open('wb') as f:
                            for chunk in r.iter_content(1048576): f.write(chunk)
                assert part.stat().st_size == e['observable_checkpoint_bytes']
                sha = ref(part)['sha256']
                assert sha.startswith(e['expected_sha256_prefix']), 'publisher SHA prefix mismatch'
                part.replace(p)
            evidence = ref(p)
            assert evidence['bytes'] == e['observable_checkpoint_bytes']
            assert evidence['sha256'].startswith(e['expected_sha256_prefix'])
            manifest['models'].append({'model_id': e['model_id'], 'revision': e['revision'],
                                      'success': True, 'file': evidence, 'publisher_hash_verified': True,
                                      'release_asset_id': e['release_asset_id'],
                                      'transport': 'Exact publisher GitHub release asset REST ID and official redirect; no weight substitution',
                                      'earlier_transport_errors': errors})
            print('DOWNLOADED', e['model_id'], evidence['sha256'], flush=True)
        except Exception as err:
            manifest['models'].append({'model_id': e['model_id'], 'revision': e['revision'],
                                      'success': False, 'error': str(err)[:1000], 'earlier_transport_errors': errors})
            print('DOWNLOAD_FAILED', e['model_id'], str(err)[:500], flush=True)
        save('module14b_download_manifest.json', manifest)
    manifest.update(completed=True, all_success=all(m['success'] for m in manifest['models']))
    save('module14b_download_manifest.json', manifest)


if __name__ == '__main__': main()
