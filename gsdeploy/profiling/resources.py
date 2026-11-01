"""Pinned local checkpoint preparation; downloads and hashes files, never loads models."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request

from .prepare import digest


def selected_files(metadata):
    files = metadata['siblings']
    names = {f['rfilename'] for f in files}
    weights = set()
    if 'model.safetensors' in names:
        weights.add('model.safetensors')
    elif 'model.safetensors.index.json' in names:
        weights.update(n for n in names if n.endswith('.safetensors') and '/' not in n and 'fp32' not in n)
    elif 'pytorch_model.bin' in names:
        weights.add('pytorch_model.bin')
    else:
        raise ValueError('No supported canonical checkpoint format')
    weights.update(n for n in names if n.endswith('/adapter_model.safetensors'))
    # The pinned Omni loader reads this even when the Talker is disabled.
    weights.update(n for n in names if n == 'spk_dict.pt')
    return [f for f in files if f['rfilename'] in weights or (
        not f['rfilename'].endswith(('.safetensors', '.bin', '.h5', '.msgpack', '.pt', '.onnx'))
        and (f['rfilename'].endswith(('.json', '.py', '.txt', '.model', '.tiktoken'))
             or f['rfilename'] in {'README.md', 'LICENSE', 'LICENSE.txt', 'NOTICE'}))]


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    if entry.get('lfs'):
        return digest(path) == entry['lfs']['sha256']
    raw = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == entry['blobId']


def download_file(url, path, entry, workers=16):
    """Parallel disjoint ranges into one sparse partial file, bounded retries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if os.environ.get('GS_DOWNLOAD_DIRECT') == '1' else urllib.request.build_opener()
    if verified(path, entry):
        return 'verified_existing'
    partial = path.with_name(path.name + '.partial')
    progress = path.with_name(path.name + '.ranges.json')
    size = entry['size']
    if size < 16 * 1024 * 1024:
        for attempt in range(5):
            try:
                with opener.open(url, timeout=25) as response:
                    raw = response.read(size + 1)
                if len(raw) != size:
                    raise ValueError('Incorrect downloaded size')
                partial.write_bytes(raw)
                if not verified(partial, entry):
                    raise ValueError('Source content hash mismatch')
                partial.replace(path)
                return 'downloaded'
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))
    chunk = 8 * 1024 * 1024
    state = {'size': size, 'chunk': chunk, 'source': entry.get('lfs', {}).get('sha256'), 'complete': []}
    if progress.exists() and partial.exists():
        old = json.loads(progress.read_text())
        if all(old.get(k) == state[k] for k in ['size', 'source']):
            if old.get('chunk') == chunk:
                state = old
            elif old.get('chunk', 0) > chunk and old['chunk'] % chunk == 0:
                factor = old['chunk'] // chunk
                state['complete'] = [j for i in old['complete'] for j in range(i * factor, min((i + 1) * factor, (size + chunk - 1) // chunk))]
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    # Reuse the resolved CDN address in memory. Resolving the Hub redirect on
    # every 8 MiB range is expensive on high-latency links. Signed addresses
    # never enter inventories/progress files; a failed cached route is resolved
    # again, and exact ranges plus the final pinned hash remain mandatory.
    route = {'url': url}

    def save():
        temporary = progress.with_suffix('.tmp')
        temporary.write_text(json.dumps(state))
        temporary.replace(progress)

    def transfer(index):
        start, end = index * chunk, min(size, (index + 1) * chunk) - 1
        for attempt in range(6):
            with lock:
                request_url = route['url']
            try:
                request = urllib.request.Request(request_url, headers={'Range': f'bytes={start}-{end}'})
                with opener.open(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError('Server did not honor the exact requested range')
                    resolved = response.geturl()
                    if resolved.startswith('https://'):
                        with lock:
                            route['url'] = resolved
                    offset = start
                    while offset <= end:
                        data = response.read(min(1024 * 1024, end - offset + 1))
                        if not data:
                            raise IOError('Incomplete range')
                        view = memoryview(data)
                        while view:
                            count = os.pwrite(fd, view, offset)
                            if count <= 0:
                                raise IOError('Incomplete disk write')
                            offset += count
                            view = view[count:]
                with lock:
                    state['complete'].append(index)
                    save()
                return
            except Exception:
                with lock:
                    if request_url != url and route['url'] == request_url:
                        route['url'] = url
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 10))
    try:
        needed = [i for i in range((size + chunk - 1) // chunk) if i not in state['complete']]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(transfer, i) for i in needed]):
                future.result()
        os.fsync(fd)
    finally:
        os.close(fd)
    if not verified(partial, entry):
        # Resume metadata cannot certify altered/corrupt partial bytes.
        progress.unlink(missing_ok=True)
        raise ValueError('Completed checkpoint hash mismatch; resume state cleared')
    partial.replace(path)
    progress.unlink(missing_ok=True)
    return 'downloaded'


def prepare_resources(study_path, metadata_root, assets, output, reuse=None, small_only=False):
    import fcntl
    assets = Path(assets).resolve()
    assets.mkdir(parents=True, exist_ok=True)
    with (assets / '.compact-resource-download.lock').open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another compact resource download is active for this asset tree') from error
        return _prepare_resources(study_path, metadata_root, assets, output, reuse, small_only)


def _prepare_resources(study_path, metadata_root, assets, output, reuse=None, small_only=False):
    study = json.loads(Path(study_path).read_text())
    assets, output, metadata_root = Path(assets).resolve(), Path(output).resolve(), Path(metadata_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reuse = reuse or {}
    mirrors = json.loads(Path(os.environ['GS_WEIGHT_MIRRORS']).read_text()) if os.environ.get('GS_WEIGHT_MIRRORS') else {}
    candidates = study['generalists'] + study['specialists']
    inventory = {'schema': 'gsdeploy.local_resources.v1', 'gpu_started': False, 'models_loaded': 0,
                 'status': 'PREPARING', 'models': {}}
    work = []
    for candidate in candidates:
        metadata = json.loads((metadata_root / (candidate['slot'] + '-hub.json')).read_text())
        if metadata['sha'] != candidate['revision']:
            raise ValueError('Metadata revision mismatch')
        entries = selected_files(metadata)
        directory = assets / 'models' / candidate['model'] / candidate['revision']
        record = {'model': candidate['model'], 'revision': candidate['revision'], 'path': str(directory), 'files': []}
        inventory['models'][candidate['slot']] = record
        for entry in entries:
            if Path(entry['rfilename']).is_absolute() or '..' in Path(entry['rfilename']).parts:
                raise ValueError('Unsafe source filename')
            path = directory / entry['rfilename']
            item = {'name': entry['rfilename'], 'size': entry['size'], 'sha256': entry.get('lfs', {}).get('sha256'),
                    'git_blob': entry['blobId'], 'status': 'PENDING'}
            record['files'].append(item)
            work.append((candidate, entry, path, item))
    def save_inventory():
        temporary = output / 'resource-inventory.json.tmp'
        temporary.write_text(json.dumps(inventory, indent=2) + '\n')
        temporary.replace(output / 'resource-inventory.json')
    save_inventory()
    # Fetch tokenizer/config/native source first so CPU audits can proceed.
    work.sort(key=lambda row: row[1]['size'])
    lock = threading.Lock()
    failures = []

    def process(row):
        candidate, entry, path, item = row
        if small_only and entry['size'] >= 100 * 1024 * 1024:
            return
        try:
            old_root = reuse.get(candidate['slot'])
            if old_root and not path.exists():
                old = Path(old_root) / entry['rfilename']
                if verified(old, entry):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.link(old, path)
            url = os.environ.get('GS_HUB_ENDPOINT', 'https://huggingface.co').rstrip('/') + '/' + candidate['model'] + '/resolve/' + candidate['revision'] + '/' + entry['rfilename']
            alternate = mirrors.get(candidate['model'], {}).get(entry['rfilename'])
            if alternate:
                if alternate['sha256'] != entry.get('lfs', {}).get('sha256'):
                    raise ValueError('Alternate source does not match the original pinned content hash')
                url = alternate['url']
            item['source_url'] = url
            status = download_file(url, path, entry)
            stat = path.stat()
            item.update(status='VERIFIED', method=status,
                        stat={'inode': stat.st_ino, 'device': stat.st_dev, 'size': stat.st_size,
                              'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns})
        except Exception as error:
            item.update(status='FAILED', error=str(error))
            failures.append((candidate['slot'], entry['rfilename'], str(error)))
        with lock:
            save_inventory()
        print(candidate['slot'], entry['rfilename'], item['status'], flush=True)
    with ThreadPoolExecutor(max_workers=3 if not small_only else 8) as pool:
        for future in as_completed([pool.submit(process, row) for row in work]):
            future.result()
    inventory['status'] = 'COMPLETE' if all(f['status'] == 'VERIFIED' for r in inventory['models'].values() for f in r['files']) else 'SMALL_FILES_READY' if small_only and not failures else 'INCOMPLETE'
    save_inventory()
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='configs/study.json')
    parser.add_argument('--metadata-root', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse', help='JSON mapping slots to existing read-only checkpoint directories')
    parser.add_argument('--small-only', action='store_true')
    args = parser.parse_args()
    reuse = json.loads(Path(args.reuse).read_text()) if args.reuse else None
    inventory = prepare_resources(args.study, args.metadata_root, args.assets, args.output, reuse, args.small_only)
    print(inventory['status'])


if __name__ == '__main__':
    main()
