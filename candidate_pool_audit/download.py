"""Pinned downloads into NEW directories; never change frozen assets/environments."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/candidate_pool_v2'
REPOS = {'roberta_sst2': 'textattack/roberta-base-SST-2',
         'paddleocr_vl': 'PaddlePaddle/PaddleOCR-VL-1.5',
         'whisper_turbo': 'openai/whisper-large-v3-turbo',
         'distil_whisper': 'distil-whisper/distil-large-v3'}


def fetch(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected is None or path.stat().st_size == expected):
        return {'path': str(path), 'bytes': path.stat().st_size, 'existing': True}
    session = requests.Session()
    session.trust_env = False  # broken host proxy; scoped to this downloader only
    temporary = path.with_suffix(path.suffix + '.partial')
    with session.get(url, timeout=(15, 60), stream=True) as response:
        response.raise_for_status()
        with temporary.open('wb') as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                stream.write(chunk)
    if expected is not None and temporary.stat().st_size != expected:
        raise ValueError(f'Incomplete download: {path}')
    temporary.replace(path)
    print(json.dumps({'downloaded': str(path), 'bytes': path.stat().st_size}), flush=True)
    return {'path': str(path), 'bytes': path.stat().st_size}


def main():
    jobs = []
    for name, repo in REPOS.items():
        meta = json.loads((OUT / 'sources' / (repo.replace('/', '--') + '.api.json')).read_text())
        for item in meta['siblings']:
            filename = item['rfilename']
            if '.fp32.' in filename:
                continue  # alternative serialization; primary weights are sufficient
            if '/' in filename or not filename.endswith(('.json', '.txt', '.model', '.jinja', '.safetensors')):
                continue
            if filename == 'training_args.bin':
                continue
            jobs.append((f'https://hf-mirror.com/{repo}/resolve/{meta["sha"]}/{filename}',
                         ROOT / 'models/candidate_pool_v2' / name / filename, item.get('size')))
    jobs.append(('https://download.pytorch.org/models/convnext_tiny-983f1562.pth',
                 ROOT / 'models/candidate_pool_v2/convnext_tiny/convnext_tiny-983f1562.pth', 114419221))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch, *job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                record = future.result()
                record['status'] = 'complete'
            except Exception as error:
                record = {'path': str(job[1]), 'status': 'failed', 'error': str(error)}
                print(json.dumps(record), flush=True)
            record['source_url'] = job[0]
            results.append(record)
            (OUT / 'download_results.json').write_text(json.dumps(results, indent=2))
    path = ROOT / 'models/candidate_pool_v2/convnext_tiny/convnext_tiny-983f1562.pth'
    if path.exists() and not hashlib.sha256(path.read_bytes()).hexdigest().startswith('983f1562'):
        raise ValueError('Torchvision weight hash mismatch')


if __name__ == '__main__':
    main()
