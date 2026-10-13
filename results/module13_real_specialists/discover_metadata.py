"""Module 13A: bounded public metadata retrieval; no torch, weights, or inference."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

OUT = Path(__file__).resolve().parent
IDS = [
    'openai/whisper-medium', 'openai/whisper-medium.en',
    'openai/whisper-large-v2', 'openai/whisper-large-v3',
    'nvidia/parakeet-tdt-0.6b-v2', 'facebook/wav2vec2-large-960h-lv60-self',
    'speechbrain/lang-id-voxlingua107-ecapa', 'facebook/mms-lid-256',
    'facebook/mms-lid-1024',
    'textattack/bert-base-uncased-SST-2',
    'philschmid/roberta-large-sst2',
    'philschmid/bert-large-cased-whole-word-masking-sst2',
    'textattack/facebook-bart-large-SST-2',
    'vroomhf/deberta-v3-base-finetuned-sst2',
    'textattack/xlnet-base-cased-SST-2',
]
TEXT_FILES = ['README.md', 'config.json', 'preprocessor_config.json',
              'generation_config.json', 'tokenizer_config.json',
              'hyperparams.yaml', 'label_encoder.txt']

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Module13A-public-metadata-audit'})
    with urllib.request.urlopen(req, timeout=35) as r:
        body = r.read(2_000_001)
    if len(body) > 2_000_000:
        raise ValueError('Metadata size cap exceeded')
    return body

def collect(model_id):
    prefix = model_id.replace('/', '--')
    records = []
    api = 'https://huggingface.co/api/models/' + model_id + '?blobs=true'
    data = get(api)
    info = json.loads(data)
    revision = info['sha']
    requests = [('api.json', api, data)]
    files = {x['rfilename'] for x in info['siblings']}
    for name in TEXT_FILES:
        if name in files:
            url = f'https://huggingface.co/{model_id}/resolve/{revision}/{name}'
            requests.append((name, url, get(url)))
    for name, url, body in requests:
        dest = OUT / 'sources' / (prefix + '.' + name)
        dest.write_bytes(body)
        records.append({'url': url, 'path': str(dest),
                        'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)})
    return {'model_id': model_id, 'revision': revision, 'sources': records,
            'access': 'PUBLIC_UNAUTHENTICATED_METADATA_OK', 'gated': info.get('gated', False)}

def main():
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(collect, i): i for i in IDS}
        for f in concurrent.futures.as_completed(futures):
            i = futures[f]
            try:
                r = f.result()
                records.append(r)
                print(i, r['revision'], len(r['sources']), flush=True)
            except Exception as e:
                records.append({'model_id': i, 'error': repr(e)})
                print(i, repr(e), flush=True)
    result = {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'metadata only; no weight downloads; no inference; no CUDA imports',
              'records': sorted(records, key=lambda x: x['model_id'])}
    (OUT / 'source_manifest.json').write_text(json.dumps(result, indent=2) + '\n')

if __name__ == '__main__':
    main()
