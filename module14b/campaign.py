"""Final four OCR paths only, sequential standalone runs, preserve every run."""
import argparse, os, subprocess, threading, time
from module14b.common import *
from module13b import campaign as original

PROTO = read(OUT / 'ocr_gapfill_shortlist.json')
ENTRIES = PROTO['entries']
original.OUT = OUT
original.RAW = RAW
original.SHORT = ENTRIES
original.PROTO = PROTO
original.ENV.update(PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')


def command(entry, cpu=False):
    if entry.get('control'):
        lib = '/root/miniconda3/envs/ocr_env/lib/python3.11/site-packages/nvidia/cu13/lib'
        prior = os.environ.get('LD_LIBRARY_PATH', '')
        return ['/usr/bin/env', 'LD_LIBRARY_PATH=' + lib + (':' + prior if prior else ''),
                '/root/miniconda3/envs/ocr_env/bin/python', str(ROOT / 'joint_deploy/ocr_worker.py')]
    return [str(PYTHON), str(ROOT / 'module14b/worker.py'), '--model-id', entry['model_id']] + (['--cpu-preflight'] if cpu else [])


original.command = command


def cpu():
    assert not (OUT / 'module14b_cpu_preflight.json').exists()
    records = []
    downloads = read(OUT / 'module14b_download_manifest.json')
    assert downloads['completed']
    downloaded = {m['model_id']: m['success'] for m in downloads['models']}
    for e in ENTRIES:
        log = RAW / (original.key(e) + '__cpu.log')
        if not downloaded[e['model_id']]:
            records.append({'model_id': e['model_id'], 'success': False, 'error': 'DOWNLOAD_FAILED'})
            continue
        assert not log.exists()
        print('CPU_PREFLIGHT', e['model_id'], flush=True)
        r = subprocess.run(command(e, True), cwd=ROOT,
                           env={**original.ENV, 'CUDA_VISIBLE_DEVICES': ''},
                           capture_output=True, text=True, timeout=600)
        log.write_text(r.stdout + r.stderr)
        parts = [l.split('|', 1)[1] for l in r.stdout.splitlines() if l.startswith('CPU_PREFLIGHT|')]
        records.append({'model_id': e['model_id'], 'revision': e['revision'],
                        'success': r.returncode == 0 and len(parts) == 1, 'exit_code': r.returncode,
                        'metadata': __import__('json').loads(parts[0]) if parts else None, 'log': str(log)})
        save('module14b_cpu_preflight.json', {'at_utc': now(), 'entries': records, 'completed': False})
        print('CPU_RESULT', e['model_id'], records[-1]['success'], flush=True)
    save('module14b_cpu_preflight.json', {'at_utc': now(), 'entries': records, 'completed': True,
                                        'all_success': all(r['success'] for r in records)})


def gpu():
    freeze = read(OUT / 'module14b_execution_freeze.json')
    for r in freeze['implementation'] + freeze['native_source'] + read(OUT / 'module14b_discovery_freeze.json')['files']:
        assert ref(r['path'])['sha256'] == r['sha256'], r['path']
    pre = read(OUT / 'module14b_cpu_preflight.json')
    ok = {e['model_id']: e['success'] for e in pre['entries']}
    original.nv.nvmlInit()
    try:
        for e in ENTRIES:
            if ok[e['model_id']]: original.run(e, 'quality')
        for rep in [1, 2, 3]:
            original.run({'task': 'ocr', 'model_id': 'paddleocr', 'control': True}, 'latency', rep)
            for e in ENTRIES:
                if ok[e['model_id']]: original.run(e, 'latency', rep)
    finally:
        original.nv.nvmlShutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['cpu', 'gpu'])
    args = p.parse_args()
    cpu() if args.mode == 'cpu' else gpu()
