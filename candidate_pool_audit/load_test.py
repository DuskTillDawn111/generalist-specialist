"""One worker at a time: load once, one warmup, three sanity calls, shutdown.

Memory samples are taken while workers await stdin, never inference peaks.
Uses the historical Worker transport with a logging stdout reader.
"""
import argparse
import json
from pathlib import Path
import queue
import subprocess
import time

from joint_deploy.run_static_hybrid import Worker
from .registry import ROOT, registry

OUT = ROOT / 'results/candidate_pool_v2'
MANIFESTS = {'text': 'text_classification', 'image': 'image_classification',
             'ocr': 'ocr', 'asr': 'asr', 'visual_qa': 'visual_qa', 'lid': 'lid'}


def rows(task):
    return [json.loads(line) for line in (ROOT / f'benchmark/manifests/{MANIFESTS[task]}.jsonl').read_text().splitlines() if line]


def choose(name):
    tasks = {'qwen': ['text', 'visual_qa', 'lid'], 'phi4mm': ['text', 'image', 'asr'],
             'minicpm': ['text', 'image', 'asr'], 'distilbert': ['text'] * 3,
             'roberta_sst2': ['text'] * 3, 'resnet50': ['image'] * 3,
             'convnext_tiny': ['image'] * 3, 'paddleocr': ['ocr'] * 3,
             'paddleocr_vl': ['ocr'] * 3, 'whisper-small': ['asr', 'lid', 'lid'],
             'whisper_turbo': ['asr', 'lid', 'lid'], 'distil_whisper': ['asr'] * 3}[name]
    counters = {}
    result = []
    for task in tasks:
        samples = rows(task)
        index = counters.get(task, 0)
        if task == 'asr':
            samples = [sample for sample in samples if 4 <= sample['duration_s'] < 6]
        elif task in ('text', 'image'):
            distinct = {}
            for sample in samples:
                distinct.setdefault(sample['label'], sample)
            samples = list(distinct.values()) + samples
        result.append((task, samples[index]))
        counters[task] = index + 1
    return result


def gpu_snapshot(pid=None):
    device = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used',
                                      '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    processes = []
    for line in raw.splitlines():
        process, memory = line.split(', ')
        processes.append({'pid': int(process), 'used_gpu_memory_mib': float(memory)})
    target = next((p['used_gpu_memory_mib'] for p in processes if p['pid'] == pid), None)
    return {'time_unix': time.time(), 'gpu_uuid': device[0], 'gpu_name': device[1],
            'capacity_mib': float(device[2]), 'device_used_mib': float(device[3]),
            'processes': processes, 'target_pid': pid, 'target_process_mib': target}


class AuditWorker(Worker):
    def _reader(self):
        with (OUT / 'raw_load_tests' / (self.name + '.log')).open('w') as stream:
            for line in self.process.stdout:
                stream.write(line)
                stream.flush()
                line = line.rstrip()
                if line.startswith(('READY|', 'RESULT|', 'ERROR|', 'BYE|')):
                    kind, payload = line.split('|', 1)
                    self.event_queue.put((self.name, kind, json.loads(payload)))
        self.event_queue.put((self.name, 'EXIT', {'exit_code': self.process.poll()}))


def event(worker, events, wanted, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _, kind, payload = events.get(timeout=.1)
        except queue.Empty:
            if worker.process.poll() is not None:
                raise RuntimeError(f'Worker exited with code {worker.process.returncode}')
            continue
        if kind in ('ERROR', 'EXIT'):
            raise RuntimeError(f'{kind}: {payload}')
        if kind == wanted:
            return payload
    raise TimeoutError(f'{worker.name} waiting for {wanted}')


def idle(pid):
    samples = []
    for _ in range(3):
        time.sleep(.2)
        samples.append(gpu_snapshot(pid))
    if any(s['target_process_mib'] is None for s in samples):
        raise RuntimeError('Missing per-process GPU memory measurement')
    if any(any(p['pid'] != pid for p in s['processes']) for s in samples):
        raise RuntimeError('Concurrent GPU process: single-model audit contaminated')
    return samples


def validate(task, sample, response):
    prediction = response.get('prediction')
    if not isinstance(prediction, str) or not prediction:
        raise ValueError('Nonempty recognized text/class prediction required')
    if task == 'text' and prediction not in ('positive', 'negative'):
        raise ValueError('Invalid SST-2 output label')
    if task == 'image' and prediction.lower() not in {s['label'].lower() for s in rows('image')}:
        raise ValueError('Invalid Imagenette output label')
    if task == 'visual_qa' and prediction not in [chr(65 + i) for i in range(len(sample['choices']))]:
        raise ValueError('Invalid ScienceQA option')
    if task == 'lid' and prediction not in {s['label'] for s in rows('lid')}:
        raise ValueError('Language outside current closed task label set')
    if task == 'asr' and not all(k in response for k in ('word_errors', 'reference_words', 'wer')):
        raise ValueError('Missing ASR scoring fields')


def run(name):
    from .inventory import NEW
    if name in NEW:
        preflight = json.loads((OUT / 'adapter_preflight.json').read_text())[name]
        if preflight['adapter_contract_status'] != 'verified':
            raise ValueError('Adapter contract must pass before minimal load')
        if any(task not in preflight['tasks'] for task, _ in choose(name)):
            raise ValueError('Sanity task was not preflight-audited')
    path = OUT / 'raw_load_tests' / (name + '.json')
    if path.exists():
        print(json.dumps({'event': 'reuse_existing_attempt', 'model': name}), flush=True)
        return
    record = {'model': name, 'status': 'started', 'load_attempts': 1,
              'not_a_benchmark': True, 'sanity_samples': [], 'warmup_samples': [],
              'idle_memory_samples': [], 'before_load': gpu_snapshot(),
              'launcher': {k: str(v) for k, v in registry()[name].items()}}
    if record['before_load']['processes']:
        raise RuntimeError('GPU busy; refusing audit')
    path.write_text(json.dumps(record, indent=2))
    events = queue.Queue()
    worker = AuditWorker(name, registry()[name], events)
    try:
        worker.start()
        record['pid'] = worker.process.pid
        record['ready'] = event(worker, events, 'READY')
        record['idle_memory_samples'].append({'phase': 'post_load', 'samples': idle(worker.process.pid)})
        samples = choose(name)
        for index, (task, sample) in enumerate([samples[0]] + samples):
            request = {'request_id': f'{name}:sanity:{index}', 'pool_id': sample['id'],
                       'arrival_index': index, 'task_type': task, 'sample': sample,
                       'submit_ns': time.time_ns()}
            worker.send(request)
            response = event(worker, events, 'RESULT')
            validate(task, sample, response)
            # Protocol emits per-call timing; it is retained as raw provenance only.
            entry = {'task': task, 'sample_id': sample['id'], 'input_sample': sample,
                     'response': response, 'output_schema_valid': True}
            record['warmup_samples' if index == 0 else 'sanity_samples'].append(entry)
            record['idle_memory_samples'].append({'phase': f'post_{"warmup" if index == 0 else "sanity_" + str(index)}',
                                                  'samples': idle(worker.process.pid)})
        if name in ('roberta_sst2', 'whisper_turbo'):
            relevant = [e for e in record['sanity_samples'] if e['task'] in ('text', 'lid')]
            if not all(e['response'].get('correct') for e in relevant):
                raise ValueError('Opposing sentiment / independent LID sanity verification failed')
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        forced = False
        if worker.process is not None:
            if worker.process.poll() is None:
                worker.shutdown()
                try:
                    record['bye'] = event(worker, events, 'BYE', timeout=30)
                    worker.process.wait(timeout=30)
                except Exception:
                    forced = True
                    worker.process.terminate()
                    try:
                        worker.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        worker.process.kill()
                        worker.process.wait(timeout=10)
            record['shutdown'] = {'exit_code': worker.process.returncode, 'forced': forced}
            worker.process.stdin.close()
        release = []
        for _ in range(3):
            time.sleep(.5)
            release.append(gpu_snapshot(record.get('pid')))
        record['after_unload'] = release
        record['gpu_memory_released'] = all(not s['processes'] and s['device_used_mib'] <= record['before_load']['device_used_mib'] + 2 for s in release)
        values = [s['target_process_mib'] for phase in record['idle_memory_samples'] for s in phase['samples']]
        record['measured_idle_vram_mib'] = max(values) if values else None
        record['memory_definition'] = 'Maximum observed quiescent NVML process MiB at post-load, post-warmup and post-sanity; includes retained CUDA allocator caches; never inference peak.'
        if not record['gpu_memory_released'] or record.get('shutdown', {}).get('forced') or record.get('shutdown', {}).get('exit_code') != 0:
            record['status'] = 'failed'
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps({'event': 'load_test_complete', 'model': name, 'status': record['status'],
                      'idle_mib': record['measured_idle_vram_mib'], 'release': record['gpu_memory_released'],
                      'error': record.get('error')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', choices=list(registry()))
    args = parser.parse_args()
    for name in args.models:
        run(name)
