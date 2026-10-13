"""CPU fixture only. Uses the real protocol; never used in production configs."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worker_protocol import run_worker


def handler(request):
    sample = request['sample']
    time.sleep(sample.get('test_delay_seconds', 0.002))
    if sample.get('test_error'):
        raise RuntimeError('deliberate protocol fixture error')
    return {'prediction': 'test-only', 'reference': 'test-only'}


run_worker('cpu_fixture', handler, {'tasks': ['text', 'asr', 'lid', 'visual_qa']})
