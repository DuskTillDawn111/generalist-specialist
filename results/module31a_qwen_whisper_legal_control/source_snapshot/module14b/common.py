"""Module14B writes new evidence only; historical Module14A stays immutable."""
from module14.common import ROOT, OUT, read, ref, now, save, csvsave
from pathlib import Path
RAW = OUT / 'raw14b'
MODELS = ROOT / 'models/module14b'
COMMIT = '1902db043c029a7e03a3818c616c06600af574be'
SOURCE = MODELS / ('parseq-' + COMMIT)
REGIME = 'QUALITY_NONDEGRADED_HEAVY_SPECIALIST_REGIME'
PYTHON = ROOT / 'envs/module14b_str/bin/python'


def eligibility(point, task):
    baseline = read(OUT / 'baseline_references.json')['tasks'][task]
    q, s, m = [point.get(k) for k in ['quality', 's0_ms', 'safe_memory_mib']]
    complete = all(v is not None for v in [q, s, m])
    flags = {
        'QUALITY_NONDEGRADED' if task == 'image' else 'QUALITY_IMPROVED':
            (q >= baseline['quality'] if task == 'image' else
             q < baseline['quality'] if task == 'asr' else q > baseline['quality']) if complete else False,
        'LATENCY_INCREASED': s > baseline['s0_ms'] if complete else False,
        'MEMORY_INCREASED': m > baseline['safe_memory_mib'] if complete else False,
    }
    return {**point, **flags, 'HEAVY_ELIGIBLE': complete and all(flags.values()),
            'status': 'HEAVY_ELIGIBLE' if complete and all(flags.values()) else 'FAIL_ELIGIBILITY',
            'failed_criteria': [k for k, v in flags.items() if not v],
            'quality_delta': q - baseline['quality'] if q is not None else None,
            'latency_ratio': s / baseline['s0_ms'] if s is not None else None,
            'memory_ratio': m / baseline['safe_memory_mib'] if m is not None else None}


def select(points, task):
    eligible = [p for p in points if p['HEAVY_ELIGIBLE']]
    return min(eligible, key=lambda p: (p['quality'] if task == 'asr' else -p['quality'],
                                       p['s0_ms'], p['safe_memory_mib'],
                                       p['model_id'] + '@' + p['revision'])) if eligible else None
