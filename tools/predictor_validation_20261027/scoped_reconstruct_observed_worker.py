"""Measure the scope of strict runtime settings without editing frozen workers.

G2 always retains the separately tested ExLlama reconstruction path and strict
settings. Specialists either retain all-strict settings or run their ordinary
fresh-process defaults. The actual post-load settings are recorded in READY.
This is a diagnostic runtime, not a qualified predictor or equivalence claim.
"""
import json
import os
from pathlib import Path
import sys


def configure_scope(slot, scope, environ, torch):
    if scope not in {'all_strict', 'g2_strict'}:
        raise ValueError('Explicit all_strict/g2_strict runtime scope required')
    if slot not in {'generalist_2', 'text', 'image', 'ocr', 'asr'}:
        raise ValueError('Declared five-model slot required')
    strict = scope == 'all_strict' or slot == 'generalist_2'
    if torch.cuda.is_initialized():
        raise ValueError('Runtime policy must be set before CUDA initialization')
    if strict:
        environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        if torch.are_deterministic_algorithms_enabled():
            raise ValueError('Ordinary specialist must start in a fresh default process')
    return strict


def runtime_evidence(torch, slot, scope, strict):
    evidence = dict(schema='gsdeploy.runtime_scope_observation.v1', slot=slot,
        scope=scope, strict_requested=strict,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    if strict and (not evidence['deterministic_algorithms'] or evidence['deterministic_warn_only']
                   or evidence['cudnn_benchmark'] or not evidence['cudnn_deterministic']
                   or evidence['cublas_workspace_config'] != ':4096:8'):
        raise ValueError('Strict runtime changed during model loading')
    if not strict and (evidence['deterministic_algorithms'] or evidence['cublas_workspace_config'] is not None):
        raise ValueError('Ordinary specialist unexpectedly inherited strict settings')
    return evidence


def main():
    source = sys.argv[sys.argv.index('--source-root')+1]
    spec = json.loads(Path(sys.argv[sys.argv.index('--spec')+1]).read_text())
    slot = spec['candidate']['slot']
    scope = os.environ.get('GS_RUNTIME_SCOPE')
    # Set workspace before importing torch; CUDA is still checked below.
    if scope == 'all_strict' or scope == 'g2_strict' and slot == 'generalist_2':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    elif scope == 'g2_strict':
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    sys.path.insert(0, source)
    import torch
    strict = configure_scope(slot, scope, os.environ, torch)
    from gsdeploy.backends import compact_native
    from reconstruct_observed_worker import configure_model
    original = compact_native.ProfileAdapter
    class ScopedAdapter(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if slot == 'generalist_2':
                configure_model(self)
    compact_native.ProfileAdapter = ScopedAdapter
    import complete_observed_worker as complete
    original_emit = complete.emit
    def emit(kind, payload):
        if kind == 'READY':
            payload = dict(payload, runtime_scope=runtime_evidence(torch, slot, scope, strict))
        original_emit(kind, payload)
    complete.emit = emit
    mode = os.environ.get('GS_SCOPED_OBSERVER')
    if mode == 'semantic':
        import base_forward_observed_worker as worker
    elif mode == 'complete':
        worker = complete
    else:
        raise ValueError('Explicit complete/semantic observer required')
    worker.main()


if __name__ == '__main__':
    main()
