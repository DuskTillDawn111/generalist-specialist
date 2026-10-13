"""Frozen v1 entry point; delegates to unchanged original implementation."""
from pathlib import Path
import json
from deployment_search.integer_search import IntegerGuidedDeploymentSearch as MethodV1
from deployment_search.search import SearchConfig
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.runtime_cache import file_hash

FREEZE = Path(__file__).resolve().parents[1] / 'results/module8e1_method_revision/method_v1_freeze.json'

def verify_source():
    record = json.loads(FREEZE.read_text())
    for group in ('sources', 'trace_sources', 'protected_inputs'):
        for r in record[group].values():
            assert file_hash(r['path']) == r['sha256'], r['path']
    return record

def create(runtime_evaluator=None, **kwargs):
    verify_source()
    kwargs.setdefault('predictor', None)
    kwargs.setdefault('runtime_objective', NoiseAwareRuntimeObjective(1., .05))
    return MethodV1(runtime_evaluator, **kwargs)

def config():
    return SearchConfig(**verify_source()['config'])
