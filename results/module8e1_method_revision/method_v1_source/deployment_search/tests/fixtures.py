"""Deterministic synthetic pools and enumeration tripwires."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import itertools

from deployment_core.models import Hardware, ModelProfile
from deployment_core.workload import Workload
from deployment_search import RuntimeEvaluation, SyntheticRuntimeEvaluator


def model(name, memory=2., **services):
    return ModelProfile(name, frozenset(services), memory, services)


def synthetic_pool(size):
    catalog = {'base': model('base', 1., text=600.)}
    for i in range(1, size):
        name = f'text_{i:02d}'
        catalog[name] = model(name, 2. + i / (100 * size), text=600 / (1 + i / size))
    return Workload({'text': 1.}), catalog, Hardware(20.)


def evaluation(mean, state='steady', p95=None, raw=None):
    return RuntimeEvaluation(state, mean, mean * 1.5 if p95 is None else p95,
                             1., 1., {'synthetic': True}, raw)


def synthetic_evaluator(catalog):
    def objective(deployment, workload, routing):
        active = [m for m, n in deployment.items() if n]
        speed = sum(600 / catalog[m].service_time_ms['text'] for m in active)
        # Known artificial diminishing returns plus co-residence penalty.
        # Neither scorer predictions nor measured production labels are used.
        mean = 100 / speed + 20 * max(0, len(active) - 3) ** 2
        return evaluation(mean, raw='synthetic://' + '+'.join(sorted(active)))
    return SyntheticRuntimeEvaluator(objective)


@contextmanager
def forbid_enumeration():
    """Guard both existing repository enumerators and combination primitives."""
    import deployment_core.candidate_generator as generator
    with ExitStack() as stack:
        mocks = []
        for name in ('enumerate_deployments', 'screen_deployments', 'generate_feasible_deployments',
                     'combinations', 'all_combinations'):
            mocks.append(stack.enter_context(patch.object(generator, name, create=True,
                side_effect=AssertionError(f'Forbidden exhaustive call: {name}'))))
        for name in ('combinations', 'product'):
            mocks.append(stack.enter_context(patch.object(itertools, name,
                side_effect=AssertionError(f'Forbidden itertools.{name}'))))
        yield mocks
        assert all(m.call_count == 0 for m in mocks), 'Enumeration was attempted'
