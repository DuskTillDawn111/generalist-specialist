"""Feature flags over immutable native code, with no copied search algorithm.

FunctionType binds the exact native bytecode to an instance-local globals table.
Only its diversified_order symbol is rebound; no global monkeypatch or mutable
cross-method state exists. Neighborhood dispatch is an ordinary method override.
All seeding, routing, objective, evaluation, rollback and stopping code is native.
"""
from dataclasses import dataclass, asdict
from types import FunctionType
from method_v1 import create as v1_create, config as v1_config
from method_v2 import MethodV2, frozen_config
from method_v2.engine import RevisedGuidedSearch
from method_v2.selection import diversified_order
from deployment_search.scoring import rank_candidates
from deployment_search.integer_search import IntegerGuidedDeploymentSearch


@dataclass(frozen=True)
class Flags:
    enable_targeted_swap: bool
    enable_replacement_closure: bool
    enable_diversified_topb: bool


FLAGS = {'M00': Flags(False, False, False), 'M10': Flags(True, True, False),
         'M01': Flags(False, False, True), 'M11': Flags(True, True, True)}


def flags_dict(method):
    return asdict(FLAGS[method])


class FlaggedMethod(MethodV2):
    def __init__(self, flags, *args, **kwargs):
        self.flags = flags
        assert flags.enable_targeted_swap == flags.enable_replacement_closure
        super().__init__(*args, **kwargs)

    def _mutations(self, deployment, catalog, workload, feedback, threshold):
        if self.flags.enable_targeted_swap:
            return MethodV2._mutations(self, deployment, catalog, workload, feedback, threshold)
        mutations = IntegerGuidedDeploymentSearch._mutations(self, deployment, catalog, workload, feedback, threshold)
        self._generation_audit = {'base_add_remove_candidates': len(mutations), 'targeted_swap_candidates': 0,
            'total_candidates': len(mutations), 'targeted_pairs_constructed': 0,
            'swap_filter_rejections': {'coverage': 0, 'memory': 0},
            'formal_ordered_pair_upper_bound': len(catalog)*(len(catalog)-1), 'deployment_enumeration_count': 0}
        return mutations


def make(method, evaluator, callback=None, flag_endpoint=False):
    """Endpoints use official entrypoints; flag_endpoint exercises shared-code regression."""
    from deployment_search.real_objective import NoiseAwareRuntimeObjective
    kwargs = dict(predictor=None, runtime_objective=NoiseAwareRuntimeObjective(1., .05), trace_callback=callback)
    if method == 'M00' and not flag_endpoint:
        obj = v1_create(evaluator, **kwargs)
    elif method == 'M11' and not flag_endpoint:
        obj = MethodV2(evaluator, **kwargs)
    else:
        flags = FLAGS[method]
        native = RevisedGuidedSearch.search
        bindings = dict(native.__globals__)
        def ranking(candidates, current, catalog, workload, memory):
            return diversified_order(candidates, current, catalog, workload, memory) if flags.enable_diversified_topb else rank_candidates(candidates)
        bindings['diversified_order'] = ranking
        rebound = FunctionType(native.__code__, bindings, native.__name__, native.__defaults__, native.__closure__)
        # MethodV2.search uses super(); the inserted MRO node dispatches the same
        # RevisedGuidedSearch.search code object with local ranking bindings.
        bound_base = type('LocallyBoundNativeLoop', (RevisedGuidedSearch,), {'search': rebound})
        bound_class = type('IsolatedFlaggedMethod', (FlaggedMethod, bound_base), {})
        assert bound_class.mro().index(bound_base) < bound_class.mro().index(RevisedGuidedSearch)
        assert rebound.__code__ is native.__code__
        assert {k for k in bindings if bindings[k] is not native.__globals__[k]} == {'diversified_order'}
        obj = bound_class(flags, evaluator, **kwargs)
    obj.ablation_flags = FLAGS[method]
    return obj


def config(method):
    return v1_config() if method == 'M00' else frozen_config()
