"""Load the frozen v2 sources in a private namespace; replace only ranking.

No copied stack, source edits, global monkeypatch of v2, or generation change.
The private engine retains canonical feedback, feasibility, and real rollback.
"""
import importlib.util,sys,types
from pathlib import Path
from deployment_search.runtime import RuntimeEvaluator
from deployment_core.search.e2e_candidate_ranker import E2ECandidateRanker

def create_search(evaluator,catalog,calibration,safety_factor=1.,ranking='delta_J',trace_callback=None):
    if ranking=='delta_z':
        from method_v2 import MethodV2
        return MethodV2(evaluator,trace_callback=trace_callback)
    if ranking!='delta_J':raise ValueError('Unknown ranking')
    name='module9bc._private_v2'
    source=Path(__file__).resolve().parents[1]/'method_v2'
    package=types.ModuleType(name);package.__path__=[str(source)];sys.modules[name]=package
    loaded={}
    for part in ['engine','integer_search']:
        spec=importlib.util.spec_from_file_location(name+'.'+part,source/(part+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);loaded[part]=module
    ranker=E2ECandidateRanker(catalog,calibration,safety_factor)
    loaded['engine'].diversified_order=ranker.order
    engine=loaded['integer_search'].MethodV2(evaluator,trace_callback=trace_callback)
    engine.e2e_ranker=ranker
    return engine

class RoutingRuntimeAdapter(RuntimeEvaluator):
    """Substitute serving routing while retaining canonical generation feedback."""
    def __init__(self,evaluator,catalog,calibration):
        self.evaluator=evaluator;self.ranker=E2ECandidateRanker(catalog,calibration);self.actual_routes=[]
    def evaluate(self,deployment,workload,routing):
        fit=self.ranker.evaluate(deployment,workload)
        if not fit['converged']:raise RuntimeError('J routing failed; do not silently substitute')
        self.actual_routes.append({'deployment':dict(deployment),'routing':dict(fit['routing'])})
        return self.evaluator.evaluate(deployment,workload,fit['routing'])
