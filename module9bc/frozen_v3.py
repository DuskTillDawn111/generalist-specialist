"""Construct the evidence-selected minimal method from its frozen gates.

The original v2 result keeps canonical planning feedback. snapshot() also records
actual serving routes and J scores so the two route identities cannot be confused.
"""
from .common import OUT,read,ROOT,serial
from .integration import create_search,RoutingRuntimeAdapter

def create(runtime_evaluator,catalog,calibration,trace_callback=None):
    path=OUT/'v3_minimal_frozen.json'
    if not path.exists():raise RuntimeError('9BC validation/selection is not complete')
    frozen=read(path)
    factor=read(ROOT/'results/module8f1a_memory_breakpoints/memory_safety_factor.json')['safety_factor']
    routing_adapter=RoutingRuntimeAdapter(runtime_evaluator,catalog,calibration) if frozen['routing']=='J_ROUTING' else None
    evaluator=routing_adapter or runtime_evaluator
    engine=create_search(evaluator,catalog,calibration,factor,
                         ranking='delta_J' if frozen['candidate_ranking']=='DELTA_J' else 'delta_z',
                         trace_callback=trace_callback)
    engine.frozen_v3_definition=frozen
    engine.serving_routing_adapter=routing_adapter
    return engine

def snapshot(engine,result,workload):
    data=result.to_dict()
    data['v3_definition']=engine.frozen_v3_definition
    adapter=engine.serving_routing_adapter
    data['actual_serving_route_evaluations']=[{'deployment':r['deployment'],'routing_rps':serial(r['routing'])}
                                            for r in adapter.actual_routes] if adapter else []
    data['canonical_fields_are_planning_feedback']=True
    ranker=getattr(engine,'e2e_ranker',None)
    if ranker:
        for iteration in data['iterations']:
            base=ranker.evaluate(iteration['deployment'],workload)
            iteration['J_current_seconds']=base['evaluation']['J_seconds']
            for candidate in iteration['candidates']:
                if candidate['canonical'] is None or candidate['rank'] is None:continue
                fit=ranker.evaluate(candidate['mutation']['deployment'],workload)
                candidate['J_star_seconds']=fit['evaluation']['J_seconds']
                candidate['delta_J_seconds']=base['evaluation']['J_seconds']-fit['evaluation']['J_seconds']
                candidate['J_routing_max_rho']=fit['evaluation']['z']
    return data
