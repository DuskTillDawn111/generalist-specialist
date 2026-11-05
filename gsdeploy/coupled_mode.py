"""Safe zero bounds for nonseparable prediction modes.

route_load_replay_v1 remains an unimplemented historical placeholder.
paper_profiled_fifo_v1 executes the explicit whole-context event predictor.
Neither mode inherits independent-service capacity or memory bounds.
"""

MODE = 'route_load_replay_v1'
PLACEHOLDER_SCHEMA = 'gsdeploy.route_load_replay.placeholder.v1'


def is_coupled(data):
    return data.get('performance_mode') in {MODE, 'paper_profiled_fifo_v1'}


def validate_placeholder(data):
    if data.get('performance_mode') == 'paper_profiled_fifo_v1':
        from .paper_forecast import validate_spec

        validate_spec(data)
        return
    spec = data.get(MODE)
    if not isinstance(spec, dict) or spec.get('schema') != PLACEHOLDER_SCHEMA or spec.get('status') != 'UNIMPLEMENTED':
        raise ValueError('ROUTE_LOAD_MODEL_NOT_IMPLEMENTED: explicit UNIMPLEMENTED placeholder schema required')
    if data.get('memory_mode') != MODE:
        raise ValueError('Route/load placeholder requires explicit coupled memory mode')


def unknown_service(data):
    """Fail closed for every complete condition, including validation bypasses."""
    try:
        validate_placeholder(data)
        reason = 'ROUTE_LOAD_MODEL_NOT_IMPLEMENTED'
    except ValueError:
        reason = 'UNSUPPORTED_ROUTE_LOAD_SCHEMA'
    return None, dict(mode=MODE, reason=reason, prediction_available=False,
                      scope='Safety placeholder only; no formal service or resource evidence')


def bound_contract(data):
    if not is_coupled(data):
        return {}
    implemented = data.get('performance_mode') == 'paper_profiled_fifo_v1'
    return dict(schema='gsdeploy.coupled_zero_search_bounds.v1',
        mode=data.get('performance_mode'), complete_prediction_implemented=implemented,
        service_lower_ms=0.0, pressure_lower_ms=0.0, memory_lower_mib=0.0,
        energy_lower_normalized=0.0, capacity_witnesses=False,
        standalone_fallback_permitted=False, diagnostic_missing_energy_mode=False,
        quality_bound='Original exact routed frozen quality, or zero for unavailable quality',
        scope='Mathematical nonnegative-cost relaxation only, never actual service, capacity, memory or energy')
