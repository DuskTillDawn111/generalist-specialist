"""Integer count schema and recorded physical bounds; CPU planning only."""
import math


def normalize_counts(deployment, models=None, bounds=None):
    if not isinstance(deployment, dict) or any(not isinstance(m, str) or not m or
            type(n) is not int or n < 0 for m, n in deployment.items()):
        raise ValueError('Deployment must be dict[str, nonnegative int]; booleans are not counts')
    names = set(deployment) if models is None else set(models)
    if set(deployment) - names:
        raise ValueError('Deployment contains an unknown model type')
    if bounds is not None:
        validate_bounds(bounds, names)
        if any(n > bounds[m] for m, n in deployment.items()):
            raise ValueError('Replica count exceeds its upper bound')
    return {m: deployment.get(m, 0) for m in sorted(names)}


def validate_bounds(bounds, models):
    if not isinstance(bounds, dict) or set(bounds) != set(models) or any(
            type(n) is not int or n < 0 for n in bounds.values()):
        raise ValueError('Replica bounds must exactly cover the catalog with nonnegative integers')


def derive_replica_bounds(catalog, hardware, worker_limits=None, binary_mode=False):
    if worker_limits is not None and (set(worker_limits) - set(catalog) or any(
            type(n) is not int or n < 0 for n in worker_limits.values())):
        raise ValueError('Worker limits must be nonnegative integer counts for known models')
    bounds, records = {}, {}
    for m, profile in sorted(catalog.items()):
        value = profile.memory_mib
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Positive measured per-replica memory required')
        memory_bound = math.floor(hardware.gpu_memory_mib / value)
        limit = None if worker_limits is None else worker_limits.get(m)
        selected = min(memory_bound, limit) if limit is not None else memory_bound
        if binary_mode:
            selected = min(selected, 1)
        bounds[m] = selected
        records[m] = {'memory_mib': value, 'memory_source': profile.sources.get('memory', {}),
                      'gpu_capacity_mib': hardware.gpu_memory_mib, 'R_mem': memory_bound,
                      'worker_process_limit': limit, 'binary_compatibility_limit': 1 if binary_mode else None,
                      'R': selected, 'bound_source': 'floor(GPU capacity / per-replica memory)' +
                      (' intersect explicit worker/process limit' if limit is not None else '') +
                      (' intersect explicit binary compatibility mode' if binary_mode else '')}
    return bounds, records


def replica_key(deployment):
    """Count-sensitive, hashable family state; zero entries omitted canonically."""
    return tuple((m, n) for m, n in normalize_counts(deployment).items() if n)


def replica_identities(deployment):
    return {m: [f'{m}#{i}' for i in range(n)] for m, n in normalize_counts(deployment).items() if n}


def presence(deployment):
    return {m: int(n > 0) for m, n in normalize_counts(deployment).items()}


def replica_statistics(deployment, roles):
    x = normalize_counts(deployment)
    if set(x) - set(roles) or any(roles[m] not in ('generalist', 'specialist') for m in x):
        raise ValueError('Explicit generalist/specialist role required for each family')
    general = [m for m in x if roles[m] == 'generalist']
    special = [m for m in x if roles[m] == 'specialist']
    return {'active_generalist_types': sum(x[m] > 0 for m in general),
            'active_specialist_types': sum(x[m] > 0 for m in special),
            'N_G': sum(x[m] for m in general), 'N_S': sum(x[m] for m in special),
            'total_generalist_replicas': sum(x[m] for m in general),
            'total_specialist_replicas': sum(x[m] for m in special)}
