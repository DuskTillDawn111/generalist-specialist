"""At most one ADD or REMOVE per model; never expand combinations."""
from dataclasses import dataclass


def deployment_key(deployment):
    return tuple(sorted(m for m, n in deployment.items() if n))


def memory_mib(deployment, catalog):
    from deployment_core.replicas import normalize_counts
    x = normalize_counts(deployment, catalog)
    return sum(catalog[m].memory_mib * n for m, n in x.items())


def missing_coverage(deployment, catalog, workload):
    covered = set()
    for m in deployment_key(deployment):
        covered.update(catalog[m].supported_tasks)
    return [k for k in workload.active_tasks if k not in covered]


@dataclass
class Mutation:
    operation: str
    model: str
    deployment: dict[str, int]


def generate_neighborhood(deployment, catalog, workload, feedback,
                          remove_pressure_threshold=.2):
    """ADD only models serving pressure tasks; REMOVE only low-pressure covered models.

    Threshold is absolute standalone utilization, not a latency threshold.
    Capacity and memory feasibility are deliberately left to candidate screening.
    """
    candidates = []
    pressure_tasks = set(feedback.pressure_tasks)
    for m in sorted(catalog):
        present = deployment.get(m, 0)
        if not present and not (catalog[m].supported_tasks & pressure_tasks):
            continue
        if present and feedback.model_pressure[m] > remove_pressure_threshold:
            continue
        x = dict(deployment)
        x[m] = 1 - present
        if present and missing_coverage(x, catalog, workload):
            continue
        candidates.append(Mutation('ADD' if not present else 'REMOVE', m, x))
    return candidates


def generate_replica_neighborhood(deployment, catalog, workload, feedback, replica_bounds,
                                 remove_pressure_threshold=.2):
    """At most two direct mutations per family; never enumerate count vectors.

    Low per-replica pressure guides SCALE_DOWN, but every positive count can be
    reduced. Feasibility/ranking decides whether that capacity saving is useful.
    SCALE_UP includes the bottleneck family and alternatives serving its tasks.
    """
    from deployment_core.replicas import normalize_counts
    x = normalize_counts(deployment, catalog, replica_bounds)
    tasks = set(feedback.pressure_tasks)
    candidates = []
    ordered = sorted(catalog, key=lambda m: (m not in feedback.bottleneck_models, m))
    for m in ordered:
        if x[m] < replica_bounds[m] and (m in feedback.bottleneck_models or catalog[m].supported_tasks & tasks):
            candidates.append(Mutation('SCALE_UP', m, dict(x, **{m: x[m] + 1})))
        if x[m] > 0:
            candidates.append(Mutation('SCALE_DOWN', m, dict(x, **{m: x[m] - 1})))
    return candidates
