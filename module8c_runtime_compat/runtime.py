"""Reuse the original harness, projecting only inactive empty transport tasks."""
from dataclasses import replace
from pathlib import Path
from integer_replica_pilot import runtime as legacy
from integer_replica_pilot.adapter import ReplicaAwareDeploymentRuntime
from deployment_search.runtime_cache import save_json, file_hash


class ZeroRateTransportRuntime(ReplicaAwareDeploymentRuntime):
    def __init__(self, config, registry, capabilities, bounds, table, capacity):
        # Validate full family input against the unchanged admission capabilities.
        config.validate(capabilities)
        active_tasks = set().union(*(set(capabilities[m])
                                    for m, count in config.deployment.items() if count))
        keys = set(config.workload_rps) | set(config.routing_rps)
        omitted = keys - active_tasks
        for task in omitted:
            if config.workload_rps.get(task, 0) != 0 or any(config.routing_rps.get(task, {}).values()):
                raise ValueError(f'Nonzero unsupported transport task: {task}')
        projected = replace(config,
            workload_rps={k: v for k, v in config.workload_rps.items() if k not in omitted},
            routing_rps={k: v for k, v in config.routing_rps.items() if k not in omitted})
        super().__init__(projected, registry, capabilities, bounds, table, capacity)
        self.family_config = config
        self.omitted_zero_rate_tasks = sorted(omitted)


_original_run = legacy.run


def run(config, context, output, **kwargs):
    result = _original_run(config, context, output, **kwargs)
    active_tasks = set().union(*(set(context['models'][m]['supported_tasks'])
                                for m, count in config.deployment.items() if count))
    result['zero_rate_transport_compatibility'] = {
        'source_path': str(Path(__file__).resolve()), 'source_sha256': file_hash(__file__),
        'omitted_transport_tasks': sorted((set(config.workload_rps) | set(config.routing_rps)) - active_tasks),
        'original_family_workload_preserved': result['workload_rps'] == config.workload_rps,
        'positive_workload_and_routes_unchanged': True,
        'capabilities_or_search_changed': False,
    }
    save_json(output, result)
    return result


def main():
    legacy.ReplicaAwareDeploymentRuntime = ZeroRateTransportRuntime
    legacy.run = run
    return legacy.main()


if __name__ == '__main__':
    raise SystemExit(main())
