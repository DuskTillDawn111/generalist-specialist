"""CPU-only regression checks for a zero-rate phase with no VQA workers."""
import ast, inspect
from dataclasses import replace
from module8c_final_real_search.common import OUT, read, ref, verify_freeze
from joint_deploy.measurement.schema import MeasurementConfig
from integer_replica_pilot.adapter import ReplicaAwareDeploymentRuntime
from integer_replica_pilot.evaluator import runner as original_runner
from .runtime import ZeroRateTransportRuntime
from .runner import runner


def main():
    verify_freeze()
    ctx = read(OUT / 'no_vqa_phase/runtime_context.json')
    failed = next((OUT / 'no_vqa_phase/search_cache/raw_runs').glob('*/rep_1.config.json'))
    config = MeasurementConfig(**read(failed))
    caps = {m: row['supported_tasks'] for m, row in ctx['models'].items()}
    args = (ctx['registry'], caps, ctx['replica_bounds'], ctx['memory_table_mib'], ctx['gpu_capacity_mib'])
    try:
        ReplicaAwareDeploymentRuntime(config, *args)
    except ValueError as exc:
        assert str(exc) == 'Unknown task in workload or routing'
    else:
        raise AssertionError('Expected the preserved pre-transport failure')
    compat = ZeroRateTransportRuntime(config, *args)
    assert compat.omitted_zero_rate_tasks == ['visual_qa']
    active = replace(config, workload_rps={k:v for k,v in config.workload_rps.items() if k != 'visual_qa'},
                     routing_rps={k:v for k,v in config.routing_rps.items() if k != 'visual_qa'})
    original = ReplicaAwareDeploymentRuntime(active, *args)
    assert compat.config.as_dict() == original.config.as_dict()
    assert compat.replica_to_family == original.replica_to_family
    assert compat.planning_memory_mib == original.planning_memory_mib
    assert compat.family_config.as_dict() == config.as_dict()
    # Positive six-task input follows the exact original transport branch.
    positive_config_path = next((OUT / 'demand_balanced/search_cache/raw_runs').glob('*/rep_1.config.json'))
    positive = MeasurementConfig(**read(positive_config_path))
    old = ReplicaAwareDeploymentRuntime(positive, *args)
    new = ZeroRateTransportRuntime(positive, *args)
    assert old.config.as_dict() == new.config.as_dict() and not new.omitted_zero_rate_tasks
    for task, rate in [('visual_qa', 0.1), ('unregistered_task', 0.)]:
        bad = replace(config, workload_rps=dict(config.workload_rps, **{task:rate}))
        try:
            ZeroRateTransportRuntime(bad, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid task accepted: {task}')
    left = ast.parse(inspect.getsource(original_runner))
    right = ast.parse(inspect.getsource(runner))
    for node in ast.walk(right):
        if isinstance(node, ast.Constant) and node.value == 'module8c_runtime_compat.runtime':
            node.value = 'integer_replica_pilot.runtime'
    assert ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)
    from pathlib import Path
    from deployment_search.runtime_cache import save_json
    save_json(OUT / 'zero_rate_transport_preflight.json', {
        'status':'PASS', 'new_gpu_runs':0, 'original_failure_reproduced':True,
        'six_dimensional_family_lambda_preserved':True, 'all_positive_routes_unchanged':True,
        'positive_six_task_transport_unchanged':True, 'invalid_positive_routes_still_rejected':True,
        'unknown_tasks_still_rejected':True, 'runner_AST_unchanged_except_module_entry_point':True,
        'original_protected_code_unchanged':True, 'source':ref(Path(__file__)),
    })
    print('CPU zero-rate transport preflight PASS; no workers launched')


if __name__ == '__main__':
    main()
