"""Restore only the three blocked real-pool planning runs; no synthetic campaign."""
import math
from pathlib import Path
import shutil

from deployment_core import ModelProfile, Workload
from deployment_core.candidate_pool_v2 import load_candidate_pool_v2, mandatory_presence
from deployment_core.replicas import replica_statistics, replica_identities
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig
from deployment_search.feedback import bottleneck_feedback
from deployment_search.result import jsonable
from .prepare import ROOT, OUT, read, write, reference
from .finalize import completeness

DESTINATION = ROOT / 'results/integer_replica_search_foundation'


def load_completed_catalog(path=None):
    path = Path(path) if path is not None else OUT / 'updated_service_demand_catalog.json'
    data = read(path)
    validation = read(OUT / 'measurement_validation.json')
    if (validation.get('status') != 'passed' or validation.get('still_pending_paths') != 0 or
        validation['catalog']['sha256'] != reference(path)['sha256']):
        raise ValueError('Catalog differs from the independently validated frozen measurement input')
    frozen, hardware = load_candidate_pool_v2()
    if (data.get('frozen') is not True or data.get('status') != 'complete' or
        data.get('unit') != 'milliseconds' or data.get('protocol') != 'active_single_profile_v1' or
        set(data['models']) != set(frozen) or data['gpu_capacity_mib'] != hardware.gpu_memory_mib):
        raise ValueError('Complete frozen real Stage-1 v2 catalog required')
    services = {m: row['service_time_ms'] for m, row in data['models'].items()}
    if completeness(frozen, services)['still_pending_paths']:
        raise ValueError('Pending demands remain; dry-run blocked')
    catalog, roles = {}, {}
    for name, record in frozen.items():
        row = data['models'][name]
        if (row['measured_idle_vram_mib'] != record.memory_mib or
            set(row['capabilities_verified']) != set(record.supported_tasks) or
            set(row['service_time_ms']) != set(record.supported_tasks) or row['role'] != record.role):
            raise ValueError('Frozen candidate capability/memory/role mismatch')
        for task, old_value in record.service_time_ms.items():
            source = row['service_time_sources'][task]
            if source['protocol'] != 'active_single_profile_v1' or source['value_ms'] != row['service_time_ms'][task]:
                raise ValueError('Service-demand provenance mismatch')
            if old_value is not None:
                if row['service_time_ms'][task] != old_value or source['kind'] != 'existing_frozen_stage1':
                    raise ValueError('Existing frozen demand cannot change')
            elif source['kind'] != 'new_real_standalone_measurement' or source['measurement_count'] != 90:
                raise ValueError('Missing real standalone 3x30 measurement evidence')
            else:
                for rep in source['repetitions']:
                    for key in ('raw', 'summary', 'validation'):
                        if reference(rep[key]['path'])['sha256'] != rep[key]['sha256']:
                            raise ValueError('Frozen raw service-demand evidence drift')
        catalog[name] = ModelProfile(name, record.supported_tasks, record.memory_mib, row['service_time_ms'],
            dict(record.sources, service_time=row['service_time_sources'], completed_catalog=reference(path)))
        roles[name] = record.role
    return catalog, hardware, roles, frozen, data


def candidate_record(candidate, current, catalog, roles):
    mutation = candidate.mutation
    delta = mutation.deployment[mutation.model] - current[mutation.model]
    canonical = candidate.canonical
    return {'operation': mutation.operation, 'model_id': mutation.model, 'role': roles[mutation.model],
        'from_count': current[mutation.model], 'to_count': mutation.deployment[mutation.model],
        'deployment': dict(mutation.deployment), 'memory_cost_mib': delta * catalog[mutation.model].memory_mib,
        'total_memory_mib': candidate.memory_mib,
        'memory_saving_mib': max(0., -delta * catalog[mutation.model].memory_mib),
        'feasible': canonical is not None, 'status': candidate.status, 'rejection_reason': candidate.reason or None,
        'delta_z': candidate.delta_z, 'z_candidate': canonical.max_pressure if canonical else None,
        'per_replica_pressure': canonical.model_pressure if canonical else None,
        'ranking': candidate.rank, 'Top_B': candidate.selected_top_b,
        'replica_counts_statistics': replica_statistics(mutation.deployment, roles)}


def run():
    catalog, hardware, roles, records, data = load_completed_catalog()
    workloads = read(ROOT / 'results/deployment_search_multiworkload/benchmark_workloads.json')['workloads']
    outputs = {}
    for index, row in enumerate(workloads, 1):
        w = Workload(row['arrival_rates_rps'])
        result = IntegerGuidedDeploymentSearch(predictor=None).search(w, catalog, hardware,
            SearchConfig(dry_run=True), roles=roles)
        if result.seed is None or not result.seed.feasible or result.canonical is None:
            raise RuntimeError('Complete real catalog produced no feasible seed; record rather than fake outputs')
        if result.runtime_evaluation_count != 0:
            raise RuntimeError('Dry-run cannot evaluate GPU runtime')
        current = result.deployment
        pressure = result.canonical
        feedback = bottleneck_feedback(pressure, current)
        candidates = [candidate_record(c, current, catalog, roles) for c in result.iterations[0]['candidates']] if result.iterations else []
        generalist_specialist_choices = []
        for task in feedback.pressure_tasks:
            capable_ups = [c for c in candidates if c['operation'] == 'SCALE_UP' and
                           task in catalog[c['model_id']].supported_tasks]
            general = [c for c in capable_ups if c['role'] == 'generalist']
            special = [c for c in capable_ups if c['role'] == 'specialist']
            generalist_specialist_choices.append({'bottleneck_task': task,
                'both_generalist_and_specialist_candidates': bool(general and special),
                'generalist_plus_one': general, 'specialist_plus_one': special,
                'ranking_scope': 'delta_z of the full family-level canonical LP, not a task-isolated objective'})
        statistics = replica_statistics(current, roles)
        value = {'workload': row['name'], 'workload_rps': dict(w.arrival_rate), 'model_types': len(catalog),
            'status': result.status, 'blocked': False, 'completed_by_module': '7A-2',
            'input_kind': 'frozen candidate pool v2 idle VRAM + new frozen real standalone Stage-1 demand',
            'service_demand_catalog': reference(OUT / 'updated_service_demand_catalog.json'),
            'workload_source': reference(ROOT / 'results/deployment_search_multiworkload/benchmark_workloads.json'),
            'memory_source': reference(ROOT / 'results/candidate_pool_v2/measured_vram.json'),
            'integer_seed_replica_vector': dict(result.seed.deployment), 'seed_memory_usage_mib': result.seed.objective_value,
            'total_memory_mib': sum(catalog[m].memory_mib * n for m, n in current.items()),
            'gpu_capacity_mib': hardware.gpu_memory_mib, **statistics,
            'canonical_routing': jsonable(pressure.routing), 'per_replica_pressure': pressure.model_pressure,
            'per_replica_pressure_definition': 'D_m/x_m for active families; inactive values are zero placeholders',
            'family_total_demand': {m: sum(v for (task, name), v in pressure.task_contributions.items() if name == m) for m in catalog},
            'task_model_service_demand': jsonable(pressure.task_contributions), 'max_pressure': pressure.max_pressure,
            'bottleneck_model_families': feedback.bottleneck_models, 'bottleneck_tasks': feedback.pressure_tasks,
            'candidate_count': len(candidates), 'one_hop_candidate_bound': 2 * len(catalog),
            'all_replica_mutation_candidates': candidates,
            'feasible_candidates': [c for c in candidates if c['feasible']],
            'top_scale_up_candidates': sorted((c for c in candidates if c['operation'] == 'SCALE_UP' and c['ranking'] is not None), key=lambda c: c['ranking']),
            'top_scale_down_candidates': sorted((c for c in candidates if c['operation'] == 'SCALE_DOWN' and c['ranking'] is not None), key=lambda c: c['ranking']),
            'Top_B': [c for c in candidates if c['Top_B']],
            'generalist_specialist_replica_choice_by_pressure_task': generalist_specialist_choices,
            'mandatory_presence': mandatory_presence(records, w),
            'generalist_composition_note': 'Qwen presence is mandatory due to unique verified VQA coverage; Generalist composition freedom is limited by VQA capability asymmetry.',
            'replica_bounds': result.replica_bounds, 'bound_provenance': result.bound_provenance,
            'future_replica_identities': replica_identities(current),
            'MILP_solves': result.milp_solve_count, 'LP_solves': result.lp_solve_count,
            'runtime_evaluations': 0, 'GPU_measurements': 0, 'real_multi_replica_workers_started': 0,
            'memory_model': 'planning memory surrogate: additive measured idle VRAM',
            'capacity_model': 'analytical planning surrogate, not a claim of linear real-GPU scaling',
            'complete_search_trace': result.to_dict()}
        filename = f'w{index}_integer_dry_run.json'
        old = DESTINATION / filename
        backup = OUT / 'blocked_dry_runs_before' / filename
        if old.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, backup)
        write(OUT / 'integer_dry_runs' / filename, value)
        write(old, value)
        outputs[row['name']] = value
    return outputs


if __name__ == '__main__':
    result = run()
    print({name: {'status': row['status'], 'seed': row['integer_seed_replica_vector'],
                  'N_G': row['N_G'], 'N_S': row['N_S'], 'candidates': row['candidate_count']} for name, row in result.items()})
