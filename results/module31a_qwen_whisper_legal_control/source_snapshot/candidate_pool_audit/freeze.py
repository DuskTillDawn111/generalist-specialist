"""Resource/compatibility freeze gate only. No optimizer or deployment generation."""


def evaluate(models, measured, compatibility, integrity_ok, capacity_mib=49140):
    admitted = [m for m in models if m['capabilities_verified'] and m['adapter_status'] == 'verified'
                and measured.get(m['id'], {}).get('status') == 'passed'
                and measured[m['id']].get('gpu_memory_released') is True]
    ids = {m['id'] for m in admitted}
    unknown = [m['id'] for m in admitted if m['measured_idle_vram_mib'] is None]
    total = None if unknown else sum(m['measured_idle_vram_mib'] for m in admitted)
    counts = {task: {'capable_models': sum(task in m['capabilities_verified'] for m in admitted),
                     'specialists': sum(m['role'] == 'specialist' and task in m['capabilities_verified'] for m in admitted),
                     'generalists': sum(m['role'] == 'generalist' and task in m['capabilities_verified'] for m in admitted)}
              for task in ('text', 'image', 'ocr', 'asr', 'visual_qa', 'lid')}
    generalists = [m for m in admitted if m['role'] == 'generalist']
    specialists = [m for m in admitted if m['role'] == 'specialist']
    consistent = all(compatibility[m['id']][task]['status'] == 'verified'
                     for m in admitted for task in m['capabilities_verified'])
    criteria = {'model_type_count_10_to_15': 10 <= len(admitted) <= 15,
                'at_least_two_verified_local_generalists': len(generalists) >= 2,
                'three_main_tasks_with_two_specialists': sum(counts[t]['specialists'] >= 2 for t in ('text', 'image', 'ocr', 'asr')) >= 3,
                'all_admitted_idle_vram_measured': not unknown,
                'pool_idle_vram_strictly_exceeds_actual_gpu_capacity': total is not None and total > capacity_mib,
                'verified_capabilities_match_matrix': consistent,
                'source_and_environment_integrity': integrity_ok}
    return {'frozen': all(criteria.values()), 'can_enter_module_7b': all(criteria.values()),
            'criteria': criteria, 'formal_model_ids': sorted(ids),
            'verified_generalists': len(generalists), 'verified_specialists': len(specialists),
            'total_model_types': len(admitted), 'task_coverage': counts,
            'tasks_with_only_one_specialist': [t for t in counts if counts[t]['specialists'] == 1],
            'tasks_with_zero_specialists': [t for t in counts if counts[t]['specialists'] == 0],
            'gpu_capacity_mib': capacity_mib, 'pool_idle_vram_mib': total,
            'excess_over_gpu_capacity_mib': total - capacity_mib if total is not None else None,
            'full_pool_all_resident_under_audited_idle_accounting': total <= capacity_mib if total is not None else None,
            'full_pool_joint_residency_empirically_tested': False,
            'generalist_idle_vram_mib': {m['id']: m['measured_idle_vram_mib'] for m in generalists},
            'verified_generalist_sum_idle_vram_mib': sum(m['measured_idle_vram_mib'] for m in generalists) if not any(m['id'] in unknown for m in generalists) else None,
            'resource_competition_established': total is not None and total > capacity_mib,
            'accounting_note': 'Sum separately measured per-process idle resident memory including worker CUDA contexts/caches. Joint load prohibited and not performed. No latency, quality ranking, replica count or optimizer input is fabricated.'}
