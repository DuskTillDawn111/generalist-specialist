"""Continue only the unmeasured no-VQA phase after a transport preflight repair."""
import fcntl, os, traceback
from pathlib import Path
from module8c_final_real_search import campaign as original
from module8c_final_real_search.common import OUT, read, ref, verify_freeze, save_json
from integer_replica_pilot import campaign as confirmation_module
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator
from .runner import runner


class CompatibleEvaluator(ReplicaRuntimeEvaluator):
    def __init__(self, *args, **kwargs):
        kwargs['run'] = runner
        super().__init__(*args, **kwargs)


def verify_compatibility_freeze():
    from deployment_search.runtime_cache import file_hash
    freeze = read(OUT / 'zero_rate_transport_recovery.json')
    for record in freeze['compatibility_sources'].values():
        assert file_hash(record['path']) == record['sha256']
    for record in freeze['failed_attempt_artifacts'].values():
        assert file_hash(record['path']) == record['sha256']
    return freeze


def finalize_report():
    from module8c_final_real_search.report import report
    report()
    verify_compatibility_freeze()
    d = OUT / 'no_vqa_phase'
    cache = read(d / 'search_cache/runtime_cache_manifest.json')
    conf = read(d / 'confirmation_results.json')
    reps = [r['runtime'] for e in cache['configurations'].values() for r in e['repetitions']]
    reps += [r for rows in conf['repetitions'].values() for r in rows]
    for rep in reps:
        raw = read(rep['raw_result_path'])
        assert raw['workload_rps'] == read(d / 'protocol.json')['workload']['arrival_rates_rps']
        assert raw['workload_rps']['visual_qa'] == 0
        assert not any(r['task'] == 'visual_qa' for field in (
            'request_records', 'warmup_request_records', 'priming_request_records') for r in raw[field])
        compatibility = raw['zero_rate_transport_compatibility']
        assert compatibility['original_family_workload_preserved']
        assert not compatibility['capabilities_or_search_changed']
        assert ref(compatibility['source_path'])['sha256'] == compatibility['source_sha256']
    summary = read(OUT / 'module8c_summary.json')
    summary['failed_premeasurement_physical_attempts'] = 2
    summary['total_physical_attempts'] = summary['total_valid_campaign_physical_attempts'] + 2
    summary['runtime_transport_compatibility_layer_used'] = True
    summary['transport_recovery_source'] = ref(OUT / 'zero_rate_transport_recovery.json')
    save_json(OUT / 'module8c_summary.json', summary)
    audit = read(OUT / 'audit_validation.json')
    audit.update(zero_rate_transport_compatibility_checked=True,
                 no_VQA_raw_repetitions_checked=len(reps),
                 failed_premeasurement_attempts_preserved=2)
    save_json(OUT / 'audit_validation.json', audit)
    path = OUT / 'module8c_summary.md'
    md = path.read_text().replace('另保留一次 premeasurement startup failure',
                                  '另保留两次 premeasurement failure')
    md += ('\n## Zero-rate transport recovery\n\n'
           'No-VQA 首次调用在 0.36 秒的 transport 配置校验中退出：没有 workers、没有测量数据。'
           '原 transport 仅根据驻留模型的 capability 校验全六维 workload，因 VQA=0 且无 VQA worker 判为未知任务。'
           '该次完整 cache/log/process/config 和 campaign failure 状态保留于 no_vqa_pretransport_failure_001/。'
           'CPU regression checks 通过后，仅本 phase 采用 module8c_runtime_compat 外部兼容层；原算法、evaluator、'
           'harness 文件、handlers、capability、quality、λ、计时、aggregation、bounds 均未改。'
           '内部 transport 仅省略无驻留 capability 且到达率/路由均为零的任务；search/cache/raw 仍保留原六维 λ。'
           '非零 unsupported 路由仍拒绝。runner AST 除模块入口外与原实现相同，前五组没有重跑。'
           '两次 premeasurement failures 单独计数，不隐藏、不替代已测数据；每个有效 state 仍恰为三次，'
           '每组有效 states≤6，confirmations=2+2。恢复 provenance 在 zero_rate_transport_recovery.json 冻结，'
           '每个 no-VQA raw 保存 compatibility source hash。\n')
    path.write_text(md)


def main():
    os.environ['LD_LIBRARY_PATH'] = '/root/miniconda3/envs/ocr_env/lib/python3.11/site-packages/nvidia/cu13/lib'
    with (OUT / '.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze = verify_freeze()
        verify_compatibility_freeze()
        assert all(read(OUT / name / 'status.json')['status'] == 'complete'
                   for name in freeze['execution_order'] if name != 'no_vqa_phase')
        original.ReplicaRuntimeEvaluator = CompatibleEvaluator
        confirmation_module.runner = runner
        save_json(OUT / 'campaign_status.json', {'status':'running', 'active_workload':'no_vqa_phase',
            'execution_order':freeze['execution_order'], 'technical_transport_recovery':True})
        try:
            original.run_workload('no_vqa_phase')
            finalize_report()
            save_json(OUT / 'campaign_status.json', {'status':'complete', 'stopped_after_module':'8C',
                'baseline':False, 'ablation':False, 'dynamic':False, 'second_hardware':False,
                'technical_transport_recovery':True})
            from module8c_final_real_search.report import artifact_manifest
            artifact_manifest()
            print('{"event":"module8c_complete","transport_recovery":true}', flush=True)
        except BaseException as exc:
            save_json(OUT / 'campaign_status.json', {'status':'blocked', 'reason':str(exc),
                'no_retry_or_budget_growth':True, 'traceback':traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
