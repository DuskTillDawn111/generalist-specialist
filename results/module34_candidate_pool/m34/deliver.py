"""Export final review documents; full immutable run data remain in Module 34."""
from .common import *
import shutil
import zipfile


def main():
    assert read(OUT / 'continuation_closeout_status.json')['status'] == 'COMPLETED'
    assert read(OUT / 'final_audit.json')['passed']
    assert read(OUT / 'continuation_scientific_audit.json')['passed']
    # The orchestration status is finalized after the final audit; refresh only
    # the artifact inventory so its checksum covers the final status bytes.
    paths = [ref(p) for p in sorted(OUT.rglob('*')) if p.is_file()
             and 'work' not in p.relative_to(OUT).parts
             and p.name not in ['scientific_artifact_sha256.json', 'campaign.lock']
             and p.suffix != '.log']
    save('scientific_artifact_sha256.json', dict(files=paths, created_unix_ns=time.time_ns(),
         history=read(OUT / 'history_integrity_after.json'),
         scope='Final scientific artifacts after closeout; excludes mutable logs, caches, lock and this inventory itself.'))
    destination = Path('/root/Documents/Codex/2026-10-11/new-chat/outputs/module34')
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        'module34_final_answers.md', 'module34_summary.md', 'ranking_error_analysis.md',
        'module34_gate.json', 'final_audit.json', 'continuation_scientific_audit.json',
        'history_integrity_after.json', 'frozen_problem_manifest.json',
        'candidate_space_definition.json', 'frozen_gpu_manifest.json', 'frozen_gpu_manifest_v2.json',
        'statistical_analysis_plan.json', 'historical_secondary_analysis_plan.json',
        'software_environment_versions.json', 'CPU_oracle_correctness.json',
        'cpu_oracle_results.csv', 'optimizer_decisions.csv', 'predicted_regret.csv',
        'optimizer_runtime_benchmark.csv', 'optimizer_evaluation_count_audit.csv',
        'gpu_oracle_runs.csv', 'gpu_empirical_regret.csv', 'gpu_contrasts.json',
        'gpu_pool_expansion_pair.json', 'gpu_stability_audit.csv',
        'prediction_search_decomposition.csv', 'method_empirical_regret.csv',
        'per_task_prediction_diagnostics.csv', 'historical_large_vs_new_small_analysis.json',
        'decision_component_agreement.json', 'V3_original_acceptance_audit.json',
        'pool_expansion_audit.json', 'unhelpful_candidate_control.json',
        'module33_GPU_reuse.csv', 'module33_GPU_reuse_independent.csv',
        'infrastructure_preflight_events.json', 'continuation_record.json',
        'continuation_closeout_status.json', 'scientific_artifact_sha256.json',
        'postprocessing_events.json',
    ]
    names.extend(str(p.relative_to(OUT)) for p in sorted((OUT / 'figures').glob('*')) if p.is_file())
    exported = []
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT / name, target)
        assert ref(target)['sha256'] == ref(OUT / name)['sha256']
        exported.append(dict(relative_path=name, sha256=ref(target)['sha256'], bytes=target.stat().st_size))
    (destination / 'delivery_index.json').write_text(json.dumps(dict(
        scope='Review bundle: final reports, tables, frozen manifests, audits and figures. Full raw GPU runs, journals, CPU census, source code, weights and inputs remain at source_root; not a standalone runtime package.',
        source_root=str(OUT), files=exported), ensure_ascii=False, indent=2) + '\n')
    (destination / 'README.md').write_text(
        '# Module 34 交付文件\n\n'
        f"先阅读[六个研究问题的最终回答]({destination / 'module34_final_answers.md'})，"
        f"完整方法和结果见[总结报告]({destination / 'module34_summary.md'})。\n\n"
        '本轮九组 CPU 实验和 24 个原冻结 GPU 窗口已完成；历史文件 SHA256 与修改时间核验通过。'
        '论文结论须保留有限域、关键实测方案及统计不确定性的边界。\n\n'
        f'本目录包含报告、Gate、表格、审计与 PNG/PDF 图。全部原始 GPU 请求、终端日志、'
        f'CPU 候选全集及实现仍保存于 `{OUT}`。此包用于评审，不是独立运行环境。\n')
    archive = destination.parent / 'module34_review_bundle.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file():
                z.write(p, str(Path('module34') / p.relative_to(destination)))
    print('DELIVERED', str(destination), archive.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
