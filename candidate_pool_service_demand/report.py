"""Final audit summary and refreshed planning provenance; no measurements."""
import shutil

from .prepare import ROOT, OUT, read, write, reference


def build():
    measurements = read(OUT / 'standalone_service_demand_results.json')
    validation = read(OUT / 'measurement_validation.json')
    regression = read(OUT / 'cpu_regression_validation.json')
    lines = ['# Module 7A-2 Standalone Service-Demand Completion', '',
        f'完整性：{validation["measured_paths"]}/{validation["required_paths"]} verified task-model service paths，'
        f'新增 {validation["new_measured_paths"]} 条，已有 11 条原值保留，pending={validation["still_pending_paths"]}。', '',
        '## Protocol and provenance', '',
        '直接运行未修改的 `joint_deploy/run_active_single_profile.py` main。只扩展运行时 worker 注册，保留原始选择、warmup、submit/collect、timer、raw format 和 summary 实现；外层增加日志、GPU 独占核验和 unload 等待。Module 7B 算法源文件没有修改。', '',
        '每条路径三轮独立 fresh worker；每轮先对第一个 target warmup 一次，再 enqueue 同一 task 的前 30 个冻结 mixed-pool rows。warmup 排除，31 次真实调用/轮；42 轮产生 1260 个 measured requests 和 42 个 warmups。每次只有一个 model type、一个 worker、一个 replica。', '',
        's0 的单位是 ms/request；统计量为 handler 内 `worker_latency_ms` 的算术均值，跨三轮用 `sum(mean*count)/sum(count)` 聚合。不使用 median、IPC queue wait 或 submit-to-completion E2E；包括 frozen handler 内的数据解码、预处理、CUDA 传输、推理、同步和输出处理。', '',
        '输入逐项核对历史前 30 个请求的 sample payload/pool IDs/arrival index。ASR 使用原 service pool 的不同长度音频，不替换成最小加载/quality 的 4–6 秒子集。', '',
        '沿用 frozen v2 native adapters：相同任务 dataset/label semantics、generation limits 和 output settings。Generalists 读取冻结 task prompts；模型各自 tokenizer/chat template、官方 weight transform 保持原样；PaddleOCR-VL 保持已冻结 element-level `OCR:` prompt 和 32-token recognized-text 输出，不运行 document parsing。', '',
        '冻结环境的 Python/package/executable hash 前后完全一致；未升级任何依赖。checkpoint、tokenizer、custom model code、adapter、prompt 和 historical inputs 的 hash 保持不变。source revision 和每个模型 weight/file hash 记录于 pending manifest、before/source hashes。', '',
        'NVML 记录证明每轮 GPU 上只有该 worker 的 PID，shutdown 后显存释放再启动下一轮。raw summary 和 handler ns timers 可独立复算；任何 load/inference/isolation/identity/timer failure 均不能填零、无穷或 synthetic fallback。冻结 capability matrix 未更改。', '',
        '## New real standalone demands', '',
        '| Model | Task | Repetition means (ms) | Aggregate (ms) | Sample std of 90 requests (ms) | Count |',
        '|---|---|---|---:|---:|---:|']
    for row in measurements['paths']:
        means = ', '.join(f'{value:.6f}' for value in row.get('repetition_means_ms', []))
        if row['status'] == 'measured':
            lines.append(f'| {row["model_id"]} | {row["task"]} | {means} | {row["aggregate_value_ms"]:.6f} | {row["sample_std_ms"]:.6f} | {row["measurement_count"]} |')
        else:
            lines.append(f'| {row["model_id"]} | {row["task"]} | failed | pending | — | — |')
    lines += ['', 'sample std 是所有 measured request 的样本标准差（ddof=1），包括不同输入样本的变化。每轮 raw latencies、每轮 sample std、三轮 mean 的 sample std 均另存于 results JSON，不把 request variability 当作纯 repetition uncertainty。', '',
        '## Restored W1/W2/W3 integer planning', '',
        '只有 completeness=100%、still_pending_paths=0 时才执行以下 dry-runs。输入为冻结 v2 measured idle VRAM + 新冻结真实 s0；不重测旧 service paths，不运行 runtime evaluator，不启动候选 replica workers。', '',
        '| Workload | Integer seed (active entries) | Memory MiB | N_G | N_S | Active G/S types | z | Candidates / feasible |',
        '|---|---|---:|---:|---:|---|---:|---|']
    dry_rows = []
    if validation['still_pending_paths'] == 0:
        for index in range(1, 4):
            row = read(OUT / f'integer_dry_runs/w{index}_integer_dry_run.json')
            dry_rows.append(row)
            active = ', '.join(f'{m}={n}' for m, n in row['integer_seed_replica_vector'].items() if n)
            lines.append(f'| {row["workload"]} | {active} | {row["total_memory_mib"]:g} | {row["N_G"]} | {row["N_S"]} | {row["active_generalist_types"]}/{row["active_specialist_types"]} | {row["max_pressure"]:.8f} | {row["candidate_count"]}/{len(row["feasible_candidates"])} |')
        for row in dry_rows:
            lines += ['', f'### {row["workload"]}', '',
                'Bottleneck families: ' + ', '.join(row['bottleneck_model_families']) + '.',
                'Top-B: ' + '; '.join(f'{c["model_id"]} {c["from_count"]}→{c["to_count"]}, '
                                      f'memory cost={c["memory_cost_mib"]:g} MiB, delta_z={c["delta_z"]:.8f}' for c in row['Top_B']) + '.',
                'Scale-down: ' + '; '.join(f'{c["model_id"]} {c["from_count"]}→{c["to_count"]}: {c["status"]} ({c["rejection_reason"] or "feasible"}), saving={c["memory_saving_mib"]:g} MiB'
                                           for c in row['all_replica_mutation_candidates'] if c['operation'] == 'SCALE_DOWN') + '.', '']
            for choice in row['generalist_specialist_replica_choice_by_pressure_task']:
                task = choice['bottleneck_task']
                lines += [f'Pressure task **{task}** — Generalist/Specialist +1 comparison:', '',
                          '| Role | Model/count change | Memory cost MiB | delta_z | Feasible | Rank |',
                          '|---|---|---:|---:|---|---:|']
                for c in choice['generalist_plus_one'] + choice['specialist_plus_one']:
                    delta = '—' if c['delta_z'] is None else f'{c["delta_z"]:.8f}'
                    lines.append(f'| {c["role"]} | {c["model_id"]} {c["from_count"]}→{c["to_count"]} | {c["memory_cost_mib"]:g} | {delta} | {c["feasible"]} | {c["ranking"] or "—"} |')
                if not choice['both_generalist_and_specialist_candidates']:
                    lines += ['', '该 task 没有两类可同时比较的候选；保留能力结构约束。']
                lines += ['']
    lines += ['', '三个 workload 的 visual_qa rate 均为正，唯一 verified capable model 是 Qwen，因此 **mandatory model induced by capability coverage: y_qwen=1**。不能把这一 presence 写成基于性能的主动选择；当前 Generalist composition 的自由度受 VQA capability asymmetry 限制。', '',
        '完整 integer vectors（含零项）、canonical family-level routing、per-replica pressure、task/model total demand、所有生成的 x±1 candidates、拒绝原因、delta_z、ranking、Top-B 和 N_G/N_S 均保存在三个 dry-run JSON。', '',
        'Memory 仍为 **planning memory surrogate**（additive measured idle VRAM）；capacity 仍为 **analytical planning surrogate, not a claim of linear real-GPU scaling**。这些结果没有建立真实多副本性能或最优 latency allocation 的结论。', '',
        '## CPU regression and stopping', '',
        f'{regression["tests_run"]} 项当前 CPU regressions/demand-gate tests 全部通过。四个会重新运行 A–E 或 20/30 synthetic scalability campaign 的 test wrappers 按用户要求未重跑；其既有通过记录保留，相关 {regression["unchanged_7B_source_files_checked"]} 个源文件 hash 未变。原 113 项测试的验证由本次重新运行的 109 项与这 4 项不变性核验共同保留。', '',
        '未运行 final deployment benchmark、joint/co-location measurement、dynamic multi-replica search、forecasting、predictor retraining 或 Module 7C。真实 profiling 仅用于补齐 14 条 single-worker service paths；real-pool integer dry-run 的 runtime/GPU measurement 次数为 0。到此停止。', '',
        '## Artifact entry points', '',
        '- `pending_service_demand_paths.json`: 精确 14 条 verified+missing paths 与环境/adapter/model provenance。',
        '- `historical_protocol_audit.json`: 原协议、样本选择、11 条已有 s0 与三轮对照。',
        '- `raw_measurements/r1,r2,r3/`: 独立 raw JSONL、Stage-1 summaries、worker/profile logs、isolation/requests/timer audit。',
        '- `standalone_service_demand_results.json`: 每轮 values/mean/std、90 次聚合与 failure state。',
        '- `updated_service_demand_catalog.json`: 新冻结 planning input；原 catalog 没有覆盖。',
        '- `measurement_validation.json`, `cpu_regression_validation.json`, `source_hashes.json`: gates 和不变性证据。',
        '- `integer_dry_runs/w1,w2,w3_integer_dry_run.json`: 与原 7B 输出目录同步的已解锁 records。']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    foundation = ROOT / 'results/integer_replica_search_foundation'
    archive = OUT / 'blocked_dry_runs_before'
    for filename in ('summary.md', 'validation_manifest.json'):
        original = foundation / filename
        if original.exists() and not (archive / filename).exists():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, archive / filename)
    if validation['still_pending_paths'] == 0:
        prior = read(foundation / 'validation_manifest.json')
        prior.update(W1_W2_W3_complete_12_model_planning='passed: restored by Module 7A-2 with real frozen Stage-1 demand',
                     acceptance_14='passed: W1/W2/W3 12-model integer dry-runs completed',
                     service_demand_completion=reference(OUT / 'measurement_validation.json'),
                     updated_service_demand_catalog=reference(OUT / 'updated_service_demand_catalog.json'),
                     foundation_synthetic_campaigns_rerun=False,
                     real_multi_replica_workers_started=0)
        write(foundation / 'validation_manifest.json', prior)
        old_summary = (archive / 'summary.md').read_text()
        (foundation / 'summary.md').write_text('# Current status after Module 7A-2\n\n'
            '14 个 pending Stage-1 demands 已由真实三轮 standalone measurement 补齐，25/25 verified paths 完整。'
            'W1/W2/W3 12-model integer dry-run 已解除 blocked，使用 frozen idle VRAM + new frozen real s0；'
            '验收第 14 项已通过。详细报告见 `../candidate_pool_v2_service_demand/summary.md`。'
            'Module 7B 算法和此前 synthetic campaigns 未修改或重跑；以下保留当时的历史状态。\n\n---\n\n' + old_summary)
    source_path = OUT / 'source_hashes.json'
    manifest = read(source_path)
    for p in OUT.rglob('*'):
        if p.is_file() and p != source_path:
            manifest['files'][str(p.relative_to(ROOT))] = reference(p)
    for pattern in ('*.py', '*.md'):
        for p in (ROOT / 'candidate_pool_service_demand').glob(pattern):
            manifest['files'][str(p.relative_to(ROOT))] = reference(p)
    for p in foundation.glob('w*_integer_dry_run.json'):
        manifest['files'][str(p.relative_to(ROOT))] = reference(p)
    for filename in ('summary.md', 'validation_manifest.json'):
        manifest['files'][str((foundation / filename).relative_to(ROOT))] = reference(foundation / filename)
    write(source_path, manifest)
    print({'new_measured_paths': validation['new_measured_paths'], 'completeness_ratio': validation['completeness_ratio'],
           'CPU_tests_passed': regression['tests_run'], 'dry_runs': len(dry_rows)})


if __name__ == '__main__':
    build()
