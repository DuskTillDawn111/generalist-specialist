"""Stage 2B-1: run and aggregate five explicit routing matrices, three seeds each."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

from joint_deploy.measurement.load_generator import periodic_arrivals
from joint_deploy.measurement.router import Router
from joint_deploy.measurement.run_measurement import DEFAULT_POOL, DEFAULT_PROFILE, ROOT, sha256
from joint_deploy.measurement.schema import MeasurementConfig

DEPLOYMENT = dict.fromkeys(('qwen', 'distilbert', 'resnet50', 'paddleocr', 'whisper'), 1)
WORKLOAD = dict(text=2.0, image=1.0, ocr=1.0, asr=0.3, visual_qa=0.5, lid=1.0)
SPECIALISTS = dict(text='distilbert', image='resnet50', ocr='paddleocr', asr='whisper', lid='whisper')
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (101, 202, 303)
OUT = ROOT / 'results/stage2b/routing_characterization'


def make_config(share, seed, min_requests=30):
    if share not in SHARES:
        raise ValueError('Unsupported characterization share')
    duration = math.ceil(min_requests / min(v for v in WORKLOAD.values() if v > 0))
    routing = {task: {model: 0.0 for model in DEPLOYMENT} for task in WORKLOAD}
    for task, rate in WORKLOAD.items():
        specialist = SPECIALISTS.get(task)
        routing[task]['qwen'] = (1 - share) * rate if specialist else rate
        if specialist:
            routing[task][specialist] = share * rate
    cfg = MeasurementConfig(dict(DEPLOYMENT), dict(WORKLOAD), routing,
                            warmup_seconds=10, measurement_seconds=duration, random_seed=seed,
                            routing_source='explicit_characterization_matrix')
    meta = json.loads(DEFAULT_PROFILE.read_text())
    cfg.validate({m: d['supported_tasks'] for m, d in meta['models'].items()})
    counts = Counter(t for _, t in periodic_arrivals(WORKLOAD, duration, seed))
    if min(counts.values()) < min_requests:
        raise ValueError('Window does not supply minimum request count')
    return cfg


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def protocol_hashes():
    paths = list((ROOT / 'joint_deploy/measurement').glob('*.py'))
    paths += [ROOT / 'joint_deploy/worker_protocol.py', DEFAULT_POOL, DEFAULT_PROFILE]
    paths += [ROOT / 'joint_deploy' / name for name in
              ('qwen_worker.py', 'distilbert_worker.py', 'resnet_worker.py', 'ocr_worker.py', 'whisper_worker.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def prepare(output):
    plan_path = output / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan['protocol_hashes'] != protocol_hashes():
            raise ValueError('Frozen measurement protocol/data changed; use a new experiment directory')
        return plan
    runs = []
    for repetition, seed in enumerate(SEEDS, 1):
        shares = list(SHARES)
        random.Random(20260929 + repetition).shuffle(shares)
        for share in shares:
            run_id = f'share_{share:.2f}_rep_{repetition}'
            cfg = make_config(share, seed)
            config_path = output / 'configs' / f'{run_id}.json'
            write_json(config_path, cfg.as_dict())
            runs.append(dict(run_id=run_id, repetition=repetition, seed=seed, specialist_share=share,
                             config=str(config_path.resolve()), result=str((output/'raw'/f'{run_id}.json').resolve()),
                             log=str((output/'logs'/f'{run_id}.log').resolve()), config_hash=sha256(config_path)))
    plan = dict(schema_version=1, deployment=DEPLOYMENT, workload_rps=WORKLOAD,
                min_measured_requests_per_active_task=30, seeds=list(SEEDS), shares=list(SHARES),
                ordering='seeded shuffled shares in each repetition block; fresh processes for every run',
                pairing='same seed across shares in each repetition; independent seeds between repetitions',
                protocol_hashes=protocol_hashes(), runs=runs)
    write_json(plan_path, plan)
    return plan


def audit(data, cfg, minimum=30):
    """Audit actual records, independently of summary counters. Never remove unstable data."""
    issues = []
    for field in ('deployment', 'workload_rps', 'routing_rps', 'random_seed', 'measurement_seconds',
                  'warmup_seconds', 'routing_source'):
        if data.get(field) != cfg[field]:
            issues.append(f'config mismatch: {field}')
    records = data.get('request_records', [])
    window = data.get('measurement_window')
    counts = Counter(row['task'] for row in records)
    observed = {task: Counter(row['model'] for row in records if row['task'] == task) for task in WORKLOAD}
    router = Router(cfg['routing_rps'])
    arrivals = list(periodic_arrivals(cfg['workload_rps'], cfg['measurement_seconds'], cfg['random_seed']))
    if len(records) != len(arrivals) or not window:
        issues.append('measurement window/arrival count missing or incomplete')
    else:
        for row, (offset, task) in zip(records, arrivals):
            expected_model = router.choose(task)
            expected_time = window['start_time_ns'] + round(offset * 1e9)
            if row['task'] != task or row['model'] != expected_model or row['scheduled_time_ns'] != expected_time:
                issues.append(f'arrival/routing mismatch: {row["request_id"]}')
            if row['status'] == 'completed':
                if not row['scheduled_time_ns'] <= row['actual_submit_time_ns'] <= row['completion_time_ns']:
                    issues.append(f'timestamp order: {row["request_id"]}')
    for task, rate in cfg['workload_rps'].items():
        if counts[task] < minimum:
            issues.append(f'insufficient offered samples: {task}={counts[task]} < {minimum}')
        for model, route in cfg['routing_rps'][task].items():
            target = counts[task] * route / rate
            if abs(observed[task][model] - target) > 1.000001:
                issues.append(f'routing count outside one-request rounding allowance: {task}/{model}')
        metric = data.get('per_task_metrics', {}).get(task, {})
        if metric.get('offered_requests') != counts[task]:
            issues.append(f'counter mismatch: {task}')
        actual_completed = sum(r['status'] == 'completed' for r in records if r['task'] == task)
        if metric.get('completed_requests') != actual_completed:
            issues.append(f'completion counter mismatch: {task}')
    clean = set(data.get('shutdown', {})) == set(DEPLOYMENT) and all(
        s['exit_code'] == 0 and s['bye_received'] and not s['forced'] for s in data['shutdown'].values())
    errors = [r['request_id'] for r in records if r['status'] == 'error']
    incomplete = [r['request_id'] for r in records if r['status'] not in ('completed', 'error')]
    completed = sum(r['status'] == 'completed' for r in records)
    ratio = completed / len(records) if records else None
    unstable = ratio is None or ratio < 1 or bool(errors) or bool(data.get('errors')) or not clean
    return dict(integrity_passed=not issues, issues=issues, clean_shutdown=clean,
                completion_ratio=ratio, worker_error_request_ids=errors, incomplete_request_ids=incomplete,
                overloaded_or_unstable=unstable, actual_offered_counts=dict(counts),
                actual_routing_counts={k:dict(v) for k,v in observed.items()},
                actual_completed_requests=completed)


def validate_smoke(output):
    path = output / 'full_deployment_smoke.json'
    data = json.loads(path.read_text())
    cfg = json.loads((output / 'configs/full_deployment_smoke.json').read_text())
    checked = audit(data, cfg, minimum=1)
    exercised = {r['model'] for r in data['request_records'] if r['status'] == 'completed'}
    checked['all_five_workers_exercised'] = exercised == set(DEPLOYMENT)
    checked['passed'] = (data['status'] == 'completed' and checked['integrity_passed']
                         and not checked['overloaded_or_unstable'] and checked['all_five_workers_exercised']
                         and data['deployment'] == DEPLOYMENT and data['workload_rps'] == WORKLOAD)
    write_json(output / 'full_deployment_smoke_validation.json', checked)
    if not checked['passed']:
        raise RuntimeError('Full deployment smoke failed; characterization is blocked. See validation JSON.')
    return checked


def stats(values):
    usable = [float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(usable), missing=len(values)-len(usable), values=usable,
                mean=statistics.mean(usable) if usable else None,
                std=statistics.stdev(usable) if len(usable) > 1 else None,
                min=min(usable) if usable else None, max=max(usable) if usable else None)


def aggregate(output, plan):
    runs, settings = [], []
    for spec in plan['runs']:
        path = Path(spec['result'])
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        cfg = json.loads(Path(spec['config']).read_text())
        checked = audit(data, cfg)
        for task, metrics in data.get('per_task_metrics', {}).items():
            rows = [r for r in data['request_records'] if r['task'] == task and r['status'] == 'completed']
            residual = [r['runtime_latency_ms'] - r['worker_latency_ms'] for r in rows
                        if r.get('worker_latency_ms') is not None]
            metrics['mean_runtime_minus_worker_ms'] = statistics.mean(residual) if residual else None
        runs.append({**spec, 'audit': checked, 'status': data['status'],
                     'system_metrics': data['system_metrics'], 'per_task_metrics': data['per_task_metrics']})
    for share in SHARES:
        selected = [r for r in runs if r['specialist_share'] == share]
        system_keys = set().union(*(set((r['system_metrics'] or {}).keys()) for r in selected))
        system = {key: stats([(r['system_metrics'] or {}).get(key) for r in selected]) for key in sorted(system_keys)}
        per_task = {}
        for task in WORKLOAD:
            metrics = [r['per_task_metrics'].get(task, {}) for r in selected]
            keys = {k for row in metrics for k, v in row.items() if isinstance(v, (int, float)) or v is None}
            per_task[task] = {k: stats([row.get(k) for row in metrics]) for k in sorted(keys)}
        settings.append(dict(specialist_share=share, routing_rps=make_config(share, SEEDS[0]).routing_rps,
                             repetitions=len(selected), system_metrics=system, per_task_metrics=per_task,
                             overloaded_or_unstable_runs=[r['run_id'] for r in selected if r['audit']['overloaded_or_unstable']]))
    summary = dict(schema_version=1, experiment='Stage 2B-1 Routing Characterization',
                   deployment=DEPLOYMENT, workload_rps=WORKLOAD, num_runs=len(runs), expected_runs=15,
                   integrity_passed=all(r['audit']['integrity_passed'] for r in runs),
                   overloaded_or_unstable=any(r['audit']['overloaded_or_unstable'] for r in runs),
                   aggregation='mean/sample standard deviation (ddof=1)/min/max across independent run-level metrics; '
                               'P95/P99 are averaged run quantiles, not pooled-request quantiles; no runs discarded',
                   settings=settings, runs=runs)
    write_json(output/'routing_characterization_summary.json', summary)
    write_markdown(output, summary)
    return summary


def write_markdown(output, summary):
    def fmt(block):
        if not block or block['mean'] is None:
            return '—'
        return f"{block['mean']:.3f} ± {block['std']:.3f}" if block['std'] is not None else f"{block['mean']:.3f} (n=1)"
    lines = ['# Stage 2B-1 Routing Characterization', '',
             f"已保存 {summary['num_runs']}/15 轮。固定五模型 deployment；workload 总负载 5.8 req/s。",
             '每轮 measurement=100 s，warmup=10 s，另有逐路径 priming；每轮重新启动全部 worker。',
             '三组 seed=101/202/303；同一重复内各 share 采用相同 seed，运行顺序已打乱并保存在 plan.json。',
             '表格为三个 run-level 指标的 mean ± sample std，P95/P99 不做请求池混合。', '',
             '## System measurements', '',
             '| specialist_share | reps | cohort completion rps | within-window rps | completion ratio | unstable runs |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    for setting in summary['settings']:
        m = setting['system_metrics']
        lines.append(f"| {setting['specialist_share']:.2f} | {setting['repetitions']} | "
                     f"{fmt(m.get('achieved_total_throughput_rps'))} | {fmt(m.get('within_window_completion_throughput_rps'))} | "
                     f"{fmt(m.get('completion_ratio'))} | {', '.join(setting['overloaded_or_unstable_runs']) or 'none'} |")
    for setting in summary['settings']:
        lines += ['', f"## Routing matrix: specialist_share={setting['specialist_share']:.2f}", '',
                  '| task | qwen | distilbert | resnet50 | paddleocr | whisper |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        for task, route in setting['routing_rps'].items():
            lines.append('| ' + task + ' | ' + ' | '.join(f'{route[m]:g}' for m in DEPLOYMENT) + ' |')
        lines += ['', '单位 req/s；specialist_share 仅为生成矩阵的实验标签。', '',
                  '| task | E2E mean ms | E2E P50 ms | E2E P95 ms | E2E P99 ms | worker mean ms | schedule mean ms | runtime−worker mean ms |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for task, metrics in setting['per_task_metrics'].items():
            keys = ('mean_e2e_latency_ms', 'p50_e2e_latency_ms', 'p95_e2e_latency_ms', 'p99_e2e_latency_ms',
                    'mean_worker_latency_ms', 'mean_schedule_delay_ms', 'mean_runtime_minus_worker_ms')
            lines.append('| '+task+' | '+' | '.join(fmt(metrics.get(k)) for k in keys)+' |')
    lines += ['', '## Scope and interpretation', '',
              '- 全部 raw result 保存在 raw/，独立 configuration 在 configs/，启动/执行日志在 logs/。',
              '- summary JSON 保留每个指标 mean/std/min/max/values，以及每轮验证、实际 offered/routing counts。',
              '- 完成吞吐按 Stage 2A 的 cohort completions / 100 s；另展示窗口内完成吞吐，避免 drain 掩盖积压。',
              '- worker latency 是 handler 墙钟时间；runtime−worker 包含 worker queue、传输和返回开销，不等于纯 queue time。',
              '- ASR 每轮只有 30 个样本，P95/P99 对少数请求敏感；三轮 std 是重复间离散程度，不是置信区间。',
              '- 出现不完整点时保留并标记 overloaded_or_unstable，不自动归因、删除或用成功重跑替换。',
              '- 本实验为固定 deployment/workload 的描述性测量，不给出最优 routing 或性能预测。', '']
    (output/'routing_characterization_summary.md').write_text('\n'.join(lines), encoding='utf-8')


def execute(output, plan):
    validate_smoke(output)
    output.joinpath('raw').mkdir(exist_ok=True)
    output.joinpath('logs').mkdir(exist_ok=True)
    for index, spec in enumerate(plan['runs'], 1):
        if sha256(spec['config']) != spec['config_hash']:
            raise ValueError(f"Configuration changed: {spec['run_id']}")
        if Path(spec['result']).exists():
            print(f"[{index}/15] preserving existing {spec['run_id']}", flush=True)
            continue
        print(f"[{index}/15] START {spec['run_id']} seed={spec['seed']}", flush=True)
        command = [sys.executable, '-m', 'joint_deploy.measurement.run_measurement', '--config', spec['config'],
                   '--output', spec['result'], '--pool', str(DEFAULT_POOL), '--profile', str(DEFAULT_PROFILE)]
        started = time.time()
        with open(spec['log'], 'w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS':'1'},
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                write_json(output/'logs'/f"{spec['run_id']}.timeout.json", dict(error='run timeout', run=spec))
                raise RuntimeError('Run timed out; raw logs preserved. Inspect before resuming.')
        if not Path(spec['result']).exists():
            raise RuntimeError(f"No result artifact for {spec['run_id']}, exit={code}; see log")
        summary = aggregate(output, plan)
        report = next(r for r in summary['runs'] if r['run_id'] == spec['run_id'])
        print(f"[{index}/15] END {spec['run_id']} exit={code} seconds={time.time()-started:.1f} "
              f"completion={report['audit']['completion_ratio']} unstable={report['audit']['overloaded_or_unstable']}", flush=True)
    return aggregate(output, plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--execute', action='store_true', help='run the prepared 15 real GPU measurements after smoke gate')
    args = parser.parse_args(argv)
    plan = prepare(args.output_dir.resolve())
    summary = execute(args.output_dir.resolve(), plan) if args.execute else aggregate(args.output_dir.resolve(), plan)
    print(json.dumps({k:summary[k] for k in ('num_runs','expected_runs','integrity_passed','overloaded_or_unstable')}, indent=2))
    return 0 if not args.execute or (summary['num_runs']==15 and summary['integrity_passed']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
