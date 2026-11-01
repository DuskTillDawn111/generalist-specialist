"""Resumable, isolated resource measurements for the seven retained models.

No quality thresholds, adapters or decoding are optimized here. Old experiments
are read only. The resource scope does not confer whole-pool admission.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/retained-replica-profiles-20261021'
QA = ROOT / 'artifacts/qa-development-revision-20261020/tydiqa-new-native-confirmation80-v2'
FRESH = ROOT / 'artifacts/remaining-task-confirmation-20261020'
OCR = ROOT / 'artifacts/ocr-small-candidate-screen-20261021'
GRANITE = ROOT / 'artifacts/chandra-replacement-20261020'
FACTOR = 1.110854
BUDGET = 49140


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def stratified(rows, n, key):
    """Input-only systematic quantiles, including both input-size extremes."""
    ordered = sorted(rows, key=lambda r: (key(r), r['id']))
    return [ordered[round(i * (len(ordered)-1)/(n-1))] for i in range(n)]


def prepare():
    OUT.mkdir(exist_ok=False)
    source_origins = {'qa': QA/'source', 'granite': GRANITE/'source'}
    origins = {}
    for name, origin in source_origins.items():
        dest = OUT / ('source-' + name)
        shutil.copytree(origin/'gsdeploy', dest/'gsdeploy',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        origins[name] = {str(p): sha(p) for p in origin.glob('gsdeploy/**/*.py')}
        runner = dest/'gsdeploy/profiling/runner.py'
        text = runner.read_text()
        text = text.replace('if spec.get("adapter") == "compact":',
            'if spec.get("adapter") == "retained_ppocrv6":\n'
            '            from .retained_ppocrv6 import Adapter\n'
            '        elif spec.get("adapter") == "compact":')
        # Isolate any native per-request files for every actual model process.
        text = text.replace('adapter = Adapter(spec, resources)',
            'resources = dict(resources)\n'
            '        resources["request_output"] = str(Path(resources["request_output"]) / str(os.getpid()))\n'
            '        adapter = Adapter(spec, resources)')
        runner.write_text(text)
        shutil.copy2(ROOT/'scripts/retained_ppocrv6_profile_adapter.py',
                     dest/'gsdeploy/profiling/retained_ppocrv6.py')

    def fresh(task, slot):
        return FRESH/'packets'/f'fresh-{task}-{slot}'/f'fresh-{task}-{slot}'/'job.json'

    originals = {task: read(fresh(task, 'generalist_1'))['samples'] for task in ['image', 'ocr', 'asr']}
    originals['text'] = read(QA/'qa80-generalist_1/job.json')['samples']
    from PIL import Image
    def pixels(r):
        with Image.open(r['input']) as im:
            return im.width * im.height
    keys = {'text': lambda r: len(r['context']) + len(r['question']),
            'image': pixels, 'ocr': pixels, 'asr': lambda r: r['duration_s']}
    cohorts = {}
    for task, rows in originals.items():
        n = 64 if task == 'image' else 32
        subset = stratified(rows, n, keys[task])
        cohorts[task] = [{k: v for k, v in r.items() if k not in
                         {'label', 'answers', 'answer', 'reference', 'references', 'is_impossible'}}
                         for r in subset]
        save(OUT/'cohorts'/(task+'.json'), cohorts[task])
    entries = []
    for slot in ['generalist_1', 'generalist_2', 'generalist_3', 'text', 'image', 'asr', 'ocr']:
        tasks = ['text', 'image', 'ocr', 'asr'] if slot.startswith('generalist') else [slot]
        for task in tasks:
            if slot == 'ocr':
                python = '/root/autodl-tmp/gsdeploy-replacement-assets/envs/chandra-native/bin/python'
                origin = None
                spec = dict(id='retained-ppocrv6', kind='compact', adapter='retained_ppocrv6',
                    candidate=dict(slot='ocr', model='PP-OCRv6-medium-det+rec', precision='float32'),
                    execution=dict(dtype='float32', allocator_conf='expandable_segments:True'),
                    resources=dict(checkpoint=str(OCR), inference_source=str(OCR/'run_ppocrv6_official.py'),
                                   inference_source_sha256=sha(OCR/'run_ppocrv6_official.py')))
                source = 'qa'
            else:
                if slot == 'asr':
                    origin = GRANITE/'packets/granite-full/granite-full/job.json'
                    source = 'granite'
                elif slot == 'image':
                    origin = fresh('image', 'image'); source = 'qa'
                elif task == 'text':
                    origin = QA/('qa80-'+slot)/'job.json'; source = 'qa'
                elif slot == 'generalist_1':
                    origin = ROOT/'artifacts/baichuan-no-synthesis-20261021'/task/'job.json'; source = 'qa'
                else:
                    origin = fresh(task, slot); source = 'qa'
                job = read(origin); python = job['python']; spec = copy.deepcopy(read(job['spec']))
                if slot == 'generalist_1':
                    spec['execution'] = read(ROOT/'configs/baichuan_no_synthesis_20261021.json')['execution']
            name = slot+'-'+task
            folder = OUT/'packets'/name
            spec['id'] = 'replica-profile-'+name
            spec['resources']['request_output'] = str(folder/'native-requests')
            save(folder/'worker.json', spec)
            # Largest inputs are warmed; the measured set always includes them.
            warmups = sorted(cohorts[task], key=keys[task], reverse=True)[:3]
            save(folder/'job.json', dict(spec=str(folder/'worker.json'), samples=cohorts[task],
                warmup_samples=warmups, rounds=3, minimum_seconds=30,
                minimum_requests=len(cohorts[task]), exact_cohort=False))
            grid = [1, 2, 3, 4, 6, 8] if slot == 'ocr' else [1, 2, 3, 4, 5]
            entries.append(dict(id=name, slot=slot, task=task, python=python,
                source=str(OUT/('source-'+source)), folder=str(folder), grid=grid,
                origin_job=str(origin) if origin else None))
    # Fast specialists first, then all retained generalists. Every profile releases the GPU.
    priority = {'image': 0, 'ocr': 1, 'asr': 2, 'text': 3,
                'generalist_1': 4, 'generalist_2': 5, 'generalist_3': 6}
    entries.sort(key=lambda e: (priority[e['slot']], ['image','text','asr','ocr'].index(e['task'])))
    bindings = {}
    for p in OUT.rglob('*'):
        if p.is_file(): bindings[str(p)] = sha(p)
    for e in entries:
        spec = read(Path(e['folder'])/'worker.json')
        for key in ['protocol', 'classmap', 'inference_source']:
            if key in spec['resources']:
                p = Path(spec['resources'][key]); bindings[str(p)] = sha(p)
    save(OUT/'release.json', dict(schema='gsdeploy.retained_replica_resource_plan.v1',
        frozen_utc=datetime.now(timezone.utc).isoformat(), entries=entries, bindings=bindings,
        original_sources=origins, controller_sha256=sha(__file__),
        budget_mib=BUDGET, safe_factor=FACTOR, pool_admitted=False,
        cohort_selection='Input-size systematic quantiles including extremes; image64, other tasks32. No output-based selection.',
        scope='Standalone same-model same-task closed loop; one outstanding request per process; no queueing SLO or mixed-model claim.',
        grid_limit='At most 5 replicas, or PP-OCR 8; this is a declared profile range, not proof of maximum feasible capacity.',
        stop_rules=['Any load, warmup, runtime or safety failure stops larger counts for that model/task.',
            'Projected (baseline + n * single-process warm residency) * safe_factor over budget excludes count conservatively; not observed OOM.',
            'Two consecutive increments with median successful-throughput gain below 5% stop larger counts as a throughput plateau.'],
        energy='Sampled NVML whole-board power integral; cumulative counter retained separately; includes idle board power.',
        quality='Resource replay only; no QA tuning, new quality gate, or admission claim. Predictions compared with 1 replica offline.'))
    print('Frozen', len(entries), 'model/task combinations', flush=True)


def measure(entry_id, replicas):
    release = read(OUT/'release.json')
    e = next(e for e in release['entries'] if e['id'] == entry_id)
    for p, digest in release['bindings'].items():
        assert sha(p) == digest, p
    sys.path.insert(0, e['source'])
    sys.path.append('/root/autodl-tmp/gsdeploy-assets/envs/profile-overlay')
    from gsdeploy.profiling.runner import run_profile
    job = read(Path(e['folder'])/'job.json'); spec = read(job['spec'])
    result = run_profile(spec, spec['resources'], job['samples'], replicas,
        OUT/'runs'/f'{entry_id}-n{replicas}', rounds=3, minimum_seconds=30,
        minimum_requests=job['minimum_requests'], warmup_samples=job['warmup_samples'], exact_cohort=False)
    raise SystemExit(0 if result['status'] == 'COMPLETE' else 1)


def throughput(profile):
    return statistics.median(r['successful_rps'] for r in profile['rounds'])


def run():
    release = read(OUT/'release.json')
    assert sha(__file__) == release['controller_sha256']
    state_path = OUT/'state.json'
    state = read(state_path) if state_path.exists() else dict(status='PREPARED', jobs=[], stopped={})
    state.update(status='RUNNING', pid=os.getpid(), pool_admitted=False)
    def update():
        state['utc'] = datetime.now(timezone.utc).isoformat(); save(state_path, state)
    with (ROOT/'artifacts/.gpu-experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for e in release['entries']:
            profiles = []
            for n in e['grid']:
                name = f'{e["id"]}-n{n}'
                path = OUT/'runs'/name/'profile.json'
                if path.exists():
                    p = read(path)
                    if p['status'] == 'COMPLETE': profiles.append(p); continue
                    if p['status'] in ['FAILED', 'INTERRUPTED']:
                        state['stopped'][e['id']] = dict(reason='MEASURED_FAILURE', replicas=n, evidence=str(path)); break
                    raise RuntimeError('Unfinished previous profile: '+name)
                if profiles:
                    single = profiles[0]
                    warm = single['startup_energy']['end_ns']
                    telemetry = read(OUT/'runs'/f'{e["id"]}-n1'/'telemetry.json')
                    snapshot = min(telemetry, key=lambda r: abs(r['time_ns']-warm))
                    process = sum(p['used_mib'] for p in snapshot['processes'])
                    baseline = snapshot['used_mib']-process
                    projected = (baseline + n*process)*FACTOR
                    if projected > BUDGET:
                        state['stopped'][e['id']] = dict(reason='DERIVED_WARM_RESIDENCY_EXCLUSION_NOT_OOM',
                            replicas=n, projected_safe_mib=projected, budget_mib=BUDGET,
                            single_warm_process_mib=process, baseline_board_mib=baseline)
                        update(); break
                    if len(profiles) >= 3 and all(throughput(profiles[i])/throughput(profiles[i-1])-1 < .05
                                                 for i in [-1, -2]):
                        state['stopped'][e['id']] = dict(reason='TWO_INCREMENT_THROUGHPUT_PLATEAU',
                            next_untested_replicas=n, tested=[p['replicas'] for p in profiles])
                        update(); break
                state['current'] = name; update()
                print('Starting', name, flush=True)
                env = dict(os.environ, PYTHONPATH=e['source'], OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0',
                    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                env.pop('DS_ACCELERATOR', None)
                log_path = OUT/(name+'.log')
                with log_path.open('x') as log:
                    child = subprocess.Popen([e['python'], '-u', str(Path(__file__).resolve()), 'measure',
                        '--entry', e['id'], '--replicas', str(n)], cwd=ROOT, env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid; update()
                    code = child.wait()
                p = read(path) if path.exists() else dict(status='FAILED', error=f'No profile; exit {code}')
                state['jobs'].append(dict(id=name, status=p['status'], path=str(path), exit_code=code)); update()
                summarize()
                if p['status'] != 'COMPLETE':
                    state['stopped'][e['id']] = dict(reason='MEASURED_FAILURE', replicas=n, evidence=str(path))
                    update(); break
                if any(r['failed_requests'] for r in p['rounds']):
                    state['stopped'][e['id']] = dict(reason='INVALID_COMPLETIONS', replicas=n, evidence=str(path))
                    update(); break
                profiles.append(p)
            else:
                state['stopped'][e['id']] = dict(reason='DECLARED_GRID_COMPLETED', tested=[p['replicas'] for p in profiles])
            update()
    state.update(status='COMPLETE_RESOURCE_GRID', current=None, finished_utc=datetime.now(timezone.utc).isoformat())
    update(); summarize()


def summarize():
    release = read(OUT/'release.json'); rows=[]
    for e in release['entries']:
        first = None
        for n in e['grid']:
            path=OUT/'runs'/f'{e["id"]}-n{n}'/'profile.json'
            if not path.exists(): continue
            p=read(path)
            row=dict(id=e['id'], replicas=n, status=p['status'], profile=str(path), error=p.get('error'))
            if p['status'] == 'COMPLETE':
                if first is None: first=p
                rounds=p['rounds']; wall=sum(r['wall_seconds'] for r in rounds)
                valid=sum(r['successful_requests'] for r in rounds); total=sum(r['completed_requests'] for r in rounds)
                telemetry=read(path.parent/'telemetry.json')
                begin=min(r['board_energy']['start_ns'] for r in rounds)
                end=max(r['board_energy']['end_ns'] for r in rounds)
                active=[v for v in telemetry if begin<=v['time_ns']<=end]
                seen={}
                workers=set()
                for raw in path.parent.glob(e['task']+'-round*.jsonl'):
                    for line in raw.open():
                        v=json.loads(line); workers.add(v['replica'])
                        pred={k:v.get(k) for k in ['raw_output','prediction','success','finish_reason']}
                        seen.setdefault(v['sample_id'], set()).add(json.dumps(pred,sort_keys=True,ensure_ascii=False))
                assert workers==set(range(n)), (e['id'],n,workers)
                assert set(seen)=={v['id'] for v in read(Path(e['folder'])/'job.json')['samples']}
                row.update(valid=valid, requests=total, median_rps=throughput(p),
                    speedup=throughput(p)/throughput(first), aggregate_rps=valid/wall,
                    mean_service_ms=statistics.mean(r['service_mean_ms'] for r in rounds),
                    p95_service_ms_by_round=[r['service_p95_ms'] for r in rounds],
                    board_resident_GiB=p['resident_after_warmup_mib']/1024,
                    board_peak_including_load_GiB=p['peak_board_used_mib']/1024,
                    board_measurement_peak_GiB=max(v['used_mib'] for v in active)/1024,
                    simultaneous_process_peak_sum_GiB=max(sum(q['used_mib'] for q in v['processes']) for v in active)/1024,
                    board_joules_per_request=sum(r['board_energy']['joules'] for r in rounds)/total,
                    energy_coverage_valid=all(r['board_energy']['coverage_valid'] for r in rounds),
                    within_safety_budget=p['peak_board_used_mib']*FACTOR<=BUDGET,
                    unique_inputs=len(seen), all_replicas_used=True,
                    samples_with_nonidentical_repeated_outputs=sum(len(v)>1 for v in seen.values()),
                    outputs_sha256=hashlib.sha256(json.dumps({k:sorted(v) for k,v in sorted(seen.items())},sort_keys=True).encode()).hexdigest())
            rows.append(row)
    save(OUT/'summary.json',dict(scope=release['scope'], pool_admitted=False, rows=rows))
    lines=['# 保留模型多副本资源画像','',release['scope'],'',
        '|模型/任务|副本|状态|有效/总数|中位吞吐 req/s|加速比|均延迟 ms|整卡测量峰 GiB|整卡 J/req|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['status']=='COMPLETE':
            lines.append(f'|{r["id"]}|{r["replicas"]}|完成|{r["valid"]}/{r["requests"]}|{r["median_rps"]:.3f}|{r["speedup"]:.2f}|{r["mean_service_ms"]:.1f}|{r["board_measurement_peak_GiB"]:.2f}|{r["board_joules_per_request"]:.1f}|')
        else: lines.append(f'|{r["id"]}|{r["replicas"]}|{r["status"]}|||||||')
    lines += ['', '固定输入子集：图像64，其余任务32，按输入大小分层并含极值；每档三轮、每轮至少30秒并覆盖全部输入。',
        '延迟为服务时间，无外部排队；能耗为整卡采样功率积分，包含空闲功耗。加载峰与测量峰在 JSON 分开。',
        '显存安全系数1.110854，预算49140 MiB。线性驻留排除是保守规划推导，不是实际 OOM。',
        '这些画像不证明完整数据集的最坏峰值、开环 SLO、混合模型共驻留或整个模型池准入。',
        'PP-OCR最多8副本，其余最多5副本；提前停止原因见 state.json。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare','run','measure','summarize'])
    parser.add_argument('--entry');parser.add_argument('--replicas',type=int)
    args=parser.parse_args()
    if args.mode=='measure': measure(args.entry,args.replicas)
    else: globals()[args.mode]()
