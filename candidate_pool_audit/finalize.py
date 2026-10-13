import json
import math
from pathlib import Path
import struct
from dataclasses import asdict
from datetime import datetime, timezone, timedelta

from .inventory import ROOT, SOURCES, EXISTING, NEW, TASKS, selected_weights
from .schema import CandidateModel
from .provenance import sha
from .freeze import evaluate

OUT = ROOT / 'results/candidate_pool_v2'


def read(path):
    return json.loads(path.read_text())


def revision(directory, meta, new):
    if directory:
        path = ROOT / 'models' / directory / '.cache/huggingface/download/config.json.metadata'
        if path.exists():
            return path.read_text().splitlines()[0]
    return meta.get('sha') if new or directory is None else None


def tensor_count(weights):
    value = 0
    for path in weights:
        if path.suffix == '.safetensors':
            with path.open('rb') as stream:
                size = struct.unpack('<Q', stream.read(8))[0]
                data = json.loads(stream.read(size))
            value += sum(math.prod(t['shape']) for key, t in data.items() if key != '__metadata__')
    return value or None


def historical_evidence(name, task):
    prefix = {'phi4mm': 'phi4', 'whisper-small': 'whisper_small'}.get(name, name)
    suffix = {'text': 'sst2', 'image': 'imagenette', 'ocr': 'iiit5k', 'asr': 'librispeech', 'visual_qa': 'scienceqa', 'lid': 'voxlingua'}[task]
    if name == 'whisper-small' and task == 'lid':
        prefix = 'whisper'
    path = ROOT / f'benchmark/results/formal/{prefix}_{suffix}_summary.json'
    if not path.is_file():
        raise ValueError(f'Missing historical task verification: {path}')
    result = read(path)
    expected_dataset = {'text': 'sst2', 'image': 'imagenette', 'ocr': 'iiit5k', 'asr': 'librispeech',
                        'visual_qa': 'scienceqa', 'lid': 'voxlingua107'}[task]
    actual_dataset = str(result.get('dataset', '')).lower().replace('-', '')
    expected_tasks = {'text': ['text_classification'], 'image': ['image_classification'],
                      'ocr': ['ocr', 'optical_character_recognition'],
                      'asr': ['asr', 'automatic_speech_recognition']}.get(task, [task])
    if actual_dataset != expected_dataset or result.get('task') not in expected_tasks or result.get('num_samples', 0) <= 0:
        raise ValueError(f'Historical evidence does not match task semantics: {path}')
    return {'source': str(path), 'sha256': sha(path), 'type': 'historical actual inference on current task/dataset; no benchmark rerun'}


def main():
    preflight = read(OUT / 'adapter_preflight.json')
    tests = {p.stem: read(p) for p in (OUT / 'raw_load_tests').glob('*.json')}
    models, matrix, metadata_notes = [], {}, {}
    launchers = read(OUT / 'environment_manifest.json')['environment_by_candidate']
    for name, (source, directory, family, role, precision) in SOURCES.items():
        meta_path = OUT / 'sources' / (source.replace('/', '--') + '.api.json')
        meta = read(meta_path) if meta_path.exists() else {}
        weights = selected_weights(directory)
        record = tests.get(name, {})
        passed = record.get('status') == 'passed' and record.get('gpu_memory_released') is True
        supported = EXISTING.get(name, NEW.get(name, []))
        pending = list(TASKS) if name == 'gemma3n' else ['ocr'] if name == 'got_ocr' else ['visual_qa', 'lid'] if name in ('phi4mm', 'minicpm') else []
        verified = []
        cells = {}
        for task in TASKS:
            evidence = []
            if task in supported:
                if name in EXISTING:
                    evidence.append(historical_evidence(name, task))
                for sample in record.get('sanity_samples', []):
                    if sample['task'] == task:
                        evidence.append({'source': str(OUT / f'raw_load_tests/{name}.json'), 'sample_id': sample['sample_id'],
                                         'type': 'local CUDA task adapter sanity inference',
                                         'output_schema_valid': sample['output_schema_valid'], 'correct': sample['response'].get('correct')})
                status = 'verified' if passed and evidence else 'pending'
                if status == 'verified' and task not in record['ready']['tasks']:
                    raise ValueError('Capability missing from actual worker READY: ' + name + ':' + task)
                if name == 'whisper_turbo' and task == 'lid' and status == 'verified':
                    lid_samples = [s for s in record['sanity_samples'] if s['task'] == 'lid']
                    current_codes = set(read(OUT / 'task_definitions.json')['lid']['label_space'])
                    if len(lid_samples) != 2 or not all(s['response']['correct'] for s in lid_samples):
                        raise ValueError('Insufficient independent LID sanity evidence')
                    if not all(current_codes <= {k[2:-2] for k in s['response']['language_token_map']} for s in lid_samples):
                        raise ValueError('Whisper Turbo language tokens do not cover current LID label set')
                if status == 'verified':
                    verified.append(task)
                elif task not in pending:
                    pending.append(task)
                reason = preflight[name]['reason']
            elif task in pending:
                status = 'pending'
                reason = preflight[name]['reason'] if name in ('gemma3n', 'got_ocr') else 'No current service-task adapter verification; input modality support is insufficient.'
            else:
                status = 'unsupported'
                reason = preflight[name]['reason'] if name == 'siebert_rejected' else 'No matching task-specific adapter in audited candidate scope.'
                if name == 'distil_whisper' and task == 'lid':
                    reason = 'English ASR candidate; no reliable adapter verified for current multilingual LID task.'
            cells[task] = {'status': status, 'capability': status == 'verified', 'reason': reason, 'evidence': evidence}
        matrix[name] = cells
        adapter_status = 'verified' if verified else 'unsupported' if name == 'siebert_rejected' else 'pending'
        weight_size = sum(p.stat().st_size for p in weights) / 1e9 if weights else None
        if weight_size is None and meta:
            remote = [x for x in meta.get('siblings', []) if x['rfilename'].endswith('.safetensors') and '.fp32.' not in x['rfilename']]
            if not remote:
                remote = [x for x in meta.get('siblings', []) if x['rfilename'] == 'pytorch_model.bin']
            if remote and all(x.get('size') is not None for x in remote):
                weight_size = sum(x['size'] for x in remote) / 1e9
        params = record.get('ready', {}).get('parameter_count') or tensor_count(weights) or (meta.get('safetensors') or {}).get('total')
        if name == 'resnet50':
            params = 25557032
        source_revision = revision(directory, meta, name in NEW)
        if name in ('resnet50', 'convnext_tiny') and weights:
            source_revision = 'sha256:' + sha(weights[0])
        license = (meta.get('cardData') or {}).get('license')
        if license == 'other':
            license = (meta.get('cardData') or {}).get('license_name') or 'other'
        if name in ('resnet50', 'convnext_tiny'):
            license = 'BSD-3-Clause (torchvision code); ImageNet dataset/weight use terms separate'
        if name == 'paddleocr':
            license = 'apache-2.0'
        item = CandidateModel(name, family, role, verified, sorted(set(pending)), source, source_revision,
                              float(params) if params is not None else None, weight_size,
                              record.get('measured_idle_vram_mib') if passed else None,
                              'paddleocr.TextRecognition CUDA process' if name == 'paddleocr' else
                              'torchvision CUDA process' if name in ('resnet50', 'convnext_tiny') else 'transformers PyTorch CUDA process',
                              precision, adapter_status, license)
        models.append(dict(asdict(item), task_compatibility={task: cells[task]['status'] for task in TASKS}))
        metadata_notes[name] = {'parameter_count_basis': 'Runtime model.parameters() at READY' if record.get('ready', {}).get('parameter_count') else
                                'selected safetensors stored tensor elements (includes repository modules/buffers, not a claim that all are active)' if tensor_count(weights) else
                                'official metadata' if params is not None else 'unknown; not invented',
                                'selected_weight_files': [str(p) for p in weights],
                                'weight_size_basis': 'selected local inference weight files, decimal GB, excludes alternate duplicate serialization' if weights else 'official repository primary weight files, decimal GB; no VRAM inference',
                                'local_directory': str(ROOT / 'models' / directory) if directory else None,
                                'environment_python': launchers.get(name),
                                'official_source_url': 'https://huggingface.co/' + source if '/' in source else 'https://docs.pytorch.org/vision/stable/models.html',
                                'runtime_scope': 'audio generation disabled' if name == 'qwen' else 'init_tts=False' if name == 'minicpm' else 'frozen adapter or candidate adapter',
                                'source_revision_basis': 'local download metadata or pinned new download; missing revision stays null',
                                'license_metadata_basis': 'Official publisher HF cardData; license_name resolves other; license may be null if publisher provides none.',
                                'load_status': record.get('status', 'not_run')}
        if name == 'paddleocr_vl':
            metadata_notes[name]['backend_warning'] = 'Transformers native backend reports repository RoPE extra mrope_section key and untied embedding warning; actual backend loaded and all three plain OCR sanity calls passed. Unique runtime parameter count differs from stored tensor count; both bases are recorded. Formal quality remains unmeasured.'
            metadata_notes[name]['stored_tensor_element_count'] = tensor_count(weights)
    integrity = read(OUT / 'download_integrity.json')
    environments = read(OUT / 'environment_manifest.json')
    source_hashes = read(OUT / 'source_hashes.json')
    for check in integrity:
        relative_path = str(Path(check['file']).relative_to(ROOT))
        if source_hashes['files'][relative_path]['sha256'] != check['expected_sha256']:
            raise ValueError('Downloaded checkpoint hash drift: ' + relative_path)
    integrity_ok = all(x['verified'] for x in integrity) and environments['unchanged_from_before'] and source_hashes['algorithm_and_baseline_files_unchanged']
    summary = evaluate(models, tests, matrix, integrity_ok)
    summary['audited_candidate_records'] = len(models)
    summary['excluded_or_pending_ids'] = sorted(set(SOURCES) - set(summary['formal_model_ids']))
    summary['provisional_generalist_status'] = {name: {'idle_vram_mib': next(m['measured_idle_vram_mib'] for m in models if m['id'] == name),
                                                     'status': next(m['adapter_status'] for m in models if m['id'] == name)} for name in ('qwen', 'phi4mm', 'gemma3n')}
    summary['provisional_qwen_phi_gemma_all_resident_simultaneously'] = None
    summary['provisional_generalists_residency_reason'] = 'Gemma not measured: gated access HTTP 403. Do not substitute file size for VRAM.'
    gsum = summary['verified_generalist_sum_idle_vram_mib']
    summary['verified_generalists_idle_sum_fits_gpu'] = gsum <= summary['gpu_capacity_mib'] if gsum is not None else None
    summary['verified_generalists_all_resident_empirically_tested'] = False
    summary['provisional_generalists_verified_count'] = sum(summary['provisional_generalist_status'][m]['status'] == 'verified' for m in ('qwen', 'phi4mm', 'gemma3n'))
    summary['additional_generalist_reason'] = 'Existing local MiniCPM-o-4.5 provides overlapping task adapters already verified by historical formal inference; one minimal load audit passed. Added as a meaningful existing model choice; Gemma remains separately pending.'
    summary['pool_status'] = 'frozen' if summary['frozen'] else 'not_frozen'
    summary['freeze_time_asia_shanghai'] = datetime.now(timezone(timedelta(hours=8))).isoformat() if summary['frozen'] else None
    summary['allowed_module_7b_scope'] = 'Next phase may use this frozen capability/resource inventory; this audit does not supply expanded-pool performance measurements or optimizer/predictor calibration.'
    for filename, value in [('candidate_models.json', {'schema_version': 2, 'models': models, 'metadata_notes': metadata_notes}),
                            ('compatibility_matrix.json', {'allowed_states': ['verified', 'unsupported', 'pending'], 'tasks': list(TASKS), 'models': matrix}),
                            ('load_test_results.json', {'protocol': 'existing Worker process abstraction; sequential single-model load once, one minimal warmup, three sanity inferences, graceful shutdown and GPU release verification',
                                                       'results': tests, 'not_run': {name: preflight[name]['reason'] for name in summary['excluded_or_pending_ids']}}),
                            ('measured_vram.json', {'unit': 'MiB', 'method': 'NVML via nvidia-smi per-process quiescent samples; max observed idle across minimal audit phases; CUDA contexts and retained allocator caches included',
                                                    'gpu_capacity_mib': 49140, 'models': {m['id']: {'measured_idle_vram_mib': m['measured_idle_vram_mib'],
                                                        'measurement_status': tests.get(m['id'], {}).get('status', 'not_run'),
                                                        'raw_evidence': str(OUT / f'raw_load_tests/{m["id"]}.json') if m['id'] in tests else None} for m in models}}),
                            ('pool_resource_summary.json', summary)]:
        (OUT / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False))
    report(summary, models, tests)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def report(summary, models, tests):
    formal = set(summary['formal_model_ids'])
    lines = ['# Module 7A-1 Expanded Candidate Pool Audit', '',
             f'结论：**{summary["pool_status"]}**；本阶段仅做兼容性、静态 metadata、单模型最小加载预审计。', '',
             '## 十二项审计结论', '',
             f'1. Verified local Generalist：**{summary["verified_generalists"]}**（Qwen、Phi、MiniCPM）。指定 Qwen/Phi/Gemma 三者中 verified 为 2；Gemma pending。',
             f'2. Verified Specialist：**{summary["verified_specialists"]}**。',
             f'3. 正式 pool：**{summary["total_model_types"]} model types**；共审查 {len(models)} 条 candidate 记录，pending/rejected 不计入正式 pool。',
             '4. 六个 task 的 capable models 数量见下表；仅计 verified。',
             f'5. 仅一个 specialist 的 task：{summary["tasks_with_only_one_specialist"] or "无"}；零 specialist 的 task：{summary["tasks_with_zero_specialists"]}。visual_qa 目前仅 Qwen capable。',
             '6. SiEBERT 被排除：其 15-dataset 通用 sentiment checkpoint 不满足本阶段严格 SST-2 fine-tuned dataset 条件，尽管 binary label semantics 相同。Gemma 与 GOT 是 pending，不能误写为语义不支持。',
             '7. 每个 model 的实测 idle VRAM 见模型表；未测模型为 null，绝不由文件大小估算。',
             f'8. 正式 pool idle VRAM 总量：**{summary["pool_idle_vram_mib"]} MiB**。',
             f'9. 相对实际容量 49,140 MiB 超出 **{summary["excess_over_gpu_capacity_mib"]} MiB**；严格大于条件：{summary["criteria"]["pool_idle_vram_strictly_exceeds_actual_gpu_capacity"]}。',
             f'10. 在审计的进程 idle 资源模型下，full-pool 不可同时驻留：{summary["resource_competition_established"]}。这是单模型实测之和的结论，未执行 full-pool 联合加载。',
             f'11. 真实资源竞争成立：**{summary["resource_competition_established"]}**；没有降低 GPU capacity 或扩大非任务能力。',
             f'12. 是否可进入 Module 7B：**{summary["can_enter_module_7b"]}**。此处只交付能力/资源前置条件；后续性能建模与 replica 优化未执行。', '',
             '## 六任务覆盖', '', '| Task | Capable models | Generalists | Specialists |', '|---|---:|---:|---:|']
    for task, counts in summary['task_coverage'].items():
        lines.append(f'| {task} | {counts["capable_models"]} | {counts["generalists"]} | {counts["specialists"]} |')
    lines += ['', '## 模型、资源与接口状态', '', '| Model | Role | Verified tasks | Idle VRAM MiB | Primary weights GB | Pool |', '|---|---|---|---:|---:|---|']
    for m in models:
        lines.append(f'| {m["id"]} | {m["role"]} | {", ".join(m["capabilities_verified"]) or "—"} | {m["measured_idle_vram_mib"] if m["measured_idle_vram_mib"] is not None else "null"} | {round(m["weight_size_gb"], 4) if m["weight_size_gb"] is not None else "null"} | {"formal" if m["id"] in formal else m["adapter_status"]} |')
    lines += ['', '## Generalist-only competition', '',
              f'Qwen：17,978 MiB；Phi：11,772 MiB；Gemma：null（HTTP 403，未获本机访问授权）。因此 Qwen/Phi/Gemma 全部共驻留结论为 **unknown**。', '',
              f'所有 verified Generalists（Qwen/Phi/MiniCPM）合计 **{summary["verified_generalist_sum_idle_vram_mib"]} MiB**，按本次 idle accounting 小于 49,140 MiB；算术上能够容纳，但只剩 {49140 - summary["verified_generalist_sum_idle_vram_mib"]} MiB。**未做联合驻留实验**，不声明任意 task inference peak 也能同时容纳。', '',
              'MiniCPM 是项目现有本地候选，有历史同任务 adapter 实验依据，本阶段新增单模型预审计。纳入它形成有意义的重叠选择；没有把它冒充为 Gemma。', '',
              '## 兼容性依据与限制', '',
              '- `verified` 依赖实际当前 task adapter inference；既有模型引用历史正式 summary 与本次 sanity 证据，新模型只在本次实际 inference 通过后置 true。官方 modality 支持只能进入 pending。',
              '- ConvNeXt 使用 torchvision 官方 IMAGENET1K_V1 transforms 与权重；保持相同 RGB 输入、Imagenette 数据集和十类闭集 label space。Transforms 可以按各 pretrained weight 的官方预处理不同。',
              '- RoBERTa 使用 TextAttack 已 fine-tuned 的 SST-2 checkpoint；config 与官方 model zoo 共同确认 task，正负样本验证 ID 映射。实测 runtime classifier 为 124,647,170 参数，相对 DistilBERT 66,955,010 为 1.86 倍；stored checkpoint 的 125,237,762 tensor elements 包含未用于 classifier 的 pooler。FP32 CUDA worker 的 idle footprint 也不同。没有创建、训练或虚构 checkpoint。',
              '- PaddleOCR-VL 只使用元素级 `OCR:` plain-text recognition，沿用 IIIT5K crop 和冻结 normalization。无 document parsing/layout/formula task。',
              '- Whisper Turbo 的 ASR 与 LID 独立检查：LID 调用声学 detect_language，取消 forced English，保存 token map，并验证中文/土耳其语样本；这只是最小接口验证，不是全语言质量 benchmark。Distil Whisper 为 ASR-only，LID=false。',
              '- Phi/MiniCPM 的 visual_qa 与 lid 未经当前 adapter 实验验证，保持 pending；Gemma 的六项 task 均 pending。GOT 为可选 fallback，尚无本地适配实验，OCR pending。',
              '- 新模型的质量没有正式测量；历史 latency 不迁移到新模型，未输出 expanded-pool latency profile。视觉 QA 单模型覆盖仍是当前资源/能力结构的一部分。', '',
              '## 单模型加载与资源测量', '',
              '每个实际测试模型只加载一次；一个最小 warmup、三次 sanity inference，随后 graceful BYE/进程退出，三次 GPU release 检查。Generalist 未在三次 sanity 内覆盖的 task 使用已有冻结 formal inference 证据。NVML idle memory 在每个 phase 采样三次，使用最大观测值，包含各进程 CUDA context 与缓存；不记录 inference peak 为 idle。单模型脚本串行，不存在 co-resident profiling。', '',
              f'成功模型数：{sum(v.get("status") == "passed" for v in tests.values())}；显存释放：{all(v.get("gpu_memory_released") for v in tests.values())}。原始协议中的逐请求时间仅作为 trace 保留，没有用于 latency benchmark、sweep 或搜索。', '',
              '所有 source_revision、选用权重文件、SHA-256、environment package/version/hash 见 JSON。复用现有兼容环境，没有 pip install、更新 Transformers 或改动冻结算法/worker。下载直连只作用于新的 requests session；未改全局代理。文件大小采用 decimal GB，显存采用 NVML MiB，两者明确区分。', '',
              '## Freeze gates', '', '| Gate | Pass |', '|---|---|']
    lines += [f'| {key} | {value} |' for key, value in summary['criteria'].items()]
    lines += ['', '## 官方来源', '',
              '- [Qwen](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)、[Phi](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)、[Gemma](https://huggingface.co/google/gemma-3n-E4B-it)、[MiniCPM](https://huggingface.co/openbmb/MiniCPM-o-4_5)。',
              '- [Torchvision ConvNeXt Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)。',
              '- [PaddleOCR-VL 1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)、[GOT plain OCR API](https://huggingface.co/stepfun-ai/GOT-OCR2_0)。',
              '- [Whisper Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)、[Distil Whisper](https://huggingface.co/distil-whisper/distil-large-v3)。',
              '- [TextAttack model zoo](https://github.com/QData/TextAttack/blob/master/textattack/models/README.md)、[SST-2 checkpoint](https://huggingface.co/textattack/roberta-base-SST-2)、[SiEBERT dataset provenance](https://huggingface.co/siebert/sentiment-roberta-large-english)。', '',
              '本阶段已停止：未运行 MILP/deployment search、full-pool benchmark、multi-replica、integer replica optimization、dynamic adaptation 或 predictor retraining。']
    (OUT / 'candidate_pool_audit.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
