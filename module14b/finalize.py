"""Verify frozen rules and historical integrity; answer all13 final questions."""
import csv, json, math
from module14b.common import *


def fmt(value, digits=3):
    return 'N/A' if value is None else f'{float(value):.{digits}f}'


def main():
    gate = read(OUT / 'module14b_gate.json')
    pool = read(OUT / 'quality_nondegraded_heavy_pool.json')
    ocrselection = read(OUT / 'ocr_gapfill_selection.json')
    imageselection = read(OUT / 'image_heavy_selection.json')
    proto = read(OUT / 'ocr_gapfill_shortlist.json')
    baseline = read(OUT / 'baseline_references.json')
    locked = read(OUT / 'module14b_execution_freeze.json')
    discovery = read(OUT / 'module14b_discovery_freeze.json')
    old = read(OUT / 'module14b_protected_before.json')['files']
    for r in old + locked['implementation'] + locked['native_source'] + locked['inputs'] + discovery['files']:
        assert ref(r['path'])['sha256'] == r['sha256'], r['path']
    for m in read(OUT / 'module14b_download_manifest.json')['models']:
        if m['success']: assert ref(m['file']['path'])['sha256'] == m['file']['sha256']
    runs = [read(p) for p in RAW.glob('*summary.json')]
    runs.sort(key=lambda r: r['at_utc'])
    assert runs and locked['at_utc'] < runs[0]['at_utc'] and proto['at_utc'] < runs[0]['at_utc']
    assert all(not r.get('foreign_process_samples') for r in runs)
    assert all(runs[i]['completed_at_utc'] <= runs[i + 1]['at_utc'] for i in range(len(runs) - 1))
    assert all(r['task'] == 'ocr' for r in runs)
    paths = []
    for entry in proto['entries']:
        own = [r for r in runs if r['model_id'] == entry['model_id']]
        for r in own:
            if not r['success']: continue
            rows = [json.loads(l) for l in Path(r['raw_path']).read_text().splitlines()]
            d = proto['datasets']['ocr']
            if r['phase'] == 'quality':
                assert len(rows) == 200
                assert [row['pool_id'].split(':', 1)[1] for row in rows] == d['quality_ids_in_order']
            else:
                assert len(rows) == 30
                assert [row['pool_id'] for row in rows] == d['latency_ids_in_order']
            assert r['ready']['dtype'] == 'torch.float32' and r['ready']['device'] == 'cuda:0'
        paths.append({'model_id': entry['model_id'], 'successful_quality_runs': sum(r['success'] and r['phase'] == 'quality' for r in own),
                      'successful_performance_repetitions': sum(r['success'] and r['phase'] == 'latency' for r in own)})
    for p in ocrselection['candidates']:
        recomputed = eligibility(p, 'ocr')
        assert p['HEAVY_ELIGIBLE'] == recomputed['HEAVY_ELIGIBLE']
        if p['safe_memory_mib'] is not None:
            assert math.isclose(p['safe_memory_mib'], p['idle_memory_mib'] * baseline['safe_factor'], rel_tol=1e-12)
    assert imageselection['selected'] == select(imageselection['candidates'], 'image')
    if ocrselection['control_valid']: assert ocrselection['selected'] == select(ocrselection['candidates'], 'ocr')
    oldselection = read(OUT / 'hq_heavy_specialist_selection.json')
    oldslo = read(OUT / 'hq_quality_slo.json')
    for task in ['asr', 'lid', 'text']:
        selected = pool['tasks'][task]['selected']
        for k in ['model_id', 'revision', 'quality', 's0_ms', 'safe_memory_mib']:
            assert selected[k] == oldselection['tasks'][task]['selected'][k]
        assert pool['quality_SLOs'][task] == oldslo['tasks'][task]['new_HQ_SLO']
    assert pool['quality_SLOs']['image']['value'] == 1.0
    if ocrselection['selected']:
        threshold = pool['quality_SLOs']['ocr']['value']
        assert baseline['tasks']['ocr']['quality'] < threshold <= ocrselection['selected']['quality']
        assert threshold == (baseline['tasks']['ocr']['quality'] + ocrselection['selected']['quality']) / 2
    validation = {'at_utc': now(), 'passed': True, 'historical_protected_files_unchanged': len(old),
        'Module14A_selections_SLOs_exactly_preserved': True, 'source_code_input_freezes_verified': True,
        'execution_lock_before_first_GPU_result': True, 'exact_frozen_sample_order_verified': True,
        'strict_OCR_and_ceiling_Image_eligibility_recomputed': True, 'original_safe_memory_definition_preserved': True,
        'GPU_intervals_nonoverlapping_no_foreign_processes': True, 'GPU_runs': len(runs),
        'successful_GPU_runs': sum(r['success'] for r in runs), 'paths': paths,
        'third_OCR_search_started': False, 'g_gt1_or_deployment_started': False,
        'old_efficiency_regime_and_Module9_unmodified': True, 'outcome_used': 'NO'}
    save('module14b_validation.json', validation)
    lines = ['# Module14B — Final Heavy Specialist Pool',
        '\n正式第二regime：**' + REGIME + '**；gate：**' + gate['gate'] + '**。旧EFFICIENCY_OPTIMAL_SPECIALIST_REGIME不修改；14A历史文件不重写。仅Image采用显式授权的ceiling exception，其余四task严格提高质量并增加s0/safe M。',
        '\n## 1–3. Image ceiling-aware reclassification 与最终选择',
        '\n| Candidate | Q | s0 ms | Safe MiB | Latency ratio | Memory ratio | Eligible |',
        '|---|---:|---:|---:|---:|---:|---|']
    for p in imageselection['candidates']:
        lines.append('| ' + ' | '.join([p['model_id'], fmt(p['quality']), fmt(p['s0_ms']), fmt(p['safe_memory_mib']),
            fmt(p['latency_ratio']), fmt(p['memory_ratio']), str(p['HEAVY_ELIGIBLE'])]) + ' |')
    image = imageselection['selected']
    lines.append('\n选择 **' + image['model_id'] + '**，Q保持1.000；三模型质量相等，选择最低真实s0，无deployment输入。数据复用14A，没有新Image GPU测量；checkpoint-native transforms和原closed-set10冻结语义不变。')
    lines += ['\n## 4. 最后一轮 OCR 搜索与冻结名单',
        '\n| Candidate | Family | Published IIIT5K quality | Frozen measurement |', '|---|---|---:|---|']
    universe = list(csv.DictReader((OUT / 'ocr_gapfill_candidate_universe.csv').open()))
    for row in universe:
        lines.append('| ' + ' | '.join([row['model_id'], row['family'], fmt(row['published_quality'] or None), row['frozen_shortlisted']]) + ' |')
    lines += ['\nSVTRv2、MATRN、VisionLAN、原ABINet及STRHub多个release均检索。冻结四路径按公开checkpoint可复现性和native STR架构/输入规模多样性选择：PARSeq128x32、ABINet、PARSeq224x224、TRBA。GoogleDrive folder checkpoint未独立resolve、VisionLAN旧Python2/Torch1栈及MATRN旧fastai栈作为metadata局限如实记录，不宣称这些模型不存在。',
        '\n四个checkpoint均固定publisher release v1.0.0文件及SHA前缀，implementation commit=' + COMMIT + '。权重完整SHA在GPU前execution lock中固定。官方GitHub浏览下载超时用同一release asset REST ID修复传输；未换权重，早期失败与完整哈希保留。独立新STR environment不修改旧runtime；native decoding defaults，不人为增加refinement/beam/生成长度来放慢模型。',
        '\n公开quality来自其他IIIT5K3000/source evaluation，不能替代本机冻结200评估；不同checkpoint的家族score不冒充exact release score。原OCR是cropped-word recognition-only；仅保留native recognizer内部TPS等组件，不调用外部detector、裁剪生成或document understanding。',
        '\n## 5–7. 本机 OCR Q / standalone s0 / safe M',
        '\n| Model | Local success | Exact match | s0 ms | Safe MiB | Latency ratio | Memory ratio | Strict eligible | Failed criteria |',
        '|---|---|---:|---:|---:|---:|---:|---|---|']
    for p in ocrselection['candidates']:
        lines.append('| ' + ' | '.join([p['model_id'], str(p['local_run_success']), fmt(p['quality']), fmt(p['s0_ms']),
            fmt(p['safe_memory_mib']), fmt(p['latency_ratio']), fmt(p['memory_ratio']), str(p['HEAVY_ELIGIBLE']),
            ', '.join(p['failed_criteria'])]) + ' |')
    lines += ['\n性能固定3 fresh-process repetitions×30既有样本，原worker完整handler timer和warmup；s0为全部90样本算术均值，不挑最快run或删除outliers。原safe M=' + str(baseline['safe_factor']) + '×实测post-warmup idle NVML，peak另报；不使用参数/文件大小/模型卡memory替代。',
        '\n## 8–9. 最终 OCR 与一次性 HQ SLO']
    winner = ocrselection['selected']
    lines.append('\n' + ('最终 **' + winner['model_id'] + '**，exact match=' + fmt(winner['quality']) + '。新OCR SLO：exact_match≥' + str(pool['quality_SLOs']['ocr']['value']) + '，等于(0.960+真实选中质量)/2。' if winner else '**OCR_HEAVY_SPECIALIST_NOT_FOUND**。新OCR HQ SLO=null；不把旧baseline填入heavy pool，不放宽质量/latency/memory门槛。'))
    lines += ['\n## 10–11. 最终五task pool / 相对旧模型的变化',
        '\n| Task | Frozen heavy model | Old Q | New Q | Signed quality delta | s0 ms | Safe MiB | Latency ratio | Memory ratio |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for task, row in pool['tasks'].items():
        w = row['selected']; b = baseline['tasks'][task]
        lines.append('| ' + ' | '.join([task, w['model_id'] if w else 'NOT_FOUND', fmt(b['quality'], 9),
            fmt(w['quality'], 9) if w else 'N/A', fmt(w['quality_delta'], 9) if w else 'N/A',
            fmt(w['s0_ms']) if w else 'N/A', fmt(w['safe_memory_mib']) if w else 'N/A',
            fmt(w['latency_ratio']) if w else 'N/A', fmt(w['memory_ratio']) if w else 'N/A']) + ' |')
    lines.append('\nQuality delta统一new-old；ASR负WER delta代表改善。ASR/LID复用同一Whisper-turbo权重，但task-path Q/s0独立保留，shared residency取已测safe envelope max，不把两个latency混为一个值。')
    lines.append('\n五任务完整=' + str(pool['complete_five_task_pool']) + '，missing=' + ', '.join(gate['missing_tasks']) + '。')
    lines += ['\nSLO：ASR≤0.036347318824321295、LID≥0.9875、Text≥0.9475完全保留14A；Image accuracy≥1.000，同时pool固定ConvNeXt-large身份；OCR按本轮一次性中点/null。原efficiency policy未改。',
        '\n## 12–13. Outcome evidence / gate / 停止',
        '\n模型与SLO选择是否使用deployment/V3 outcome：**NO**。未运行optimizer/deployment/lowQPS/multi-hotspot/Task-Rich/replica frontier/g(x>1)，Module9不修改。',
        '\nGate：**' + gate['gate'] + '**。OCR final search round已用尽，无第三轮模型搜索。',
        '\n核验：' + str(len(old)) + '个历史受保护文件未变；新GPU运行' + str(len(runs)) + '次，成功' + str(sum(r['success'] for r in runs)) + '次。freeze timestamps、source/inputs hashes、样本顺序、资格和SLO核验通过。',
        '\n机器为RTX4090，实际49140MiB，不宣称标准24GB本机验证。同期OCR baseline control漂移独立记录，历史分母不重定义。这里只确认单replica operating points，不提供deployment性能结论。',
        '\nPrimary sources： [PARSeq/STRHub](https://github.com/baudm/parseq)、[PARSeq paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177.pdf)、[SVTRv2](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md)、[MATRN](https://github.com/byeonghu-na/MATRN)、[VisionLAN](https://github.com/wangyuxin87/VisionLAN)、[ABINet](https://github.com/FangShancheng/ABINet)。完整metadata、release、implementation源码与raw evidence均保留。完成后停止。']
    (OUT / 'module14b_summary.md').write_text('\n'.join(lines) + '\n')
    print('FINAL_VALIDATION_PASS', len(old), 'gate', gate['gate'], flush=True)


if __name__ == '__main__': main()
