"""One final OCR shortlist and Image exception, before new GPU measurements."""
import csv, json, ast, shutil
from module14b.common import *


def main():
    assert not (OUT / 'ocr_gapfill_shortlist.json').exists()
    assert not list(RAW.glob('*summary.json'))
    image = []
    for row in csv.DictReader((OUT / 'image_quality_latency_memory.csv').open()):
        p = {k: row[k] for k in ['model_id', 'revision']}
        p.update({k: float(row[k]) for k in ['quality', 's0_ms', 'safe_memory_mib']})
        image.append(eligibility(p, 'image'))
    baseline = read(OUT / 'baseline_references.json')
    assert baseline['tasks']['image']['quality'] == 1.0
    policy = ['higher quality', 'lower latency', 'lower safe memory', 'deterministic model_id@revision']
    save('image_ceiling_exception.json', {
        'at_utc': now(), 'authorized_by': 'Module14B explicit user instruction',
        'regime': REGIME, 'image_only_exception': True,
        'rule': 'accuracy_new>=1.0 AND s0_new>s0_old AND safe_memory_new>safe_memory_old',
        'other_four_tasks_strict_quality_improvement': True, 'baseline': baseline['tasks']['image'],
        'dataset_metric_preprocessing_changed': False, 'Image_admission': 'accuracy>=1.000',
        'threshold_above1_created': False, 'historical_Module14A_not_overwritten': True})
    save('image_heavy_selection.json', {
        'at_utc': now(), 'frozen': True, 'source': ref(OUT / 'image_quality_latency_memory.csv'),
        'candidates': image, 'selected': select(image, 'image'), 'selection_rule': policy,
        'deployment_or_V3_outcome_used': 'NO', 'new_Image_GPU_measurements': False})
    release = read(OUT / 'sources14b/parseq_release_api.json')
    urls = ast.literal_eval(next(n.value for n in ast.parse((SOURCE / 'strhub/models/utils.py').read_text()).body
                                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_WEIGHTS_URL' for t in n.targets)))
    chosen = ['parseq', 'abinet', 'parseq-patch16-224', 'trba']
    universe, entries = [], []
    for name, url in urls.items():
        asset = next(a for a in release['assets'] if a['browser_download_url'] == url)
        row = {
            'model_id': 'baudm/parseq:' + name, 'family': 'PARSeq' if name.startswith('parseq') else name.upper(),
            'checkpoint': asset['name'], 'revision': 'v1.0.0:' + asset['name'],
            'implementation': 'https://github.com/baudm/parseq', 'implementation_commit': COMMIT,
            'checkpoint_url': url, 'release_asset_id': asset['id'], 'observable_checkpoint_bytes': asset['size'],
            'expected_sha256_prefix': asset['name'].rsplit('-', 1)[1].split('.')[0],
            'published_quality': .99 if name == 'parseq' else None,
            'quality_evidence': 'Publisher released PARSeq IIIT5K-3000 evaluation99.00%; not frozen200 local evidence' if name == 'parseq' else 'Released reproduced checkpoint; exact release-specific IIIT5K score not asserted. Family evidence recorded separately.',
            'semantics_status': 'TASK_SEMANTICS_COMPATIBLE', 'public_pretrained': True,
            'no_mandatory_retraining': True, 'local_inference': 'native strhub create_model + strict local state_dict; native forward/tokenizer; no online service',
            'frozen_shortlisted': name in chosen,
            'shortlist_reason': 'Released real-data STR model; native reproducible checkpoint. Two PARSeq input/patch scales plus vision-language refinement ABINet and TPS/CNN/LSTM attention TRBA diversity.' if name in chosen else 'Kept in final discovery universe; smaller/context-free alternatives not additional frozen paths',
            'RTX4090_feasibility': 'Batch1 FP32 expected feasible; actual VRAM measured, no parameter/file-size surrogate'}
        universe.append(row)
        if name in chosen:
            config = __import__('yaml').safe_load((SOURCE / 'configs/experiment' / (name + '.yaml')).read_text())
            base = __import__('yaml').safe_load((SOURCE / 'configs/model' / (config['defaults'][0]['override /model'] + '.yaml')).read_text())
            maincfg = __import__('yaml').safe_load((SOURCE / 'configs/main.yaml').read_text())['model']
            charcfg = __import__('yaml').safe_load((SOURCE / 'configs/charset/94_full.yaml').read_text())['model']
            model_config = {**maincfg, **charcfg, **base, **config.get('model', {})}
            entries.append({**row, 'task': 'ocr', 'experiment': name,
                            'directory': str(MODELS / name), 'weight_path': str(MODELS / name / asset['name']),
                            'native_model_config': model_config,
                            'preprocessing': 'existing word crop decoded RGB; publisher SceneTextDataModule.get_transform native image size/normalization; no augmentation/rotation/detection. Same lowercase a-z0-9 exact-match scorer.',
                            'decoding': 'native model forward defaults, tokenizer softmax greedy decode; PARSeq decode_ar=True/refine_iters=1 publisher defaults; ABINet/TRBA native inference, no lexicon or post-hoc decoder tuning'})
    seeds = [
        ('Topdu/OpenOCR:SVTRv2-B-Union14M', .987, 'SVTRv2', 'https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md', 'https://drive.google.com/drive/folders/11u11ptDzQ4BF9RRsOYdZnXl6ell2h4jN', 'Public folder release; exact asset/config not individually resolved for frozen local path. Retained, no claim of absence of public weights.'),
        ('byeonghu-na/MATRN:best-train-matrn', .967, 'MATRN', 'https://github.com/byeonghu-na/MATRN', 'https://www.dropbox.com/s/pjcarm73cqwbxh4/best-train-matrn.pth?dl=0', 'Public pretrained, original Torch1.7/fastai1 stack; not shortlisted under four-slot reproducibility/diversity cap. No local quality claimed.'),
        ('wangyuxin87/VisionLAN:LA-final', .959, 'VisionLAN', 'https://github.com/wangyuxin87/VisionLAN', 'https://github.com/wangyuxin87/VisionLAN#evaluation', 'Publisher IIIT5K95.9%, Python2.7/Torch1.3 stack and password-backed public artifacts; lower preliminary quality evidence, not locally measured.'),
        ('FangShancheng/ABINet:ABINet-LV', .970, 'ABINet', 'https://github.com/FangShancheng/ABINet', 'https://drive.google.com/file/d/1mYM_26qHUom_5NU7iutHneB_KHlLjL5y/view', 'Original checkpoint bundle; selected separate publisher STRHub reproduced ABINet checkpoint, not silently interchangeable.')]
    for model_id, q, family, source, url, reason in seeds:
        universe.append({'model_id': model_id, 'family': family, 'published_quality': q,
                         'quality_evidence': 'Publisher IIIT5K3000 word accuracy; different subset from local200',
                         'implementation': source, 'checkpoint_url': url, 'public_pretrained': True,
                         'semantics_status': 'TASK_SEMANTICS_COMPATIBLE', 'frozen_shortlisted': False,
                         'shortlist_reason': reason, 'no_mandatory_retraining': True})
    csvsave('ocr_gapfill_candidate_universe.csv', universe)
    entries.sort(key=lambda e: chosen.index(e['experiment']))
    for e in entries: Path(e['directory']).mkdir(exist_ok=True)
    oldproto = read(OUT / 'measurement_protocol_freeze.json')
    protocol = {k: oldproto[k] for k in ['latency_repetitions', 'requests_per_repetition', 'warmup_per_fresh_process', 'latency_estimator', 'timer', 'idle_memory', 'peak_memory', 'safe_memory', 'threads', 'worker_architecture']}
    protocol.update(at_utc=now(), frozen=True, round='ONE_FINAL_OCR_GAP_FILLING_ROUND',
                    entries=entries, datasets={'ocr': oldproto['datasets']['ocr']},
                    eligibility='exact_match>0.960 AND s0>frozen_full_precision6.0675249ms AND safeM>frozen_full_precision630.9650465847725MiB',
                    comparison_uses_full_precision_frozen_baseline=True,
                    baseline=baseline['tasks']['ocr'], safe_factor=baseline['safe_factor'],
                    dtype='float32 native unquantized/no autocast; unchanged Torch version inherited',
                    measurement_order='All4 quality runs in listed order, then3 repetitions. Each repetition original PP-OCRv6 control first, then all4 listed candidates; fresh process each run.',
                    selection_rule=policy, OCR_SLO_rule='After all measurements, one-time midpoint(oldQ,selectedQ); exact_match>=threshold; null if no eligible; no deployment input',
                    no_third_round=True, deployment_or_V3_outcome_used='NO')
    save('ocr_gapfill_shortlist.json', protocol)
    files = [OUT / n for n in ['ocr_gapfill_shortlist.json', 'ocr_gapfill_candidate_universe.csv', 'image_ceiling_exception.json', 'image_heavy_selection.json', 'module14b_protected_before.json']]
    save('module14b_discovery_freeze.json', {'at_utc': now(), 'before_any_new_OCR_GPU_measurement': True,
         'files': [ref(p) for p in files], 'native_source': [ref(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()],
         'source_archive': ref(MODELS / 'parseq_source.tar.gz'), 'no_outcome_conditioning': True})
    print('FROZEN_FINAL_OCR_ROUND', len(universe), 'discovered', chosen)


if __name__ == '__main__':
    main()
