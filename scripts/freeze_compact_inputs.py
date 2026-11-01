"""Version protocol metadata; preserve selected IDs/media and repair calibration gold."""
import hashlib
import json
from pathlib import Path

from gsdeploy.profiling.evaluation import read_rows
from gsdeploy.profiling.prepare import digest, write
from gsdeploy.profiling.protocols import restore_sroie_reference
from gsdeploy.tasks import OCR_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/four-task-small-pool-20261015'
ASSETS = Path('/root/autodl-tmp/gsdeploy-assets')


def main():
    old = BASE / 'evaluation-v3'
    new = BASE / 'evaluation-v4'
    new.mkdir()
    report = json.loads((old / 'evaluation-status.json').read_text())
    for task, entry in report['tasks'].items():
        assert digest(entry['path']) == entry['sha256']
        rows = read_rows(entry['path'])
        if task == 'ocr':
            for row in rows:
                row['protocol'] = OCR_PROTOCOL
        report['tasks'][task].update(write(new / (task + '.jsonl'), rows))
        report['tasks'][task]['status'] = 'INPUTS_AND_SCORING_FROZEN_NOT_ADMITTED'
    report['status'] = 'FOUR_TASK_INPUTS_AND_PROTOCOL_FROZEN_NOT_ADMITTED'
    report['predecessor_manifest_sha256'] = digest(old / 'evaluation-status.json')
    report['protocol_sha256'] = digest(ROOT / 'configs/compact_protocol.json')
    report['protocol_revision_note'] = 'Same 1600 IDs, references and media bytes as v3. OCR v4 additionally handles single-column Markdown tables. No model outputs used.'
    report['tasks']['ocr']['reference_audit']['limitation'] = 'Native annotation line order is the fixed evaluation order; no prediction-dependent reading-order alignment.'
    (new / 'evaluation-status.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    source = ROOT / 'artifacts/new-model-profiles-20261015'
    target = BASE / 'calibration-v4'
    target.mkdir()
    native = ASSETS / 'datasets/sroie-doctr/41b3c746a20226fddc80d86d4b2a903d43b5be4f521dd1bbe759dbf8844745e2'
    import pyarrow.parquet as pq
    original = pq.read_table(next((ASSETS / 'datasets/sroie').glob('*/test.parquet'))).to_pylist()
    original = {row['key']: row for row in original}
    entries, all_rows, repairs = {}, [], []
    declared = json.loads((source / 'sample-manifests.json').read_text())['tasks']
    for task in ['text', 'image', 'ocr', 'asr']:
        assert digest(source / (task + '.jsonl')) == declared[task]['sha256']
        rows = read_rows(source / (task + '.jsonl'))
        if task == 'ocr':
            for row in rows:
                raw = (native / (row['id'] + '.txt')).read_bytes()
                new_gold = restore_sroie_reference(original[row['id']], raw.decode('utf-8-sig'))
                repairs.append({'id': row['id'], 'changed': row['label'] != new_gold})
                row.update(label=new_gold, protocol=OCR_PROTOCOL, native_revision=native.name,
                           annotation_sha256=hashlib.sha256(raw).hexdigest())
        entries[task] = write(target / (task + '.jsonl'), rows)
        all_rows.extend(rows)
        if task == 'asr':
            write(target / 'speech.jsonl', rows)
    write(target / 'samples.jsonl', all_rows)
    calibration = {'kind': 'performance_calibration_not_quality_admission', 'source': str(source),
                   'protocol_sha256': digest(ROOT / 'configs/compact_protocol.json'), 'tasks': entries,
                   'ocr_repairs': repairs, 'historical_raw_evidence_modified': False}
    (target / 'sample-manifests.json').write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + '\n')
    study_path = ROOT / 'configs/study.json'
    study = json.loads(study_path.read_text())
    study['quality_preparation'].update(status=report['status'], manifest=str((new / 'evaluation-status.json').relative_to(ROOT)),
                                        manifest_sha256=digest(new / 'evaluation-status.json'))
    study['calibration_preparation'] = {'manifest': str((target / 'sample-manifests.json').relative_to(ROOT)),
                                       'manifest_sha256': digest(target / 'sample-manifests.json')}
    study['tasks']['text']['protocol'] = 'squad2_official_normalization_em_f1_frozen_v1'
    study['tasks']['ocr'].update(protocol=OCR_PROTOCOL, limitation=report['tasks']['ocr']['reference_audit']['limitation'])
    study['tasks']['asr']['quality_protocol'] = 'speaker_disjoint_4_to_30_seconds_quotas_200_150_120_30_frozen_v1'
    study_path.write_text(json.dumps(study, ensure_ascii=False, indent=2) + '\n')
    print({'quality_samples': sum(e['samples'] for e in report['tasks'].values()),
           'calibration_samples': len(all_rows), 'repaired_ocr_labels': sum(r['changed'] for r in repairs)})


if __name__ == '__main__':
    main()
