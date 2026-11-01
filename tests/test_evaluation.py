"""CPU-only regression checks for reference corruption and evaluation leakage."""

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from gsdeploy.profiling.evaluation import select_qa, select_images, select_asr
from gsdeploy.profiling.protocols import page_ocr_text, parse_sroie_annotation, restore_sroie_reference


class EvaluationProtocolTests(unittest.TestCase):
    def test_native_sroie_preserves_all_commas_and_ignored_regions(self):
        rows = parse_sroie_annotation(
            '\ufeff1,2,3,4,5,6,7,8,NO.2, JALAN TEMENGGUNG 19/9, SELANGOR\n'
            '1,2,3,4,5,6,7,8,###\n')
        self.assertEqual(rows[0]['text'], 'NO.2, JALAN TEMENGGUNG 19/9, SELANGOR')
        self.assertEqual(rows[0]['polygon'], list(range(1, 9)))
        self.assertFalse(rows[0]['ignored'])
        self.assertTrue(rows[1]['ignored'])
        for invalid in ['', '1,2,TEXT', 'x,2,3,4,5,6,7,8,TEXT']:
            with self.assertRaises(ValueError):
                parse_sroie_annotation(invalid)

    def test_reference_recovery_rejects_mismatched_source_geometry_or_text(self):
        annotation = '1,2,3,4,5,6,7,8,NO.2, JALAN TEMENGGUNG 19/9, SELANGOR'
        source = {'words': ['NO.2'], 'bboxes': [[1, 2, 5, 6]]}
        self.assertEqual(restore_sroie_reference(source, annotation),
                         'NO.2, JALAN TEMENGGUNG 19/9, SELANGOR')
        for wrong in [{'words': ['NO.3'], 'bboxes': [[1, 2, 5, 6]]},
                      {'words': ['NO.2'], 'bboxes': [[1, 2, 5, 7]]}]:
            with self.assertRaises(ValueError):
                restore_sroie_reference(wrong, annotation)

    def test_ocr_formatting_keeps_prices_commas_and_visible_content(self):
        html = '<table><tr><td>ITEM</td><td>PRICE</td></tr><tr><td>A, B &amp; C</td><td>-1.20</td></tr></table>'
        markdown = '| ITEM | PRICE |\n| --- | ---: |\n| A, B & C | -1.20 |'
        plain = 'ITEM PRICE A, B & C -1.20'
        self.assertEqual(page_ocr_text(html), page_ocr_text(plain))
        self.assertEqual(page_ocr_text(markdown), page_ocr_text(plain))
        self.assertEqual(page_ocr_text('x < 3 | total -1.20, USD'), 'x < 3 | total -1.20, usd')
        self.assertEqual(page_ocr_text('| TOTAL |\n| --- |\n| -1.20 |'), 'total -1.20')
        self.assertIn('title:', page_ocr_text('---\ntitle: RECEIPT\n---\nTOTAL 9.00'))

    def test_qa_excludes_entire_calibration_articles_and_is_deterministic(self):
        articles = []
        for title in ['calibration', 'heldout-a', 'heldout-b']:
            articles.append({'title': title, 'paragraphs': [{'context': 'c' * 1100, 'qas': [
                {'id': title + str(i), 'question': 'q', 'is_impossible': bool(i % 2),
                 'answers': [] if i % 2 else [{'text': 'answer'}]} for i in range(8)]}]})
        calibration = [{'id': 'calibration0', 'article': 'calibration'}]
        chosen = select_qa({'data': articles}, calibration, 8)
        self.assertEqual(len(chosen), 8)
        self.assertEqual(sum(bool(row['answers']) for row in chosen), 4)
        self.assertNotIn('calibration', {row['article'] for row in chosen})
        self.assertEqual(chosen, select_qa({'data': list(reversed(articles))}, calibration, 8))
        with self.assertRaises(ValueError):
            select_qa({'data': articles}, calibration, 20)

    def test_image_selection_excludes_renamed_calibration_content(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / 'source.tar'
            with tarfile.open(shard, 'w') as tar:
                for label in range(200):
                    for suffix in ['seen', 'unseen']:
                        data = (str(label) + suffix).encode()
                        for extension, payload in [('jpg', data), ('cls', str(label).encode())]:
                            info = tarfile.TarInfo(f'{label}-{suffix}.{extension}')
                            info.size = len(payload)
                            tar.addfile(info, io.BytesIO(payload))
            calibration = [{'id': 'different-name', 'input_sha256': hashlib.sha256((str(i) + 'seen').encode()).hexdigest()}
                           for i in range(200)]
            rows = select_images([shard], calibration, {str(i): [f'n{i}', f'name{i}'] for i in range(200)},
                                 list(range(200)), root / 'images', 'pinned-revision', per_class=1)
            self.assertEqual(len(rows), 200)
            self.assertTrue(all(row['id'].endswith('-unseen') for row in rows))
            self.assertEqual(len({row['input_sha256'] for row in rows}), 200)


class AudioSplitTests(unittest.TestCase):
    def test_asr_excludes_calibration_speakers_and_duplicate_audio(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import soundfile as sf
            import numpy as np
        except ImportError:
            self.skipTest("Optional CPU data dependencies not installed")
        import hashlib
        records = []
        duplicate = None
        for index, duration in enumerate([5, 9, 13, 21]):
            buffer = io.BytesIO()
            audio = np.zeros(duration * 16000, dtype=np.float32)
            audio[index + 1] = 0.5
            sf.write(buffer, audio, 16000, format="FLAC")
            data = buffer.getvalue()
            records.extend([{'id': f'old-speaker-{index}', 'speaker_id': 1, 'text': 'OLD', 'audio': {'bytes': data}},
                            {'id': f'heldout-{index}', 'speaker_id': 2, 'text': 'NEW', 'audio': {'bytes': data}}])
            if index == 0:
                buffer = io.BytesIO()
                sf.write(buffer, np.zeros(duration * 16000, dtype=np.float32), 16000, format="FLAC")
                duplicate = buffer.getvalue()
                records.append({'id': 'renamed-calibration-audio', 'speaker_id': 3,
                                'text': 'LEAK', 'audio': {'bytes': duplicate}})
        calibration = [{'id': 'calibration', 'speaker_id': 1,
                        'input_sha256': hashlib.sha256(duplicate).hexdigest()}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'source.parquet'
            pq.write_table(pa.Table.from_pylist(records), path)
            selected = select_asr(path, calibration, root / 'audio', per_stratum=1)
            self.assertEqual({row['id'] for row in selected}, {f'heldout-{i}' for i in range(4)})
            self.assertEqual({row['speaker_id'] for row in selected}, {2})


if __name__ == '__main__':
    unittest.main()
