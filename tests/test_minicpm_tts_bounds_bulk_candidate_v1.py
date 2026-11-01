"""Pure CPU value semantics only; no torch import or worker modification."""
import itertools
import unittest

from tools.predictor_validation_20261027.minicpm_tts_bounds_bulk_candidate_v1 import (
    scan_integer_ids, tts_bounds_bulk,
)


def original_bounds(ids, bos, eos, teacher):
    bos_indices, eos_indices = [], []
    for i, x in enumerate(ids):
        if x == bos:
            bos_indices.append(i + 1)
        elif x == eos:
            if teacher and i == len(ids) - 1:
                continue
            eos_indices.append(i)
    return (bos_indices[-1] if bos_indices else -1,
            eos_indices[-1] if eos_indices else None)


class CandidateTests(unittest.TestCase):
    def test_exhaustive_short_integer_sequences_match_original(self):
        for length in range(7):
            for ids in itertools.product((0, 1, 2), repeat=length):
                for teacher in (False, True):
                    self.assertEqual(scan_integer_ids(ids, 1, 2, teacher),
                                     original_bounds(ids, 1, 2, teacher))

    def test_equal_special_ids_preserve_if_elif_precedence(self):
        self.assertEqual(scan_integer_ids([1, 0, 1], 1, 1, True), (3, None))

    def test_last_eos_teacher_forcing_and_repeated_markers(self):
        self.assertEqual(scan_integer_ids([1, 2, 0, 1, 2], 1, 2, True), (4, 1))
        self.assertEqual(scan_integer_ids([1, 2, 0, 1, 2], 1, 2, False), (4, 4))
        self.assertEqual(scan_integer_ids([], 1, 2), (-1, None))

    def test_bulk_read_is_once_and_original_row_is_unchanged(self):
        class FakeRow:
            def __init__(self):
                self.calls = []; self.values = [0, 1, 2]
            def detach(self):
                self.calls.append('detach'); return self
            def cpu(self):
                self.calls.append('cpu'); return self
            def tolist(self):
                self.calls.append('tolist'); return list(self.values)
        row = FakeRow()
        self.assertEqual(tts_bounds_bulk(row, 1, 2), (2, 2))
        self.assertEqual(row.calls, ['detach', 'cpu', 'tolist'])
        self.assertEqual(row.values, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
