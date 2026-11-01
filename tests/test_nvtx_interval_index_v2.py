"""Brute-force ownership oracle, adversarial intervals and read-only SQLite."""
import hashlib
import math
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest

from tools.predictor_validation_20261027.nvtx_interval_index_v2 import NVTXRangeIndex
from tools.predictor_validation_20261027.nvtx_interval_audit_v2 import audit_sqlite


def interval(tid, start, end, stage=True, name=None):
    return dict(global_tid=tid, nsys_start_ns=start, nsys_end_ns=end,
                stage=('r', 'gpu', name or str(start)) if stage else None,
                label=name, kernels=[])


def oracle(ranges, tid, start, end):
    selected = []
    for value in ranges:
        if value['global_tid'] != tid:
            continue
        if start < value['nsys_start_ns'] or end > value['nsys_end_ns']:
            continue
        selected.append(value)
    staged = [r for r in selected if r['stage']]
    preferred = staged if staged else selected
    return selected, min(preferred, key=lambda r: r['nsys_end_ns']-r['nsys_start_ns']) if preferred else None


class IntervalIndexTests(unittest.TestCase):
    def test_stage_priority_nesting_crossing_and_stable_duplicate_ties(self):
        ranges = [interval(1, 0, 100, name='first'), interval(1, 50, 150, name='crossing'),
                  interval(1, 10, 20, False), interval(1, 0, 100, name='duplicate')]
        index = NVTXRangeIndex(ranges)
        self.assertIs(index.owner_for_launch(1, 12, 13), ranges[0])  # stage beats smaller nonstage
        self.assertIs(index.owner_for_launch(1, 60, 70), ranges[0])  # equal duration, source order
        self.assertIs(index.owner_for_launch(1, 90, 110), ranges[1])
        self.assertIsNone(index.owner_for_launch(1, 140, 160))
        self.assertEqual(index.containing(1, 60, 70), [ranges[0], ranges[1], ranges[3]])
        index.owner_for_launch(1, 12, 13)['kernels'].append('k')
        self.assertEqual(ranges[0]['kernels'], ['k'])  # original object retained

    def test_both_launch_endpoints_and_inclusive_equal_boundaries(self):
        value = interval(1, 10, 20)
        index = NVTXRangeIndex([value])
        for start, end in ((10, 20), (10, 10), (20, 20)):
            self.assertIs(index.owner_for_launch(1, start, end), value)
        for start, end in ((9, 15), (15, 21), (0, 30), (21, 21)):
            self.assertIsNone(index.owner_for_launch(1, start, end))

    def test_exact_thread_identity_not_only_process_or_correlation(self):
        pid1, pid2 = 123 << 24, 456 << 24
        ranges = [interval(pid1+1, 0, 100), interval(pid1+2, 10, 20), interval(pid2+1, 11, 19)]
        index = NVTXRangeIndex(ranges)
        self.assertIs(index.owner_for_launch(pid1+1, 12, 18), ranges[0])
        self.assertIs(index.owner_for_launch(pid1+2, 12, 18), ranges[1])
        self.assertIs(index.owner_for_launch(pid2+1, 12, 18), ranges[2])
        stats = {}
        self.assertIsNone(index.owner_for_launch(pid2+2, 12, 18, stats=stats))
        self.assertEqual(stats['visited_nodes'], 0)

    def test_random_crossing_duplicate_unsorted_ranges_match_independent_oracle(self):
        rng = random.Random(2026102801)
        ranges = [interval(rng.randrange(4), (left := rng.randrange(500)), left+rng.randrange(200),
                           bool(rng.randrange(2)), str(i)) for i in range(300)]
        ranges.extend([dict(ranges[7]), dict(ranges[7]), interval(2, 0, 1000)])
        rng.shuffle(ranges)
        index = NVTXRangeIndex(ranges)
        for _ in range(1200):
            tid, start = rng.randrange(5), rng.randrange(700)
            end = start+rng.randrange(60)
            expected, owner = oracle(ranges, tid, start, end)
            self.assertEqual([id(r) for r in index.containing(tid, start, end)], [id(r) for r in expected])
            self.assertIs(index.owner_for_launch(tid, start, end), owner)

    def test_sparse_query_node_visits_scale_with_height_not_all_ranges(self):
        for count in (255, 2047, 8191):
            ranges = [interval(1, 10*i, 10*i+5) for i in range(count)]
            index = NVTXRangeIndex(ranges)
            work = 0
            for chosen in range(0, count, max(1, count//64)):
                stats = {}
                self.assertIs(index.owner_for_launch(1, 10*chosen+1, 10*chosen+2, stats=stats), ranges[chosen])
                self.assertLessEqual(stats['visited_nodes'], 2*math.ceil(math.log2(count+1))+1)
                self.assertEqual(stats['matched_ranges'], 1)
                work += stats['visited_nodes']
            self.assertLess(work, count*len(range(0, count, max(1, count//64)))/10)

    def test_large_nested_output_is_explicit_linear_work_and_not_claimed_constant(self):
        ranges = [interval(1, i, 10000-i) for i in range(1023)]
        stats = {}
        index = NVTXRangeIndex(ranges)
        self.assertIs(index.owner_for_launch(1, 2000, 2001, stats=stats), ranges[-1])
        self.assertEqual(stats['matched_ranges'], len(ranges))
        self.assertEqual(stats['visited_nodes'], len(ranges))

    def test_invalid_boundaries_refused_and_empty_index_valid(self):
        for bad in (interval(1, 10, 9), interval(1, 0., 10), interval(True, 0, 10)):
            with self.assertRaises(ValueError):
                NVTXRangeIndex([bad])
        index = NVTXRangeIndex([])
        self.assertIsNone(index.owner_for_launch(1, 0, 1))
        with self.assertRaises(ValueError):
            index.owner_for_launch(1, 10, 9)

    def test_sqlite_process_scoped_launch_lookup_full_kernel_equivalence_and_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'multi-process.sqlite'
            connection = sqlite3.connect(path)
            connection.executescript('''
                create table StringIds(id integer,value text);
                create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
                create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer);
                create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer);
            ''')
            p1, p2 = 123 << 24, 456 << 24
            connection.executemany('insert into NVTX_EVENTS values(?,?,?,?,?)', [
                (0, 100, 'GS_STAGE:r:gpu:outer', None, p1+1),
                (10, 20, 'generic', None, p1+1),
                (10, 20, 'GS_STAGE:r:gpu:second-thread', None, p1+2),
                (10, 20, 'GS_STAGE:r:gpu:other-process', None, p2+1)])
            connection.executemany('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?)', [
                (12, 15, p1+1, 7), (12, 15, p2+1, 7), (19, 21, p1+2, 8)])
            connection.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?)', [
                (15, 30, p1, 7), (15, 30, p2, 7), (22, 30, p1, 8), (25, 30, p2, 999), (35, 40, p1, 7)])
            connection.commit(); connection.close()
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            result = audit_sqlite(path)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['kernel_count'], 5)
            self.assertEqual(result['attributed_kernel_count'], 3)
            self.assertEqual(result['unique_launch_intervals'], 3)
            self.assertEqual(result['process_count'], 2)
            self.assertEqual(result['indexed_kernel_ownership_sha256'], result['brute_kernel_ownership_sha256'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertFalse(path.with_name(path.name+'-journal').exists())


if __name__ == '__main__':
    unittest.main()
