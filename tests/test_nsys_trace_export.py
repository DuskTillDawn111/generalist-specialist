import sqlite3
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.predictor_validation_20261027.trace_export import parse_sqlite


class TraceExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "trace.sqlite"
        c = sqlite3.connect(self.path)
        c.executescript("""
          create table StringIds(id integer,value text);
          create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
          create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer);
          create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,demangledName integer,
            globalPid integer,correlationId integer,deviceId integer,streamId integer,
            registersPerThread integer,staticSharedMemory integer,dynamicSharedMemory integer);
        """)
        c.execute("insert into StringIds values(1,'real_kernel')")
        pid, tid = (0x1000000 + 123) << 24, ((0x1000000 + 123) << 24) + 124
        c.execute("insert into NVTX_EVENTS values(100,200,'GS_STAGE:r:gpu:s',null,?)", (tid,))
        c.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?)", [(110,115,tid,1),(120,125,tid,2)])
        c.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,1,?,?,0,7,32,0,0)", [(130,160,pid,1),(150,180,pid,2)])
        c.commit(); c.close()
        self.anchor = {"request_id": "r", "segment_id": "s", "kind": "gpu", "pid": 123,
                       "start_ns": 1000100, "end_ns": 1000200}

    def test_real_kernel_union_mapping_and_unknown_demand(self):
        result = parse_sqlite(self.path, [self.anchor])
        phase = result["phases"][0]
        self.assertEqual(phase["kernel_union_busy_ns"], 50)
        self.assertEqual(phase["kernel_duration_sum_ns"], 60)
        self.assertEqual(phase["kernels"][0]["start_ns"], 1000130)
        self.assertIsNone(phase["demand"])
        self.assertEqual(result["clock_mappings"]["123"]["max_boundary_residual_ns"], 0)

    def test_no_anchor_does_not_claim_common_clock(self):
        result = parse_sqlite(self.path)
        self.assertEqual(result["phases"], [])
        self.assertEqual(len(result["nvtx_ranges"][0]["kernels"]), 2)
        self.assertNotIn("start_ns", result["nvtx_ranges"][0]["kernels"][0])

    def test_bad_mapping_keeps_kernel_clock_unknown(self):
        anchor = {**self.anchor, "end_ns": self.anchor["end_ns"] + 10000000}
        phase = parse_sqlite(self.path, [anchor], max_anchor_skew_ns=100)["phases"][0]
        self.assertFalse(phase["clock_mapping_valid"])
        self.assertNotIn("start_ns", phase["kernels"][0])

    def test_launch_correlation_is_process_scoped(self):
        c = sqlite3.connect(self.path)
        c.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(110,115,?,1)", ((456 << 24)+1,))
        c.commit(); c.close()
        self.assertEqual(parse_sqlite(self.path, [self.anchor])["phases"][0]["kernel_count"], 2)

    def test_absent_static_geometry_is_explicitly_unknown(self):
        kernel = parse_sqlite(self.path)['nvtx_ranges'][0]['kernels'][0]
        self.assertIsNone(kernel['block'])
        self.assertIsNone(kernel['grid'])
        self.assertIsNone(kernel['allocated_registers_per_block'])
        self.assertIsNone(kernel['allocation_evidence'])

    def test_observed_geometry_is_preserved_without_inventing_architecture_allocation(self):
        c = sqlite3.connect(self.path)
        for field in ('blockX', 'blockY', 'blockZ', 'gridX', 'gridY', 'gridZ', 'contextId',
                      'clusterX', 'clusterY', 'clusterZ'):
            c.execute('alter table CUPTI_ACTIVITY_KIND_KERNEL add column ' + field + ' integer')
        c.execute('update CUPTI_ACTIVITY_KIND_KERNEL set blockX=128,blockY=1,blockZ=1,gridX=40,gridY=2,gridZ=1,contextId=3,clusterX=0,clusterY=0,clusterZ=0')
        c.commit(); c.close()
        kernel = parse_sqlite(self.path)['nvtx_ranges'][0]['kernels'][0]
        self.assertEqual(kernel['block'], [128, 1, 1])
        self.assertEqual(kernel['grid'], [40, 2, 1])
        self.assertEqual(kernel['context_id'], 3)
        self.assertEqual(kernel['cluster_raw'], [0, 0, 0])
        self.assertIsNone(kernel['cluster'])
        self.assertIsNone(kernel['allocated_registers_per_block'])

    def test_integrated_index_preserves_full_parser_output_against_scan_oracle(self):
        class BruteIndex:
            def __init__(self, ranges):
                self.ranges = ranges

            def owner_for_launch(self, tid, start, end):
                matching = [r for r in self.ranges if r['global_tid'] == tid
                            and r['nsys_start_ns'] <= start and end <= r['nsys_end_ns']]
                stages = [r for r in matching if r['stage']]
                return min(stages or matching, key=lambda r: r['nsys_end_ns']-r['nsys_start_ns']) if matching else None

        indexed = parse_sqlite(self.path, [self.anchor])
        with patch('tools.predictor_validation_20261027.trace_export.NVTXRangeIndex', BruteIndex):
            scanned = parse_sqlite(self.path, [self.anchor])
        self.assertEqual(indexed, scanned)

    def test_full_json_equivalence_normalizes_tuple_arrays_and_detects_numeric_drift(self):
        from tools.predictor_validation_20261027.nvtx_parser_equivalence_v2 import main
        legacy = Path(self.tmp.name)/'legacy.json'
        document = parse_sqlite(self.path, [self.anchor])
        legacy.write_text(json.dumps(document))
        output = Path(self.tmp.name)/'equivalence'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--legacy-parsed', str(legacy), '--output', str(output)]), 0)
        report = json.loads((output/'report.json').read_text())
        self.assertEqual(report['status'], 'PASS')
        self.assertTrue(report['traces'][0]['full_output_exactly_equal'])
        document['clock_mappings']['123']['offset_ns'] += 1
        legacy.write_text(json.dumps(document))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--legacy-parsed', str(legacy), '--output', str(output)+'-changed']), 1)
        changed = json.loads(Path(str(output)+'-changed/report.json').read_text())
        self.assertIn('offset_ns', changed['traces'][0]['first_difference'])


if __name__ == "__main__":
    unittest.main()
