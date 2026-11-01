"""Read-only capability audit fixtures; no device/profiler process is run."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027.capability_audit_v2 import audit, main, permission_evidence


class CapabilityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.probe, self.proc = self.root / 'legacy', self.root / 'proc'
        self.probe.mkdir(); (self.proc / 'driver/nvidia').mkdir(parents=True)
        (self.proc / 'self').mkdir()
        (self.proc / 'driver/nvidia/params').write_text('RmProfilingAdminOnly: 1\n')
        (self.proc / 'driver/nvidia/version').write_text('synthetic-version\n')
        (self.proc / 'self/status').write_text('Uid:\t0\t0\t0\t0\nCapEff:\t0000000000000000\n')
        device = self.proc / 'driver/nvidia/gpus/synthetic/information'
        device.parent.mkdir(parents=True); device.write_text('Model: synthetic-device\nDevice Minor: 0\n')
        c = sqlite3.connect(self.probe / 'tiny-cuda-nvtx.sqlite')
        c.executescript('''
          create table StringIds(id integer,value text);
          create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
          create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer);
          create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,demangledName integer,
            globalPid integer,correlationId integer,deviceId integer,streamId integer,
            registersPerThread integer,staticSharedMemory integer,dynamicSharedMemory integer,
            blockX integer,blockY integer,blockZ integer,gridX integer,gridY integer,gridZ integer);
          create table TARGET_INFO_CUDA_DEVICE(gpuId integer,numMultiprocessors integer);
          insert into StringIds values(1,'synthetic_kernel');
          insert into CUPTI_ACTIVITY_KIND_KERNEL values(10,20,1,16777216,1,0,1,32,0,0,128,1,1,128,1,1);
          insert into TARGET_INFO_CUDA_DEVICE values(0,2);
        ''')
        c.commit(); c.close()
        (self.probe / 'tiny-counters.csv').write_text('ERR_NVGPUCTRPERM\n')
        (self.probe / 'nsys.stdout.log').write_text('{"gpu":"synthetic-device","shape":[2,2]}\n')
        (self.probe / 'parsed-kernels.json').write_text('{"legacy":true}\n')
        receipt = dict(schema='gsdeploy.hardware_probe.v1', commands=[
            dict(name='nsys', returncode=0, command=['/nonexistent/nsys', 'profile']),
            dict(name='ncu', returncode=1, command=['/nonexistent/ncu', '--metrics', 'compute,dram'])],
            files=[dict(path=str(self.probe / 'tiny-counters.csv'),
                        sha256=hashlib.sha256((self.probe / 'tiny-counters.csv').read_bytes()).hexdigest())])
        (self.probe / 'receipt.json').write_text(json.dumps(receipt))

    def run_audit(self, output=None):
        with (patch('subprocess.Popen', side_effect=AssertionError('No process execution')),
              patch('subprocess.run', side_effect=AssertionError('No profiler/device query execution'))):
            return audit(output or self.root / 'new', probe_directory=self.probe, proc_root=self.proc)

    def test_reparse_preserves_geometry_missing_allocation_and_legacy_bytes(self):
        before = {p.name: p.read_bytes() for p in self.probe.iterdir()}
        result = self.run_audit()
        self.assertEqual(result['status'], 'CPU_READ_ONLY_AUDIT_COMPLETE')
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.probe.iterdir()})
        self.assertTrue(result['legacy_files_unchanged'])
        self.assertEqual(result['historical_source_integrity'], 'MATCH')
        self.assertEqual(result['kernel_static_resources']['observed_geometry_counts'], dict(block=1, grid=1))
        self.assertEqual(result['kernel_static_resources']['architecture_allocation_missing_indices'], [0])
        self.assertEqual(result['kernel_static_resources']['occupancy_status'], 'UNKNOWN')
        self.assertFalse(result['GPU_started']); self.assertFalse(result['permissions_changed'])
        self.assertFalse(result['physically_validated'])
        self.assertEqual(result['actual_model_baselines']['status'], 'UNKNOWN')
        parsed = json.loads((self.root / 'new/reparsed-kernels.json').read_text())
        self.assertEqual(parsed['unattributed_kernels'][0]['block'], [128, 1, 1])
        self.assertIsNone(parsed['unattributed_kernels'][0]['allocated_registers_per_block'])

    def test_admin_restriction_and_missing_capability_explain_old_denial_without_reprobe(self):
        result = self.run_audit()
        self.assertEqual(result['historical_nsys']['returncode'], 0)
        self.assertEqual(result['historical_ncu']['status'], 'HISTORICAL_PERMISSION_DENIED')
        self.assertEqual(result['historical_ncu']['requested_metrics'], ['compute', 'dram'])
        permission = result['permission_evidence']
        self.assertFalse(permission['cap_sys_admin_effective'])
        self.assertEqual(permission['status'], 'ADMIN_PROFILING_RESTRICTION_WITHOUT_CAP_SYS_ADMIN')
        self.assertEqual(permission['counter_access_current'], 'UNKNOWN_NOT_REPROBED')
        self.assertEqual(result['device_queries_executed'], [])

    def test_missing_proc_metadata_never_implies_permission_or_zero_restriction(self):
        (self.proc / 'driver/nvidia/params').unlink(); (self.proc / 'self/status').unlink()
        result = self.run_audit()
        self.assertEqual(result['permission_evidence']['status'], 'UNKNOWN_PERMISSION_METADATA')
        self.assertIsNone(result['permission_evidence']['rm_profiling_admin_only'])
        self.assertIsNone(result['permission_evidence']['cap_sys_admin_effective'])

    def test_capability_presence_is_not_a_successful_access_test(self):
        result = permission_evidence('RmProfilingAdminOnly: 1\n', 'CapEff: 200000\n')
        self.assertTrue(result['cap_sys_admin_effective'])
        self.assertEqual(result['counter_access_current'], 'UNKNOWN_NOT_REPROBED')
        self.assertEqual(result['status'], 'METADATA_DOES_NOT_ESTABLISH_COUNTER_ACCESS')

    def test_existing_output_or_legacy_subdirectory_is_rejected_before_mutation(self):
        output = self.root / 'occupied'; output.mkdir(); (output / 'keep').write_text('unchanged')
        with self.assertRaises(FileExistsError): self.run_audit(output)
        self.assertEqual((output / 'keep').read_text(), 'unchanged')
        with self.assertRaises(ValueError): self.run_audit(self.probe / 'new')
        self.assertFalse((self.probe / 'new').exists())

    def test_cli_accepts_only_new_output_and_manifest_hashes_match(self):
        output = self.root / 'cli-output'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--output', str(output), '--probe-directory', str(self.probe),
                                   '--proc-root', str(self.proc)]), 0)
        manifest = json.loads((output / 'output-manifest.json').read_text())
        for entry in manifest['files']:
            self.assertEqual(hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest(), entry['sha256'])
        with self.assertRaises(FileExistsError):
            main(['--output', str(output), '--probe-directory', str(self.probe), '--proc-root', str(self.proc)])


if __name__ == '__main__': unittest.main()
