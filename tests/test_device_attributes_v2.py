"""Fake driver-library tests: no CUDA calls are made by these fixtures."""
import ctypes
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import device_attributes_v2 as module


class Function:
    def __init__(self, function): self.function = function
    def __call__(self, *args): return self.function(*args)


class FakeCuda:
    def __init__(self, *, fail_init=False, failed_attribute=None, odd_warps=False):
        self.calls, self.fail_init, self.failed_attribute = [], fail_init, failed_attribute
        self.values = {1: 1024, 8: 49152, 10: 32, 12: 65536, 16: 2,
                       39: 1025 if odd_warps else 1024, 75: 8, 76: 0, 81: 65536,
                       82: 32768, 97: 65536, 106: 8, 111: 1024}
        self.functions = {name: Function(lambda *args, name=name: self.call(name, *args))
                          for name in module.ALLOWED_APIS}

    def __getattr__(self, name):
        if name not in self.functions: raise AssertionError('Disallowed API resolved: ' + name)
        return self.functions[name]

    def call(self, name, *args):
        self.calls.append(name)
        def integer(pointer, value): ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = value
        if name == 'cuInit': return 999 if self.fail_init else 0
        if name == 'cuDriverGetVersion': integer(args[0], 13000)
        elif name == 'cuDeviceGetCount': integer(args[0], 1)
        elif name == 'cuDeviceGet': integer(args[0], 100 + args[1])
        elif name == 'cuDeviceGetName':
            assert args[2] == 100
            ctypes.memmove(args[0], b'fake-device\x00', 12)
        elif name == 'cuDeviceGetUuid':
            assert args[1] == 100
            target = ctypes.cast(args[0], ctypes.POINTER(module.CUuuid)).contents
            for index in range(16): target.bytes[index] = index + 1
        elif name == 'cuDeviceGetAttribute':
            assert args[2] == 100
            if args[1] == self.failed_attribute: return 801
            integer(args[0], self.values[args[1]])
        return 0


class DeviceAttributesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.header = self.root / 'cuda.h'
        self.header.write_text('\n'.join(f'{name} = {value},' for name, value in module.ATTRIBUTES.values()))
        self.trace = self.root / 'legacy/trace.sqlite'; self.trace.parent.mkdir()
        c = sqlite3.connect(self.trace)
        c.execute('create table TARGET_INFO_CUDA_DEVICE(cudaId integer,gpuId integer,uuid blob)')
        c.execute('insert into TARGET_INFO_CUDA_DEVICE values(0,0,null)'); c.commit(); c.close()

    def test_exact_limits_derive_warps_and_record_only_allowlisted_api_calls(self):
        fake = FakeCuda(); result = module.query_attributes(fake)
        self.assertTrue(result['driver_initialized'])
        self.assertEqual(result['driver_api_version'], 13000)
        device = result['devices'][0]
        self.assertEqual(device['hardware']['max_warps_per_sm'], 32)
        self.assertEqual(device['hardware']['hardware_id'], 'GPU-01020304-0506-0708-090a-0b0c0d0e0f10')
        self.assertEqual(device['attributes']['compute_capability_minor']['value'], 0)
        self.assertEqual(device['handle'], 100)  # not ordinal0
        self.assertEqual(set(fake.calls), set(module.ALLOWED_APIS))
        self.assertFalse(result['context_creation_requested'])
        self.assertEqual(result['kernel_launches_requested'], 0)
        self.assertTrue(all(row['returncode'] == 0 for row in result['api_calls']))

    def test_failed_attribute_or_nonintegral_warp_derivation_keeps_hardware_unknown(self):
        for fake in (FakeCuda(failed_attribute=82), FakeCuda(odd_warps=True)):
            device = module.query_attributes(fake)['devices'][0]
            self.assertIsNone(device['hardware'])
            self.assertEqual(device['hardware_limits_status'], 'UNKNOWN')
        result = module.query_attributes(FakeCuda(failed_attribute=82))
        self.assertEqual(result['devices'][0]['attributes']['registers_per_sm']['returncode'], 801)

    def test_init_failure_stops_before_device_queries(self):
        fake = FakeCuda(fail_init=True)
        result = module.query_attributes(fake)
        self.assertEqual(fake.calls, ['cuInit'])
        self.assertEqual(result['status'], 'UNKNOWN_DRIVER_INITIALIZATION_FAILED')

    def test_trace_uuid_must_match_exactly_without_ordinal_fallback(self):
        devices = module.query_attributes(FakeCuda())['devices']
        result = module.trace_uuid_mapping(self.trace, devices)
        self.assertIsNone(result['rows'][0]['queried_ordinal'])
        c = sqlite3.connect(self.trace)
        c.execute('update TARGET_INFO_CUDA_DEVICE set uuid=?', (bytes(range(1, 17)),)); c.commit(); c.close()
        self.assertEqual(module.trace_uuid_mapping(self.trace, devices)['rows'][0]['status'], 'EXACT_UUID_MATCH')
        c = sqlite3.connect(self.trace)
        c.execute('update TARGET_INFO_CUDA_DEVICE set uuid=?', (bytes(range(2, 18)),)); c.commit(); c.close()
        self.assertIsNone(module.trace_uuid_mapping(self.trace, devices)['rows'][0]['queried_ordinal'])

    def test_header_mismatch_prevents_library_loading(self):
        self.header.write_text(self.header.read_text().replace('WARP_SIZE = 10', 'WARP_SIZE = 11'))
        with patch.object(module.ctypes, 'CDLL', side_effect=AssertionError('Header must be verified first')):
            with self.assertRaises(ValueError):
                module.audit(self.root / 'bad', header=self.header, trace=self.trace)
        self.assertFalse((self.root / 'bad').exists())

    def test_audit_fake_cdll_refuses_overwrite_and_preserves_old_trace(self):
        original = self.trace.read_bytes(); fake = FakeCuda()
        with patch('subprocess.Popen', side_effect=AssertionError('No processes')):
            result = module.audit(self.root / 'new', header=self.header, trace=self.trace,
                                  library_loader=lambda name: fake)
        self.assertEqual(self.trace.read_bytes(), original)
        self.assertTrue(result['historical_trace_unchanged'])
        self.assertFalse(result['gpu_experiment_started'])
        self.assertEqual(result['occupancy_status'], 'UNKNOWN_NO_KERNEL_ALLOCATION_EVIDENCE')
        self.assertTrue(result['query']['devices'][0]['hardware']['evidence'])
        self.assertEqual(json.loads((self.root / 'new/device-attributes.json').read_text())['schema'], module.SCHEMA)
        with self.assertRaises(FileExistsError):
            module.audit(self.root / 'new', header=self.header, trace=self.trace, library_loader=lambda name: fake)


if __name__ == '__main__': unittest.main()
