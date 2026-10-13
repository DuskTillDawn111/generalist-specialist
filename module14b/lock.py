"""Execution/environment lock after CPU compatibility, before GPU results."""
import subprocess
from module14b.common import *
import pynvml as nv


def main():
    assert not (OUT / 'module14b_execution_freeze.json').exists()
    assert not list(RAW.glob('*summary.json'))
    pre = read(OUT / 'module14b_cpu_preflight.json')
    downloads = read(OUT / 'module14b_download_manifest.json')
    assert pre['completed'] and downloads['completed']
    for m in downloads['models']:
        if m['success']: assert ref(m['file']['path'])['sha256'] == m['file']['sha256']
    discovery = read(OUT / 'module14b_discovery_freeze.json')
    for r in discovery['files'] + discovery['native_source']:
        assert ref(r['path'])['sha256'] == r['sha256'], r['path']
    # Reuse exact frozen quality inputs and latency rows; no new subset.
    oldinputs = read(OUT / 'dataset_input_hashes.json')['files']
    proto = read(OUT / 'ocr_gapfill_shortlist.json')
    d = proto['datasets']['ocr']
    for r in [d['quality_manifest'], d['latency_sample_file']] + oldinputs:
        assert ref(r['path'])['sha256'] == r['sha256'], r['path']
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    hardware = {'name': nv.nvmlDeviceGetName(h), 'uuid': nv.nvmlDeviceGetUUID(h),
                'total_vram_mib': nv.nvmlDeviceGetMemoryInfo(h).total / 1048576,
                'driver': nv.nvmlSystemGetDriverVersion(),
                'processes': [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h)]}
    assert not hardware['processes']
    nv.nvmlShutdown()
    runtimes = {}
    for name, python in [('new_STR_overlay', PYTHON), ('original_Paddle_control', Path('/root/miniconda3/envs/ocr_env/bin/python'))]:
        r = subprocess.run([str(python), '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True, check=True)
        runtimes[name] = {'python': str(python), 'pip_freeze_all': r.stdout.splitlines()}
    save('module14b_local_environment.json', {'at_utc': now(), 'hardware': hardware,
        'runtimes': runtimes, 'CPU_preflight': pre, 'old_runtime_packages_modified': False,
        'new_environment_isolated_overlay': True, 'dtype': 'float32', 'standard24GB_device_claim': False})
    code = [ref(p) for p in sorted((ROOT / 'module14b').glob('*.py'))]
    code += [ref(ROOT / path) for path in ['module13b/campaign.py', 'module13b/analyze.py',
             'module14/common.py', 'module14/analyze.py', 'joint_deploy/worker_protocol.py',
             'joint_deploy/ocr_worker.py']]
    save('module14b_execution_freeze.json', {
        'at_utc': now(), 'before_any_new_OCR_GPU_measurement': True,
        'implementation': code, 'native_source': discovery['native_source'],
        'protocol': ref(OUT / 'ocr_gapfill_shortlist.json'),
        'environment': ref(OUT / 'module14b_local_environment.json'),
        'downloads': ref(OUT / 'module14b_download_manifest.json'),
        'inputs': [d['quality_manifest'], d['latency_sample_file']] + oldinputs,
        'selection_before_deployment': True, 'third_OCR_round_allowed': False})
    print('EXECUTION_LOCKED', hardware, 'CPU success', pre['all_success'], flush=True)


if __name__ == '__main__': main()
