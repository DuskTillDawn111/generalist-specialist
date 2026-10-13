"""Finish analysis after the original campaign exits; never launch GPU windows."""
from .common import *
import subprocess
import sys


def main():
    begin = time.monotonic()
    while True:
        status = read(OUT / 'campaign_status.json')
        if status['status'] != 'RUNNING':
            break
        if time.monotonic() - begin > 3600:
            raise RuntimeError('Original campaign did not complete within the continuation monitoring bound')
        time.sleep(2)
    if status['status'] != 'COMPLETED':
        raise RuntimeError(('Original frozen campaign stopped', status))
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': 'results/module34_candidate_pool:.',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    steps = [('analyze_gpu', []), ('method_gpu_analysis', []), ('reuse33_independent', []),
             ('diagnostics', []), ('history', ['after']), ('report', []),
             ('plot_results', []), ('continuation_audit', []), ('closeout', []), ('final_audit', [])]
    for name, args in steps:
        print('CLOSEOUT_START', name, flush=True)
        save('continuation_closeout_status.json', dict(status='RUNNING', step=name))
        with (OUT / (name + '_continuation.log')).open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'm34.' + name, *args],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save('continuation_closeout_status.json', dict(status='FAILED', step=name,
                 exit_code=result.returncode, log=str(log.name)))
            raise RuntimeError(('Closeout failed', name, result.returncode))
        print('CLOSEOUT_DONE', name, flush=True)
    save('continuation_closeout_status.json', dict(status='COMPLETED', no_new_GPU_windows=True,
         frozen_campaign_total=24, final_audit=read(OUT / 'final_audit.json')['passed']))
    print('MODULE34_FINISHED', flush=True)


if __name__ == '__main__':
    main()
