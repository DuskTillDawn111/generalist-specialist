"""Conditionally diagnose unresolved heap-trim repeats after the complete ABBA.

The raw repeat statistic is only a diagnostic scheduling trigger. It neither
rejects nor qualifies a prediction model; original training/validation gates
remain necessary. The profiled domain is the original untrimmed host capture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import save


def unresolved_trim_repeats(report):
    if report.get('schema') != 'gsdeploy.host_trim_runtime_diagnostic.v1':
        raise ValueError('Complete heap comparison analysis required')
    if report['issues'] or report['output_changes']:
        raise ValueError('Investigate capture/output issues before another mechanism probe')
    bands = report['bands']
    if {b['mode'] for b in bands} != {'off', 'on'}:
        raise ValueError('Both fully repeated runtime arms required')
    return [b for b in bands if b['mode'] == 'on' and b['raw_constant_minimax'] > .2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--upstream-base', type=Path, required=True)
    parser.add_argument('--upstream-pid', type=int, required=True)
    args = parser.parse_args()
    base, upstream = args.base.resolve(), args.upstream_base.resolve()
    path = base/'followup-state.json'
    if path.exists() or (base/'state.json').exists():
        raise FileExistsError('Do not restart an existing profiler continuation')
    design = json.loads((base/'design.json').read_text())
    upstream_design = json.loads((upstream/'design.json').read_text())
    state = dict(status='waiting_for_complete_host_trim_ABBA', physical_predictor_qualified=False,
                 formal_utility_qualified=False)
    plan = dict(schema='gsdeploy.minimal_cupti_followup.v1', upstream_base=str(upstream),
        upstream_pid=args.upstream_pid,
        upstream_design_sha256=hashlib.sha256((upstream/'design.json').read_bytes()).hexdigest(),
        diagnostic_trigger='At least one trim-arm repeated input band has raw constant minimax > .2; not root-order-adjusted and not a qualification gate.',
        profiled_domain='Original untrimmed five-model host trace; exact 320-request history.',
        current_ABBA_must_finish=True, repeat_failed_capture_unchanged=False,
        physical_predictor_qualified=False, formal_utility_qualified=False)
    save(base/'followup-plan.json', plan)
    save(path, state)

    def check():
        if hashlib.sha256((upstream/'design.json').read_bytes()).hexdigest() != plan['upstream_design_sha256']:
            raise ValueError('Upstream design changed')
        for receipt in design['source_receipts'] + upstream_design['source_receipts']:
            if any(hashlib.sha256(Path(receipt[k]).read_bytes()).hexdigest() != receipt['sha256']
                   for k in ('path', 'snapshot')):
                raise ValueError('Frozen source changed: '+receipt['path'])

    try:
        check()
        while True:
            current = json.loads((upstream/'state.json').read_text())
            if current['status'] == 'host_trim_comparison_completed':
                break
            if 'failed' in current['status']:
                raise ValueError('Upstream comparison failed; preserve and inspect it first')
            proc = Path('/proc')/str(args.upstream_pid)/'cmdline'
            try:
                command = proc.read_bytes()
            except FileNotFoundError:
                command = b''
            if b'continue_activation_host_trim_comparison.py' not in command:
                current = json.loads((upstream/'state.json').read_text())
                if current['status'] == 'host_trim_comparison_completed':
                    break
                raise RuntimeError('Upstream process exited before complete comparison analysis')
            time.sleep(2)
        check()
        content = (upstream/'description.json').read_bytes()
        report = json.loads(content)
        expected = {name: str(upstream/name/'raw.json')+':sha256:'+hashlib.sha256(
            (upstream/name/'raw.json').read_bytes()).hexdigest()
            for name in ('control-round1', 'trim-round1', 'trim-round2', 'control-round2')}
        if report['sources'] != expected:
            raise ValueError('Complete four-capture raw receipts differ from trigger analysis')
        unresolved = unresolved_trim_repeats(report)
        state.update(upstream_report_sha256=hashlib.sha256(content).hexdigest(),
                     diagnostic_trigger_bands=unresolved)
        if not unresolved:
            state['status'] = 'not_scheduled_no_raw_repeat_trigger'
            save(path, state)
            return
        state['status'] = 'minimal_cupti_probe_running'
        save(path, state)
        subprocess.run([sys.executable, str(ROOT/'experiments/run_activation_cupti_host_probe.py'),
                        '--base', str(base)], cwd=ROOT, check=True)
        check()
        state['status'] = 'minimal_cupti_probe_completed'
        save(path, state)
    except BaseException as exc:
        state.update(status='minimal_cupti_followup_failed', error=type(exc).__name__+': '+str(exc))
        save(path, state)
        raise


if __name__ == '__main__':
    main()
