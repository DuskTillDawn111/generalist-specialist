"""Read-only preservation of temporary logs from our owned profiled worker.

Does not signal workers, modify profiler settings, or acquire the GPU. Logs
removed before discovery cannot be recovered; this is an auxiliary evidence
collector, never a claim that profiling succeeded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.collect_activation_training import save
from tools.predictor_validation_20261027.native_collect import descendant_or_self


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-base', type=Path, required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, output = args.probe_base.resolve(), args.output.resolve()
    design = json.loads((base/'design.json').read_text())
    original = json.loads((Path(design['unprofiled_capture'])/'context.json').read_text())
    command = original['worker_bindings'][design['profiled_alias']]['command']
    spec = command[command.index('--spec')+1]
    proc = Path('/proc')/str(args.controller_pid)/'cmdline'
    if b'continue_activation_minimal_cupti_probe.py' not in proc.read_bytes():
        raise ValueError('Live owned continuation controller required')
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (output/'source.py').write_bytes(source)
    state = dict(status='watching_owned_profiler_worker_logs', probe_base=str(base),
        controller_pid=args.controller_pid, worker_spec=spec, tracked={}, errors=[],
        source_sha256=hashlib.sha256(source).hexdigest(), physical_predictor_qualified=False,
        scope='Read-only auxiliary copy of append-only temporary stdout/stderr logs. Only descendants carrying the exact profiled worker spec are discovered. Removed-before-discovery logs may be unavailable.')
    known_pids = set()
    root = Path('/tmp/nvidia/nsight_systems')

    def scan():
        for path in root.glob('quadd_session_*/streams/pid_*_*.log'):
            match = re.fullmatch(r'pid_(\d+)_(stdout|stderr)\.log', path.name)
            if not match:
                continue
            pid = int(match[1])
            if pid not in known_pids:
                try:
                    cmd = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
                except FileNotFoundError:
                    continue
                if spec.encode() not in cmd or not descendant_or_self(pid, args.controller_pid):
                    continue
                known_pids.add(pid)
            key = str(path)
            entry = state['tracked'].setdefault(key, dict(pid=pid, bytes=0, generation=0, copies=[]))
            try:
                size = path.stat().st_size
                if size < entry['bytes']:
                    entry['generation'] += 1
                    entry['bytes'] = 0
                target = output/(path.parent.parent.name+'-'+path.name+'.g'+str(entry['generation']))
                with path.open('rb') as stream:
                    stream.seek(entry['bytes'])
                    chunk = stream.read()
                if chunk:
                    with target.open('ab') as stream:
                        stream.write(chunk)
                    entry['bytes'] += len(chunk)
                    if str(target) not in entry['copies']:
                        entry['copies'].append(str(target))
                    entry['last_copy_perf_ns'] = time.perf_counter_ns()
            except FileNotFoundError:
                continue  # Nsight may clean temporary streams at exit.

    try:
        while True:
            scan()
            current = json.loads((base/'followup-state.json').read_text())
            state['followup_status'] = current['status']
            save(output/'state.json', state)
            terminal = (current['status'] in ('minimal_cupti_probe_completed', 'not_scheduled_no_raw_repeat_trigger')
                        or 'failed' in current['status'])
            if terminal:
                scan()
                state['status'] = 'preservation_finished'
                break
            if not proc.exists():
                # A missing process is terminal evidence for this auxiliary
                # watcher; the GPU pipeline's own state remains untouched.
                scan()
                state['status'] = 'preservation_controller_exited'
                break
            time.sleep(2)
    except BaseException as exc:
        state['status'] = 'preservation_failed'
        state['errors'].append(type(exc).__name__+': '+str(exc))
        raise
    finally:
        state['receipts'] = [dict(path=name, bytes=Path(name).stat().st_size,
            sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest())
            for entry in state['tracked'].values() for name in entry['copies']]
        save(output/'state.json', state)


if __name__ == '__main__':
    main()
