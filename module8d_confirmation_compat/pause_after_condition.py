"""Pause only the owned coordinator after a completed, fully released condition."""
import argparse,ctypes,os,select,signal,struct,time
from pathlib import Path
from module8d_real_baselines.common import OUT,read,save_json,ref


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--condition',required=True)
    args=parser.parse_args();pid=args.pid;directory=OUT/args.condition;directory.mkdir(parents=True,exist_ok=True)
    target=directory/'status.json'
    if target.exists():raise RuntimeError('Condition already complete; do not pause during a following measurement')
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK)
    assert fd>=0
    wd=libc.inotify_add_watch(fd,os.fsencode(directory),0x80) # IN_MOVED_TO, atomic save_json
    assert wd>=0
    while not target.exists():
        select.select([fd],[],[],30)
        try:os.read(fd,65536)
        except BlockingIOError:pass
    # The condition saves status only after workers exited, GPU released, raw
    # validated and both confirmations aggregated. Pause before next job gate.
    os.kill(pid,signal.SIGSTOP)
    last=read(OUT/'new_measurement_manifest.json')['runs'][-1]
    if last.get('status')=='started':
        os.kill(pid,signal.SIGCONT)
        raise RuntimeError('Missed safe coordinator boundary; measurement allowed to continue')
    from replica_scaling.telemetry import snapshot
    gpu=snapshot()
    if gpu['processes'] or gpu['used_mib']>128:
        os.kill(pid,signal.SIGCONT)
        raise RuntimeError('GPU not fully released; coordinator resumed without interruption')
    save_json(OUT/'coordinator_pause_record.json',{'status':'paused_between_complete_conditions',
        'pid':pid,'condition':args.condition,'completed_condition_status':read(target),'gpu_idle':gpu,
        'last_physical_run':last,'timed_measurements_interrupted':0,'new_physical_attempts':len(read(OUT/'new_measurement_manifest.json')['runs']),
        'pause_source':ref(Path(__file__).resolve()),'unix':time.time()})
    print('COORDINATOR_PAUSED_AT_COMPLETE_CONDITION; GPU_IDLE; timed interruptions=0',flush=True)


if __name__=='__main__':main()
