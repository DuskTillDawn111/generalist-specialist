"""Read-only common-host-clock GPU frequency audit; not model training counters."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--stop-file',required=True,type=Path);p.add_argument('--seconds',type=float,default=21600)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    query='name,uuid,clocks.current.graphics,clocks.current.sm,clocks.current.memory,temperature.gpu,pstate,power.draw'
    end=time.monotonic()+args.seconds
    with args.output.open('x',buffering=1) as output:
        output.write(json.dumps(dict(schema='gsdeploy.native_clock_audit.v1',kind='measured',
            scope='read-only nvidia-smi frequency/temperature; external descriptive audit, no SM occupancy/DRAM/L2 claim',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),query=query,
            start_perf_ns=time.perf_counter_ns(),start_unix_ns=time.time_ns()))+'\n')
        while time.monotonic()<end and not args.stop_file.exists():
            a=time.perf_counter_ns()
            result=subprocess.run(['nvidia-smi','--query-gpu='+query,'--format=csv,noheader,nounits'],
                                  capture_output=True,text=True,timeout=10)
            b=time.perf_counter_ns()
            if result.returncode:
                output.write(json.dumps(dict(time_ns=(a+b)//2,error=result.stderr))+'\n');return 1
            values=[s.strip() for s in result.stdout.strip().split(',')]
            if len(values)!=8:raise ValueError('Single GPU audit query required')
            name,uuid,graphics,sm,memory,temp,pstate,power=values
            output.write(json.dumps(dict(time_ns=(a+b)//2,query_start_ns=a,query_end_ns=b,
                name=name,uuid=uuid,graphics_mhz=float(graphics),sm_mhz=float(sm),memory_mhz=float(memory),
                temperature_celsius=float(temp),pstate=pstate,board_watts=float(power)))+'\n')
            time.sleep(1)
    return 0


if __name__=='__main__':raise SystemExit(main())
