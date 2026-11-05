"""Describe measured kernel-time variation inside legacy pilot native calls.

Timestamp intersections are not per-launch ownership. Profiler timings do not
replace natural-repeat latency or create device utilization counters.
"""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();base=args.base.resolve()
    anchors=json.loads((base/'pilot-trace-01-anchors.json').read_text())
    summary=json.loads((base/'pilot-trace-01-summary.json').read_text())
    raw=json.loads((base/'pilot-profiled-01/raw.json').read_text())
    rows={r['request_id']:r for r in raw['request_records']}
    phases=defaultdict(list)
    for a in anchors:
        if a['kind']!='gpu':continue
        mapping=summary['clock_mappings'][str(a['pid'])]
        if not mapping['valid']:raise ValueError('Verified host/profiler clock map required')
        lo=a['start_ns']-mapping['offset_ns'];hi=a['end_ns']-mapping['offset_ns']
        row=rows[a['request_id']]
        phases[a['pid']].append(dict(request_id=a['request_id'],alias=row['model'],task=row['task'],
            pool_id=row['pool_id'],group=row['profile_group'],start_ns=lo,end_ns=hi,
            bins=[dict(kernel_intersections=0,kernel_sum_ns=0,busy_union_ns=0,last_end_ns=-1,names=Counter()) for _ in range(10)]))
    for values in phases.values():values.sort(key=lambda r:r['start_ns'])
    starts={pid:[r['start_ns'] for r in values] for pid,values in phases.items()}
    path=base/'pilot-trace-01.sqlite'
    connection=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    strings=dict(connection.execute('select id,value from StringIds'))
    kernels=0;intersected=0
    for lo,hi,global_pid,name in connection.execute(
            'select start,end,globalPid,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start,end'):
        kernels+=1;pid=(global_pid>>24)&0xffffff
        if pid not in phases:continue
        values=phases[pid];index=bisect_right(starts[pid],lo)-1
        selected=values[max(0,index):min(len(values),index+2)]
        hit=False
        for phase in selected:
            if hi<=phase['start_ns'] or lo>=phase['end_ns']:continue
            hit=True;duration=phase['end_ns']-phase['start_ns']
            for i,bin in enumerate(phase['bins']):
                a=max(lo,phase['start_ns']+round(duration*i/10))
                b=min(hi,phase['start_ns']+round(duration*(i+1)/10))
                if b<=a:continue
                bin['kernel_intersections']+=1;bin['kernel_sum_ns']+=b-a
                bin['busy_union_ns']+=max(0,b-max(a,bin['last_end_ns']))
                bin['last_end_ns']=max(bin['last_end_ns'],b);bin['names'][strings.get(name,str(name))]+=b-a
        intersected+=hit
    connection.close();result=[]
    for pid,values in phases.items():
        for phase in values:
            for i,bin in enumerate(phase['bins']):
                duration=round((phase['end_ns']-phase['start_ns'])*(i+1)/10)-round((phase['end_ns']-phase['start_ns'])*i/10)
                bin.update(relative_start=i/10,relative_end=(i+1)/10,
                    kernel_busy_fraction=bin['busy_union_ns']/duration,
                    top_kernel_names_by_duration=bin['names'].most_common(3))
                del bin['last_end_ns'];del bin['names']
            result.append(dict(pid=pid,**phase))
    report=dict(schema='gsdeploy.pilot_kernel_shape_description.v1',kernels=kernels,
                kernels_intersecting_native_host_windows=intersected,phases=result,
                clock_summary_source=str(base/'pilot-trace-01-summary.json'),
                sqlite_source=str(path),sqlite_sha256=json.loads((base/'pilot-trace-01-export.json').read_text())['output']['sha256'],
                analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='measured CUPTI interval intersections inside legacy host native-call windows; profiler diagnostic only',
                causal_predictor=False,phase_templates_qualified=False,device_demand_counters_measured=False)
    with args.output.open('x') as out:json.dump(report,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps(dict(output=str(args.output),kernels=kernels,phases=len(result))))


if __name__=='__main__':main()
