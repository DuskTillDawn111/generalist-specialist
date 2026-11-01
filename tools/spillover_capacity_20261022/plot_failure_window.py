#!/usr/bin/env python3
"""Plot signed deadline slices only, including clipped in-flight handler context."""
import importlib.util
import json
from pathlib import Path
SPEC=importlib.util.spec_from_file_location('_failure_window_plot_shared',Path(__file__).with_name('plot_sustained.py'))
plot=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(plot)


def load(directory):
    directory=Path(directory).resolve();manifest=json.loads((directory/'manifest.json').read_text())
    def read(name):
        if Path(name).name!=name or name not in manifest['files'] or plot.sha(directory/name)!=manifest['files'][name]:raise ValueError('Failure slice receipt differs')
        return json.loads((directory/name).read_text())
    report=read('report.json')
    if report.get('schema')!='gsdeploy.sustained_failure_window.v1':raise ValueError('Expected failure-window report')
    jobs=[]
    for item in report['jobs']:
        detail=read(item['detail_file']);deadline=detail['deadline'];origin=deadline['cutoff_unix_ns']-int(deadline['duration_seconds']*1e9)
        intervals=[dict(alias=r['alias'],role=r['role'],request_id=r['request_id'],start_seconds=(r['start_ns']-origin)/1e9,
            duration_seconds=(r['end_ns']-r['start_ns'])/1e9) for r in detail['observed_handler_intervals_clipped_to_deadline']]
        jobs.append(dict(job_id=detail['job_id'],state=detail['state'],status=detail['status'],active_aliases=detail['active_aliases'],
            intervals=intervals,primary_window_seconds=deadline['duration_seconds'] if detail['raw_status']=='completed' else None,
            measurement_seconds=deadline['duration_seconds'],support_drain_seconds=None,coverage_status=detail['coverage']['status'],
            service_unknown_primary=sum(v['primary']['service_unknown_requests'] for v in detail['aliases'].values()),
            service_unknown_support=sum(v['support']['service_unknown_requests'] for v in detail['aliases'].values())))
    pairs=[]
    for row in report['paired_primary_repetitions']:
        if row['alias']=='generalist_2#0':pairs.append(dict(joint_job=row['joint_job'],pool_id=row['pool_id'],repetition=row['repetition'],
            joint_over_baseline=row['joint_over_baseline'],common_handler_fraction=(row.get('joint_activity') or {}).get('all_active_aliases_busy_fraction'),
            prediction_same=row['prediction_same'],generated_tokens_same=row['generated_tokens_same']))
    return dict(status='INSUFFICIENT_DEADLINE_QUALIFIED_PARTIAL_DIAGNOSTIC',all_four_terminal=False,jobs=jobs,generalist_pairs=pairs,
        source_analysis=dict(path=str(directory),report_sha256=manifest['files']['report.json'],manifest_sha256=plot.sha(directory/'manifest.json')),
        independent_inputs_per_alias=32,raw_files_read=0,U=None,
        scope='Deadline-qualified completed primary only for service points; timeline includes clipped observed in-flight handler context. Missing repeats remain UNKNOWN.')


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--failure-window',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=load(args.failure_window);plot.render(data,args.output)
    (args.output/'INTERPRETATION.txt').write_text('FAILED PARTIAL DEVELOPMENT DIAGNOSTIC.\nTimeline shows observed handler activity clipped at600s; visible in-flight fragments do not prove completed requests.\nService scatter includes only76 G2 primary responses received within cutoff, not the77 terminal responses.\nThe additional1 G2 primary and2 specialist support completions after cutoff are UNKNOWN for measurement-qualified service/quality/coverage.\n19 G2 primary were unreleased; second block never started. No complete3-repeat or cross-block validation claim.\nMemory full-window coverage remains UNKNOWN due independent-audit tail gap. U=null.\n')
    manifest=json.loads((args.output/'manifest.json').read_text());manifest['failure_plot_tool']={'path':str(Path(__file__).resolve()),'sha256':plot.sha(__file__)}
    manifest['files']['INTERPRETATION.txt']=plot.sha(args.output/'INTERPRETATION.txt');(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=data['status'],output=str(args.output.resolve()),qualified_G2_pairs=sum(r['joint_over_baseline'] is not None for r in data['generalist_pairs']),raw_files_read=0,U=None)))


if __name__=='__main__':main()
