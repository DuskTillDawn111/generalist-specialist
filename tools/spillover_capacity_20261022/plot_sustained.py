#!/usr/bin/env python3
"""Render only published sustained analysis; never open GPU raw/live artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plot_data(directory):
    """Validate analysis-owned file receipts, not the raw paths it references."""
    directory=Path(directory).resolve();manifest=json.loads((directory/'manifest.json').read_text())
    names=manifest.get('files') or {}
    if 'report.json' not in names:raise ValueError('Published report receipt missing')
    def read(name):
        if Path(name).name!=name or not name.endswith('.json') or name not in names:
            raise ValueError('Only analysis-owned registered JSON basenames accepted')
        path=directory/name
        if path.resolve().parent!=directory or sha(path)!=names[name]:raise ValueError('Analysis file receipt differs')
        return json.loads(path.read_text())
    report=read('report.json')
    if report.get('schema')!='gsdeploy.sustained_capacity_analysis.v1' or report.get('U') is not None:
        raise ValueError('Expected sustained diagnostic analysis with U=null')
    jobs=[]
    for item in report.get('jobs',[]):
        detail=read(item['detail_file'])
        if detail.get('schema')!='gsdeploy.sustained_capacity_analysis.job.v1' or detail['job_id']!=item['job_id']:
            raise ValueError('Analysis job identity differs')
        window=detail['measurement_timing']['measurement_window'];rows=detail['request_diagnostics']
        observed=[r for r in rows if r['service_status']=='AVAILABLE']
        origin=window.get('start_unix_ns')
        if type(origin) is not int:origin=min((r['worker_start_unix_ns'] for r in observed),default=None)
        intervals=[]
        for row in observed:
            if row['role'] not in {'primary','support'}:raise ValueError('Unexpected measured role')
            intervals.append(dict(alias=row['model'],role=row['role'],request_id=row['request_id'],
                pool_id=row['pool_id'],repetition=row['repetition'],
                start_seconds=(row['worker_start_unix_ns']-origin)/1e9,
                duration_seconds=(row['worker_end_unix_ns']-row['worker_start_unix_ns'])/1e9))
        timing=detail['measurement_timing']
        jobs.append(dict(job_id=detail['job_id'],state=detail['state'],status=detail['status'],
            active_aliases=detail['active_aliases'],intervals=intervals,primary_window_seconds=timing['primary_window_seconds'],
            measurement_seconds=timing['measurement_seconds'],support_drain_seconds=timing['support_drain_seconds'],
            coverage_status=detail['coverage']['status'],service_unknown_primary=detail['primary']['service_unknown_requests'],
            service_unknown_support=detail['support']['service_unknown_requests']))
    paired=[]
    for row in report.get('paired_primary_repetitions',[]):
        activity=row.get('joint_activity') or {};ratio=row.get('joint_over_baseline');fraction=activity.get('all_active_aliases_busy_fraction')
        if row['alias']=='generalist_2#0':
            paired.append(dict(joint_job=row['joint_job'],pool_id=row['pool_id'],repetition=row['repetition'],
                joint_over_baseline=ratio,common_handler_fraction=fraction,prediction_same=row.get('prediction_same'),
                generated_tokens_same=row.get('generated_tokens_same')))
    return dict(status=report['status'],all_four_terminal=report['all_four_terminal'],jobs=jobs,generalist_pairs=paired,
        independent_inputs_per_alias=32,scope='DESCRIPTIVE_PRIMARY_SUPPORT_HANDLER_TIMELINE_NOT_GPU_EXECUTION_OR_CAPACITY',
        source_analysis=dict(path=str(directory),report_sha256=names['report.json'],manifest_sha256=sha(directory/'manifest.json')),
        raw_files_read=0,U=None)


def render(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output=Path(output)
    if output.exists():raise FileExistsError('Never overwrite plot evidence')
    if not data['jobs']:raise ValueError('No published terminal analysis jobs to plot')
    output.mkdir(parents=True,exist_ok=False)
    aliases=['text#0','text#1','generalist_2#0'];colors={'primary':'#146c94','support':'#e3a129'}
    fig,axes=plt.subplots(len(data['jobs']),1,figsize=(12,2.1*len(data['jobs'])+1.3),squeeze=False)
    maximum=max((j['measurement_seconds'] or 0 for j in data['jobs']),default=1)
    for ax,job in zip(axes[:,0],data['jobs']):
        for i,alias in enumerate(aliases):
            for role in ['primary','support']:
                intervals=[(r['start_seconds'],r['duration_seconds']) for r in job['intervals'] if r['alias']==alias and r['role']==role]
                if intervals:ax.broken_barh(intervals,(i-.29,.58),facecolors=colors[role],linewidth=0)
        if job['primary_window_seconds'] is not None:
            ax.axvline(job['primary_window_seconds'],color='#8b3f63',linestyle='--',linewidth=1)
        ax.set_yticks(range(3),labels=aliases);ax.set_ylim(-.6,2.6);ax.invert_yaxis();ax.set_xlim(0,max(1,maximum)*1.015)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.set_title(f"{job['job_id']} | {job['coverage_status']} | unknown primary/support={job['service_unknown_primary']}/{job['service_unknown_support']}",fontsize=9,loc='left')
        ax.set_xlabel('Seconds since common measurement release (observed worker handler intervals)')
    handles=[Line2D([0],[0],color=colors[r],linewidth=6,label=r) for r in ['primary','support']]
    handles.append(Line2D([0],[0],color='#8b3f63',linestyle='--',label='last primary host completion'))
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.965),frameon=False)
    fig.suptitle(f"Sustained activity development diagnostic: {data['status']}",fontsize=11,y=.995)
    fig.text(.02,.005,'Support maintains context; only 32 primary input clusters per alias. Handler overlap is not GPU execution. U=null.',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,.94))
    for ext in ['png','pdf']:fig.savefig(output/f'primary-support-timeline.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));markers=['o','s','^'];block_colors=['#146c94','#c26230','#377b50']
    job_ids=sorted({r['joint_job'] for r in data['generalist_pairs']})
    for j,job in enumerate(job_ids):
        for repeat in range(3):
            rows=[r for r in data['generalist_pairs'] if r['joint_job']==job and r['repetition']==repeat and
                r['joint_over_baseline'] is not None and r['joint_over_baseline']>0 and r['common_handler_fraction'] is not None]
            if rows:ax.scatter([r['common_handler_fraction'] for r in rows],[r['joint_over_baseline'] for r in rows],
                c=block_colors[j%len(block_colors)],marker=markers[repeat],s=35,alpha=.65,label=f'{job}, repeat {repeat+1}')
    ax.axhline(1,color='#555555',linestyle=':',label='same service as paired B')
    ax.axvline(.9,color='#888888',linestyle='--',label='preregistered coverage threshold')
    ax.set_yscale('log');ax.set_xlim(-.025,1.025);ax.grid(alpha=.2);ax.set_axisbelow(True)
    ax.set_xlabel('C primary handler fraction with both specialist handlers active')
    ax.set_ylabel('G2 C/B native service time ratio (same input and repeat)')
    ax.set_title('G2 primary observations: service ratio and measured joint exposure')
    ax.legend(fontsize=7,loc='best');fig.text(.02,.005,'Descriptive development observations, clustered by 32 inputs; no causal treatment, fit or capacity claim.',fontsize=8)
    fig.tight_layout(rect=(0,.025,1,1))
    for ext in ['png','pdf']:fig.savefig(output/f'generalist-paired-service-overlap.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (output/'plot-data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    manifest=dict(source_analysis=data['source_analysis'],tool={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()},raw_files_read=0,gpu_started=False,U=None)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=load_plot_data(args.analysis);render(data,args.output)
    print(json.dumps(dict(status='RENDERED_PUBLISHED_ANALYSIS_ONLY',output=str(args.output.resolve()),raw_files_read=0,gpu_started=False,U=None)))


if __name__=='__main__':main()
