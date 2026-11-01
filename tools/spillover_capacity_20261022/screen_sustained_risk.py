"""CPU sensitivity screen of a censored development context, never a predictor.

The observed joint-service table is incomplete as a calibration experiment. Its
use here illustrates route workload under that one observed scenario; every
candidate remains NO_GO because background, repeatability and validation lack
evidence. It does not bound physical capacity at an unmeasured load.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def receipt(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Never overwrite research evidence')
    packet=json.loads(args.packet.read_text());queue=json.loads(args.queue.read_text())
    if queue['status']!='PAUSED_ON_FAILED_ATTEMPT_NO_RETRY': raise ValueError('Expected preserved failed queue')
    if queue['packet_sha256']!=receipt(args.packet)['sha256']: raise ValueError('Queue packet identity differs')
    registered=[r for r in queue['jobs'] if r['job_id']=='block1_C_seed103']
    if len(registered)!=1 or registered[0]['status']!='failed': raise ValueError('Exact failed diagnostic receipt required')
    source=registered[0]['raw'];rawpath=Path(source['path'])
    if rawpath.name!='run.json' or receipt(rawpath)['sha256']!=source['sha256']: raise ValueError('Final raw hash differs')
    raw=json.loads(rawpath.read_text())
    if raw['status']!='failed' or raw['measurement_window']['primary_completion_time_ns'] is not None: raise ValueError('Expected incomplete primary cohort')
    analysis_path=Path(__file__).with_name('analyze.py')
    spec=importlib.util.spec_from_file_location('_joint_risk_activity',analysis_path)
    analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
    end=raw['measurement_window']['end_time_ns'];offset=raw['measurement_window']['start_unix_ns']-raw['measurement_window']['start_time_ns']
    # Clip all activity at the real measurement boundary. A response arriving in
    # shutdown is not a completed primary observation, but its pre-cutoff handler
    # may still form part of another request's activity context.
    intervals=[]
    for row in raw['request_records']:
        response=row.get('worker_response') or {}
        begin,finish=response.get('start_ns'),response.get('end_ns')
        if response.get('success') is True and type(begin) is int and type(finish) is int and begin<min(finish,end+offset):
            intervals.append(dict(alias=row['model'],start_ns=begin,end_ns=min(finish,end+offset)))
    active=raw['job']['active_aliases']
    activity=analysis.common_activity(intervals,active,0)
    vectors={};coverage={}
    for alias in active:
        per_input={pid:[] for pid in set(raw['job']['input_pool_ids'])}
        for row in raw['request_records']:
            if row['model']!=alias or row['role']!='primary' or row['status']!='completed' or row['completion_time_ns']>end: continue
            response=row['worker_response']
            fraction=analysis.interval_exposure(response['start_ns'],response['end_ns'],activity['segments'],active,alias)['all_active_aliases_busy_fraction']
            if fraction>=.90: per_input[row['pool_id']].append(response['service_ms'])
        if len(per_input)!=32 or any(len(values)<2 for values in per_input.values()): raise ValueError('Even exploratory covered-input medians unavailable')
        vectors[alias]={pid:statistics.median(values) for pid,values in sorted(per_input.items())}
        coverage[alias]={pid:len(values) for pid,values in sorted(per_input.items())}
    generalist_ms=statistics.mean(vectors['generalist_2#0'].values())
    specialist_alias_means={a:statistics.mean(vectors[a].values()) for a in ['text#0','text#1']}
    specialist_ms=max(specialist_alias_means.values())
    config=json.loads(Path(packet['system_config']).read_text())
    factors=config['catalog']['capacity_factors']
    gc=factors['generalist_2']['1'];sc=factors['text']['2']
    if gc!=1 or sc!=2: raise ValueError('Original capacity factors differ')
    shares=[0,1/64,1/32,1/16,1/8,1/4]
    rows=[]
    for tick in range(10,61):
        rate=tick/20
        for share in shares:
            g=rate*share*generalist_ms/1000/gc
            s=rate*(1-share)*specialist_ms/1000/sc
            rows.append(dict(text_rps=rate,generalist_share=share,
                joint_observed_scenario_text_only_rho_G2=g,joint_observed_scenario_text_only_rho_specialist=s,
                scenario_exceeds_original_point8_ignoring_background=max(g,s)>.8,
                total_rho_with_background=None,formal_rho=None,U=None,
                execution_decision='NO_GO_BACKGROUND_AND_REPLICATION_UNKNOWN'))
    summary=[]
    for share in shares:
        necessary_g2_rate=.8*gc/(share*generalist_ms/1000) if share else None
        necessary_spec_rate=.8*sc/((1-share)*specialist_ms/1000)
        summary.append(dict(generalist_share=share,
            joint_scenario_text_only_rate_at_rho_point8_G2=necessary_g2_rate,
            joint_scenario_text_only_rate_at_rho_point8_specialist=necessary_spec_rate,
            scope='Algebra in one incomplete observed context, not a measured/sustainable arrival rate or a guarantee.'))
    inputs=[receipt(args.packet),receipt(args.queue),receipt(rawpath),receipt(analysis_path),receipt(packet['system_config']),receipt(__file__)]
    result=dict(schema='gsdeploy.sustained_partial_risk_sensitivity.v1',status='CPU_SENSITIVITY_ONLY_ALL_CANDIDATES_NO_GO',
        origin='Failed600s single C block; within-window completed primary only, >=.90 joint overlap; 32 input clusters, >=2 covered observations each; third repetition incomplete, no block2.',
        generalist_equal_input_mean_of_covered_primary_medians_ms=generalist_ms,
        specialist_alias_equal_input_means_ms=specialist_alias_means,
        specialist_sensitivity_ms=specialist_ms,specialist_choice='Larger alias mean, explicitly exploratory; not a validated family service estimate or probabilistic upper bound',
        original_capacity_factors=dict(generalist_2=gc,text=sc),original_formal_rho_limit=.8,
        service_vectors=vectors,covered_observations_per_input=coverage,
        scenario_route_limits=summary,candidates=rows,
        chosen_load=None,chosen_route=None,formal_search_enabled=False,GPU_started=False,U=None,
        unresolved=['19 unreleased G2 primary; one shutdown-only response excluded','no cross-block repeatability','missing full-input/background joint response','no validated rate/share service mapping','memory final bracket missing in failed attempt'],
        interpretation='Tiny-share CPU candidates are worth evaluating. The original1/8 and1/4 must not be presumed safe near specialist pressure. No fraction is selected on utility or guaranteed to help.',
        inputs=inputs)
    for item in inputs:
        if receipt(item['path'])['sha256']!=item['sha256']: raise ValueError('Input changed')
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'screen.json').open('x') as stream: json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    with (args.output/'manifest.json').open('x') as stream: json.dump(dict(inputs=inputs,files=[receipt(args.output/'screen.json')]),stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],candidate_count=len(rows),G2_ms=generalist_ms,specialist_ms=specialist_ms,chosen_load=None,GPU_started=False,U=None)))


if __name__=='__main__': main()
