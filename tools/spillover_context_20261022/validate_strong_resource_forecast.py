"""Validate sealed numerical probes against one explicitly authorized terminal.

No queue discovery, model fitting, candidate selection, GPU or score activation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'artifacts/static-spillover-study-20261022'
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5';sys.path.insert(0,str(SOURCE))
from gsdeploy.energy import integrate_power


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def error(predicted,actual):
    if predicted is None:return dict(predicted=None,actual=actual,error=None,relative_error=None)
    return dict(predicted=predicted,actual=actual,error=predicted-actual,
        relative_error=(predicted-actual)/actual if actual!=0 else None,
        absolute_relative_error=abs(predicted-actual)/abs(actual) if actual!=0 else None)


def validate(terminal,terminal_sha,output):
    output=Path(output).resolve();terminal=Path(terminal).resolve()
    if output.exists():raise FileExistsError(output)
    opened={}
    def read(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Source SHA changed: '+str(path))
        opened[str(path)]=actual;return json.loads(path.read_text())
    forecast_dir=BASE/'strong-controls-resource-prequential-forecast-v1'
    manifest=read(forecast_dir/'manifest.json','1308b1202de56c2aec4d3cbd578f9a6b24c69135ed6c9eab5ce170ba7a5d1e70')
    forecast=read(forecast_dir/'forecast.json',manifest['files']['forecast.json'])
    if terminal.name!='run.json':raise ValueError('Only explicit immutable terminal raw allowed')
    raw=read(terminal,terminal_sha)
    if raw['transport_config']['random_seed']!=107:raise ValueError('Only strong107 authorized here')
    if raw['experiment_job_id']!='ccc1f89ccadc72208cc32b0c34777a16a3227a6e7d4a87761b3f9674ad4396d2':raise ValueError('Unexpected strong107 job')
    if raw['status']!='completed' or raw['plan']!=forecast['plan']:raise ValueError('Unexpected terminal identity')
    config=raw['config'];memory=raw['memory_evidence'];energy=raw['energy_evidence'];utility=raw['evaluation']['utility']
    if not all([memory['measured'],memory['resource_valid'],energy['coverage_valid'],utility['eligible']]):raise ValueError('Terminal resource/utility incomplete')
    if config['gpu_memory_mib']!=49140 or config['catalog']['safe_factor']!=1.110854:raise ValueError('Resource contract changed')
    if raw['preflight']['gpu']['uuid']!='GPU-8a262fe0-598c-46b7-c953-41d9da84a47e':raise ValueError('Wrong device identity')
    samples=[*memory['samples'],*[s for s in [memory.get('before'),memory.get('after')] if s is not None]]
    if not samples or any(s['foreign_pids'] for s in samples):raise ValueError('Owned memory evidence incomplete')
    peak=max(s['owned_used_mib'] for s in samples);horizon=raw['transport_config']['measurement_seconds']
    if horizon!=600:raise ValueError('Wrong formal window')
    window=raw['measurement_window'];start=energy['startup_begin_time_ns']
    window_integral=integrate_power(energy['samples'],window['start_time_ns'],window['end_time_ns'])
    startup_integral=integrate_power(energy['samples'],start,window['start_time_ns'])
    if not close(window_integral,energy['run_joules']) or not close(startup_integral,energy['startup_joules']):raise ValueError('Energy reintegration failed')
    actual=dict(M_raw_peak_over_budget=peak/49140,M_owned_sampled_peak_mib=peak,M_norm=peak*1.110854/49140,
        E_window_watts=window_integral/600,E_window_joules=window_integral,E_startup_joules=startup_integral,
        E_norm=(window_integral/600+startup_integral/3600)/450)
    if not close(peak*1.110854,memory['safe_memory_mib']):raise ValueError('Safe peak reproduction failed')
    for t in ['M','E']:
        if not close(actual[t+'_norm'],utility[t+'_norm']):raise ValueError('Normalization reproduction failed')
    for t in 'QTLPME':
        if not close(-config['weights'][t]*utility[t+'_norm'],utility[t+'_contribution']):raise ValueError('Measured contribution mismatch')
    measured_u=sum(utility[t+'_contribution'] for t in 'QTLPME')
    if not close(measured_u,utility['U']):raise ValueError('Measured U reproduction failed')
    model_errors={}
    for name,model in forecast['models'].items():
        model_errors[name]=dict(target=model['target'],formal_prediction_status=model['status'],
            valid_prediction=model['forecast'],candidate_feature_hull=model['model_feature_projection_hull'],
            diagnostic_extrapolation_only=error(model['diagnostic_extrapolation_only']['value'],actual[model['target']]))
    memory_errors=[dict(model_name=m['model_name'],diagnostic_M_norm=error(m['diagnostic_extrapolation_only']['M_norm'],actual['M_norm']),
        diagnostic_owned_sampled_peak_mib=error(m['diagnostic_extrapolation_only']['owned_sampled_peak_mib'],peak)) for m in forecast['memory_forecasts']]
    pipeline_errors=[dict(name=p['name'],diagnostic_E_norm=error(p['diagnostic_extrapolation_only']['E_norm'],actual['E_norm']),
        window_model=p['window_model'],startup_model=p['startup_model']) for p in forecast['E_pipelines']]
    report=dict(schema='gsdeploy.strong107_prequential_resource_probe_validation.v1',
        status='OUT_OF_DOMAIN_NUMERICAL_PROBE_ERROR_ONLY',authorized_seed=107,terminal=dict(path=str(terminal),sha256=terminal_sha),
        forecast=dict(path=str(forecast_dir/'forecast.json'),sha256=opened[str(forecast_dir/'forecast.json')]),
        actual=actual,model_errors=model_errors,memory_errors=memory_errors,E_pipeline_errors=pipeline_errors,
        field_audit=dict(owned_peak_reproduced=True,safe_factor_applied_once=True,energy_integrals_reproduced=True,
            measured_six_contributions_reproduced=True,measured_U=measured_u,full_predicted_U=None),
        startup=dict(observed_total_seconds=(window['start_time_ns']-start)/1e9,
            priming_count=len(raw['priming_request_records']),warmup_count=len(raw['warmup_request_records']),
            subphase_energies=None,subphase_causes='UNKNOWN; no stage boundaries/power model fitted; equal counts/inputs do not imply equal initialization/inference/power elapsed behavior'),
        limitations=['The deployment is absent from training and full M/E contexts are outside their training hulls. Simpler candidates retain their own descriptive feature support.',
            'All formal predictions remain UNKNOWN. Numerical errors test presealed out-of-domain hypotheses; they do not establish independent applicability, physical bounds, activation or optimality.',
            'Only one explicitly authorized registered strong107 terminal opened. No strong108/current/live results, refitting, coefficient edits or candidate selection.',
            'Sampled memory is original owned process evidence after READY; it is not continuous/load-phase physical peak.',
            'Observed measured U is reproduced solely to audit the terminal; no full predicted U or algorithmic ranking is computed.'],
        refit=False,selected_model=None,predictor_activated=False,full_predicted_U=None,GPU_started=False,gpu_queried=False,
        reads_108_actual_results=False,queue_read=False)
    output.mkdir(parents=True,exist_ok=False);save(output/'report.json',report)
    opened[str(Path(__file__).resolve())]=sha(__file__);opened[str(SOURCE/'gsdeploy/energy.py')]=sha(SOURCE/'gsdeploy/energy.py')
    if any(sha(p)!=s for p,s in opened.items()):raise ValueError('Input changed during validation')
    save(output/'manifest.json',dict(inputs=opened,files={'report.json':sha(output/'report.json')},sealed_forecast_unchanged=True))
    print(json.dumps(dict(output=str(output),report_sha256=sha(output/'report.json'),actual=actual,
        M_full=model_errors['M_full']['diagnostic_extrapolation_only'],E_pipelines=pipeline_errors)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--terminal',type=Path,required=True)
    parser.add_argument('--terminal-sha256',required=True);parser.add_argument('--output',type=Path,default=BASE/'strong107-resource-prequential-validation-v1')
    args=parser.parse_args();validate(args.terminal,args.terminal_sha256,args.output)
