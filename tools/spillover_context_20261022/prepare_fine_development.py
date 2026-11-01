"""Freeze a small paired open-load development study with the original runtime."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/static-study-gpu-20261022/execution-source-v5'
sys.path.insert(0,str(SOURCE))
from gsdeploy import static_study
from gsdeploy.config import SystemConfig
from gsdeploy.quality import quality_digest


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')


def prepare():
    destination=ROOT/'artifacts/static-spillover-study-20261022/text185-fine-development-v1'
    if destination.exists():raise FileExistsError(destination)
    original_path=ROOT/'artifacts/static-study-gpu-20261022/execution-packet-v2.json'
    original=json.loads(original_path.read_text())
    scenario_path=ROOT/'artifacts/static-spillover-study-20261022/fine-route-development-scenarios-v2/report.json'
    scenario=json.loads(scenario_path.read_text())
    for item in scenario['inputs']:
        assert static_study._file(item['path'])==item
    candidates=[c for c in scenario['cases'] if c['workload']['text']==1.85 and c['generalist_share']=='1/32']
    assert {c['seed'] for c in candidates}=={107,108}
    assert all(c['scenarios'][0]['L_delta_from_no_split']<0 and c['scenarios'][0]['P_delta_from_no_split']<0 for c in candidates)
    raw=copy.deepcopy(original['config'])
    raw.update(workload=dict(text=1.85,image=.01,asr=.01,ocr=.01),horizon_seconds=600,random_seed=107)
    raw['catalog']['performance_mode']='task_replica_proxy'
    raw['catalog']['memory_mode']='standalone_proxy'
    raw['catalog']['measurement_context']={**raw['catalog'].get('measurement_context',{}),
        'gpu_uuid':'GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'}
    for key in ['weights','latency_reference_ms','gpu_memory_mib','drain_seconds','arrival_process',
                'energy_reference_watts','startup_amortization_seconds','utility_schema']:
        assert raw[key]==original['config'][key]
    for key in ['models','quality','safe_factor','capacity_factors','memory_table_mib','task_replica_profiles']:
        assert raw['catalog'][key]==original['config']['catalog'][key]
    config=SystemConfig(**raw)
    order=[(107,'0'),(107,'1/32'),(108,'1/32'),(108,'0')]
    protocol=dict(schema='gsdeploy.text185_fine_development.v1',
        purpose='Paired exploratory open-load test of tiny spillover; not algorithm selection or optimum confirmation.',
        workload=raw['workload'],horizon_seconds=600,arrival_process='poisson',warmup_seconds=30,
        deployment=dict(text=2,generalist_2=1,ocr=1),
        text_routes=['text100%','text31/32 + G2 1/32'],
        background_routes=dict(image='generalist_2',asr='generalist_2',ocr='ocr'),
        development_seeds=[107,108],execution_order=order,
        selection_reason='At1.85 the unchanged standalone rho proxy admits both routes; explicit full-joint-text FIFO sensitivity improves L/P in both declared development seeds. Native interference, Q/M/E and stability can reverse that result.',
        quality='Actual concurrent native output rescored in original six-term utility; seed108 historical Q worsens with split and is retained.',
        small_input_exposure='Original expanded SWRR visits only2 G2 text inputs per finite seed; this is disclosed, never replaced with random/ideal fractional routing.',
        route_domain='Concrete native development route1/32. Current search adds pair denominators32/64 separately; old K8 certificates are not extended by this study.',
        diagnostic_background='New full-input background diagnostic continues separately. Its failure/incompleteness is retained; this bounded open-load development test does not activate a predictor.',
        stability='Unchanged frozen v5 including original steady/recoverable states and rho<=.8; no threshold edits.',
        reserved_seeds=dict(independent_validation=[81,82],final_confirmation=[71,72,73]),
        no_hidden_retry=True,all_conditions_retained=True,all_utility_terms_unchanged=True,
        evidence=scenario_path.as_posix(),full_pool_optimum=False,predictor_activated=False)
    destination.mkdir(parents=True,exist_ok=False)
    write(destination/'system-config.json',raw);write(destination/'discovery-protocol.json',protocol)
    bindings=static_study._bindings(original['inputs'],SOURCE)
    bindings += [static_study._file(p) for p in [original_path,scenario_path,Path(__file__).resolve(),
                 destination/'system-config.json',destination/'discovery-protocol.json']]
    bindings=sorted({b['path']:b for b in bindings}.values(),key=lambda b:b['path'])
    jobs=[]
    for seed,share_text in order:
        share=Fraction(share_text);text={'text':1.85*(1-float(share))}
        if share:text['generalist_2']=1.85*float(share)
        plan=dict(deployment=dict(text=2,generalist_2=1,ocr=1),routing=dict(text=text,
            image={'generalist_2':.01},asr={'generalist_2':.01},ocr={'ocr':.01}))
        constraints=config.validate_plan(**plan)
        if not constraints['constraints_valid']:raise ValueError(str(constraints))
        label=f'text185_g2share_{share.numerator}of{share.denominator}_seed{seed}'
        job=static_study._job(config,original['workers'],plan,label,'calibration',seed,bindings,30)
        job.update(purpose='tiny_spillover_open_load_development',proxy_constraints=constraints)
        jobs.append(job)
    packet=dict(schema='gsdeploy.static_experiments.v1',status='PREDECLARED_FINE_ROUTE_DEVELOPMENT',
        inputs=original['inputs'],config=raw,workers=original['workers'],bindings=bindings,
        frozen_files_sha256=quality_digest(bindings),jobs=jobs,gpu_started=False,
        discovery_protocol=str(destination/'discovery-protocol.json'),
        execution=dict(policy='Serial native collection, preserved attempt per condition. Run-level ineligibility retained; infrastructure failure stops queue.',
                       runtime_source=str(SOURCE),scheduled=False))
    validation=static_study.validate_packet(packet);assert validation['valid']
    write(destination/'experiment-packet.json',packet)
    write(destination/'preparation-receipt.json',dict(packet=static_study._file(destination/'experiment-packet.json'),
        validation=validation,source=static_study._file(Path(static_study.__file__)),jobs=4,GPU_started=False))
    print(json.dumps(dict(destination=str(destination),validation=validation,jobs=4)))


if __name__=='__main__':prepare()
