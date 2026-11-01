"""Prepare only the paired all-specialist text1.85 challenger; never launches GPU."""
import copy
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
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def main():
    destination=ROOT/'artifacts/static-spillover-study-20261022/text185-strong-controls-v1'
    if destination.exists():raise FileExistsError(destination)
    original_path=ROOT/'artifacts/static-study-gpu-20261022/execution-packet-v2.json'
    fine_path=ROOT/'artifacts/static-spillover-study-20261022/text185-fine-development-v1/experiment-packet.json'
    screen_path=ROOT/'artifacts/static-spillover-study-20261022/text185-strong-challengers-cpu-v4/report.json'
    review_path=screen_path.parent/'shortlist-review-v1/review.json'
    original=json.loads(original_path.read_text());fine=json.loads(fine_path.read_text())
    assert static_study.validate_packet(fine)['valid']
    config_data=copy.deepcopy(fine['config']);config=SystemConfig(**config_data)
    assert config.workload==dict(text=1.85,image=.01,asr=.01,ocr=.01)
    assert config.horizon_seconds==600 and config.arrival_process=='poisson'
    for k in ('weights','latency_reference_ms','gpu_memory_mib','drain_seconds','arrival_process',
              'energy_reference_watts','startup_amortization_seconds','utility_schema'):
        assert config_data[k]==original['config'][k]
    for k in ('models','quality','safe_factor','capacity_factors','memory_table_mib','task_replica_profiles'):
        assert config_data['catalog'][k]==original['config']['catalog'][k]
    plan=dict(deployment=dict(text=2,image=1,ocr=1,asr=1),
        routing={t:{t:q} for t,q in config.workload.items()})
    constraints=config.validate_plan(**plan);assert constraints['constraints_valid']
    protocol=dict(schema='gsdeploy.text185_strong_controls.v1',
        purpose='Prepare-only strong all-specialist challenger to fixed-deployment tiny-spillover development; no optimum claim.',
        workload=config.workload,horizon_seconds=600,arrival_process='poisson',warmup_seconds=30,
        plan=plan,development_seeds=[107,108],execution_order=[107,108],
        comparison_contract='Same task/input/arrival traces per seed as text185 fine4; preserve all original Q/T/L/P/M/E terms, constraints and failures.',
        selection_reason='All-specialist text2 is formal standalone-proxy feasible, gives stronger historical background quality, and removes generalist interference at higher residency cost. Real utility is unknown.',
        constraints=constraints,
        actual_resource_preflight_required_before_GPU=True,
        actual_resource_preflight='Original native runtime must run real hardware/checkpoint/memory preflight; CPU packet validity does not establish co-resident feasibility.',
        no_gpu_started=True,no_auto_execution=True,no_other_challengers_enqueued=True,
        no_hidden_retry=True,all_attempts_retained=True,
        reserved_seeds=dict(independent_validation=[81,82],final_confirmation=[71,72,73]),
        predictor_activated=False,full_pool_optimum=False,
        allowed_followup='Root reviews and may serially collect these two jobs after fine4; preparation itself does not schedule or launch them.')
    destination.mkdir(parents=True,exist_ok=False)
    write(destination/'system-config.json',config_data);write(destination/'discovery-protocol.json',protocol)
    bindings=static_study._bindings(original['inputs'],SOURCE)
    bindings += [static_study._file(p) for p in (original_path,fine_path,screen_path,review_path,Path(__file__).resolve(),
        destination/'system-config.json',destination/'discovery-protocol.json')]
    bindings=sorted({b['path']:b for b in bindings}.values(),key=lambda b:b['path'])
    jobs=[]
    for seed in (107,108):
        label=f'text185_all_specialist_text2_seed{seed}'
        job=static_study._job(config,original['workers'],plan,label,'calibration',seed,bindings,30)
        job.update(purpose='text185_all_specialist_strong_challenger_development',proxy_constraints=constraints)
        assert job['plan']==plan
        jobs.append(job)
    packet=dict(schema='gsdeploy.static_experiments.v1',status='PREDECLARED_STRONG_CONTROL_DEVELOPMENT',
        inputs=original['inputs'],config=config_data,workers=original['workers'],bindings=bindings,
        frozen_files_sha256=quality_digest(bindings),jobs=jobs,gpu_started=False,
        discovery_protocol=str(destination/'discovery-protocol.json'),
        execution=dict(policy='Prepare only. Root review and real preflight required; serial native collection after fine4 if scheduled. All failures preserved; no hidden retries.',
            runtime_source=str(SOURCE),scheduled=False,auto_execute=False))
    validation=static_study.validate_packet(packet);assert validation['valid']
    write(destination/'experiment-packet.json',packet)
    write(destination/'preparation-receipt.json',dict(packet=static_study._file(destination/'experiment-packet.json'),
        validation=validation,source=static_study._file(Path(static_study.__file__)),jobs=2,GPU_started=False,
        scheduled=False,actual_resource_preflight_required_before_GPU=True))
    print(json.dumps(dict(destination=str(destination),validation=validation,jobs=2,scheduled=False)))

if __name__=='__main__':main()
