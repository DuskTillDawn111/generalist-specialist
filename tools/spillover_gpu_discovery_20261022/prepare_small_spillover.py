"""Predeclare a smaller-spillover development scan after the negative 25% pilot.

All data are development data. Never changes a predictor, threshold or holdout.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'artifacts/static-study-gpu-20261022'
BASE = ROOT / 'artifacts/static-spillover-study-20261022'
SOURCE = OLD / 'execution-source-v5'
sys.path.insert(0, str(SOURCE))
from gsdeploy.config import SystemConfig
from gsdeploy.static_study import _bindings, _file, _job, validate_packet
from gsdeploy.quality import quality_digest


def main():
    out = BASE / 'development-small-spillover-v2'
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((OLD / 'execution-packet-v2.json').read_text())
    source = BASE / 'development-asr18-v1/runs/queue-progress.json'
    previous = json.loads(source.read_text())
    if previous['status'] != 'CALIBRATION_COMPLETE':
        raise ValueError('Both first development measurements must be complete')
    settings = [(1.8, .875), (1.9, 1.), (1.9, .875)]
    raw = copy.deepcopy(original['config'])
    raw.update(workload={'text': .05, 'image': .05, 'ocr': .05, 'asr': 1.8},
               horizon_seconds=600, random_seed=51, route_grid=8)
    raw['catalog']['performance_mode'] = 'task_replica_proxy'
    raw['catalog']['memory_mode'] = 'standalone_proxy'
    raw['catalog']['measurement_context'] = {**raw['catalog'].get('measurement_context', {}),
        'gpu_uuid': 'GPU-8a262fe0-598c-46b7-c953-41d9da84a47e'}
    protocol = dict(schema='gsdeploy.spillover_development_scan.v1',
        observed_development_source=dict(path=str(source.resolve()),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(), jobs=previous['jobs']),
        rationale='25 percent spillover worsened utility at 1.8 rps; test less generalist queueing and a modestly higher target load',
        settings=[dict(asr_rps=rate, specialist_share=share) for rate, share in settings],
        deployment={'asr': 2, 'generalist_2': 1, 'ocr': 1},
        horizon_seconds=600, seed=51, warmup_seconds=30,
        purpose='Exploration only; no independent validation or optimality claim',
        holdout_observations_used=False, predictors_changed=False,
        all_attempts_preserved=True, no_confirmation_seeds_used=True)
    (out/'discovery-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'system-config.json').write_text(json.dumps(raw,indent=2)+'\n')
    bindings = _bindings(original['inputs'], SOURCE)
    bindings += [_file(out/name) for name in ('discovery-protocol.json','system-config.json')]
    bindings += [_file(Path(__file__).resolve())]
    jobs=[]
    for rate, share in settings:
        config_raw=copy.deepcopy(raw)
        config_raw['workload']['asr']=rate
        config=SystemConfig(**config_raw)
        routing={'text': {'generalist_2': .05}, 'image': {'generalist_2': .05},
                 'ocr': {'ocr': .05}, 'asr': {'asr': rate*share}}
        if share < 1:
            routing['asr']['generalist_2']=rate*(1-share)
        plan=dict(deployment=protocol['deployment'],routing=routing)
        constraints=config.validate_plan(**plan)
        if not constraints['constraints_valid']:
            raise ValueError(str(constraints))
        label=('asr'+str(rate)+'_specialist_share_'+str(share)).replace('.','p')
        job=_job(config,original['workers'],plan,label,'calibration',51,bindings,30)
        job['purpose']='new_load_exploratory_development_not_final_confirmation'
        job['proxy_constraints']=constraints
        jobs.append(job)
    packets=[]
    for rate in sorted({rate for rate, _ in settings}):
        config_raw=copy.deepcopy(raw)
        config_raw['workload']['asr']=rate
        rate_jobs=[j for j,(job_rate,_) in zip(jobs,settings) if job_rate==rate]
        packet=dict(schema='gsdeploy.static_experiments.v1',status='PREDECLARED_SMALL_SPILLOVER_DEVELOPMENT',
            inputs=original['inputs'],config=config_raw,workers=original['workers'],bindings=bindings,
            frozen_files_sha256=quality_digest(bindings),gpu_started=False,jobs=rate_jobs,
            discovery_protocol=str(out/'discovery-protocol.json'),
            execution=dict(policy='Serial native collection using unchanged frozen v5 source'))
        result=validate_packet(packet)
        if not result['valid']:
            raise ValueError(str(result))
        packet_path=out/('experiment-packet-asr'+str(rate).replace('.','p')+'.json')
        packet_path.write_text(json.dumps(packet,indent=2)+'\n')
        packets.append(dict(path=str(packet_path),jobs=len(rate_jobs),validation=result))
    (out/'manifest.json').write_text(json.dumps(dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir() if p.is_file()},gpu_started=False),indent=2)+'\n')
    print(json.dumps(dict(packets=packets)))


if __name__=='__main__':
    main()
