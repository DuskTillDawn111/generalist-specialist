"""Matched native candidate episodes with durable progress and complete energy."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_activation_capture import gpu
from experiments.run_activation_references import reporting_runtime,save
from gsdeploy.config import SystemConfig
from gsdeploy.runtime import runner
from gsdeploy import normalization as norm


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--reference',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--role',choices=('development','normalization_development','confirmation'),default='development')
    parser.add_argument('--confirmation-protocol',type=Path)
    args=parser.parse_args();output=args.output.resolve();reference=args.reference.resolve()
    config=SystemConfig.load(args.config);plan=json.loads(args.plan.read_text())
    if set(plan)!={'deployment','routing'}:raise ValueError('Candidate plan contains deployment and QPS routing only')
    validation=config.validate_plan(plan['deployment'],plan['routing'])
    if not validation['constraints_valid']:raise ValueError('Candidate lacks valid original capacity/memory constraints')
    protocol=norm.contract(config.catalog)
    if protocol is None:raise ValueError('Explicit reference-v2 episode contract required')
    reserved=protocol.get('confirmation_seeds',[6110491,6110492])
    if args.role!='confirmation' and args.seed in reserved:raise ValueError('Reserved confirmation seed cannot enter development')
    if args.role=='confirmation':
        if args.confirmation_protocol is None:raise ValueError('Precommitted candidate/predictor/criteria confirmation protocol required')
        confirmation=json.loads(args.confirmation_protocol.read_text())
        from gsdeploy.paper_forecast import digest
        from gsdeploy.continuous.predictor_adapter import source_identity
        if (confirmation.get('schema')!='gsdeploy.activation_confirmation_protocol.v1'
                or args.seed not in confirmation['seeds'] or args.seed not in reserved
                or sha(args.plan) not in confirmation['candidate_file_sha256']
                or sha(args.config) not in confirmation['config_file_sha256']
                or confirmation.get('predictor_sha256')!=digest(config.catalog)
                or confirmation.get('predictor_source_sha256')!=source_identity()
                or not confirmation.get('criteria')):
            raise ValueError('Candidate is outside its frozen confirmation protocol')
        norm.require_ready(config.catalog,config.workload,config.latency_reference_ms,
                           config.horizon_seconds,config.gpu_memory_mib)
    output.mkdir(parents=True,exist_ok=False)
    actual=gpu()
    if actual['name']!='NVIDIA GeForce RTX 4090' or actual['memory_mib']<config.gpu_memory_mib:
        raise ValueError('Same-model RTX 4090 episode environment required')
    workers=json.loads((reference/'worker-registry.json').read_text())
    lock=(ROOT/'artifacts/.gpu-experiment.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=runner.DeploymentRuntime;runner.DeploymentRuntime=reporting_runtime(output)
    try:
        paths=sorted((ROOT/'gsdeploy').rglob('*.py'))+[Path(__file__).resolve(),args.config.resolve(),args.plan.resolve()]
        paths.extend(Path(row['spec']) for row in workers.values())
        # Retained backends execute from separately pinned source trees as well.
        paths.extend(Path(r['source']) for r in json.loads((reference/'sources.json').read_text()))
        paths=sorted(set(paths))
        receipts=[]
        for index,path in enumerate(paths):
            dest=output/'source-snapshot'/str(index)/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
            content=path.read_bytes();dest.write_bytes(content)
            receipts.append(dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),snapshot=str(dest)))
        save(output/'protocol.json',dict(config=config.as_dict(),plan=plan,seed=args.seed,role=args.role,
            gpu=actual,compatibility_policy='same_model',sources=receipts,
            confirmation_protocol=(json.loads(args.confirmation_protocol.read_text()) if args.confirmation_protocol else None)))
        raw=runner.run(config,plan,ROOT/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl',
            backend='native',workers=workers,duration=config.horizon_seconds,warmup=10,
            seed=args.seed,arrival_process=config.arrival_process,startup_timeout=300,
            drain_timeout=config.drain_seconds,admission=reference/'quality-evidence.json',
            priming_policy='largest_three_per_task')
        changed=[r['path'] for r in receipts if sha(Path(r['path']))!=r['sha256']]
        raw['measurement_role']=args.role;raw['source_drift_paths']=changed
        if changed:
            raw['status']='failed';raw['errors'].append('Executable/configuration sources changed during candidate measurement')
            if 'evaluation' in raw:raw['evaluation']['utility'].update(U=None,eligible=False,source_drift=True)
        save(output/'run.json',raw)
        print(json.dumps(dict(status=raw['status'],output=str(output/'run.json'),errors=raw['errors'],
            requests=len(raw['request_records']),energy_j=raw['energy_evidence'].get('episode_joules'),
            U=raw.get('evaluation',{}).get('utility',{}).get('U')),ensure_ascii=False))
        return 0 if raw['status']=='completed' else 1
    finally:
        runner.DeploymentRuntime=previous;fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':raise SystemExit(main())
