"""Read/bind every distinct proposed contract; query GPU, never start workers."""
import argparse,copy,importlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.queue_refined_20261023 import protocol,admission
from tools.queue_refined_20261023.native_runner import frozen_runtime


def run(plan_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(plan_path).read_text());original=frozen_runtime();package=original.__package__.split('.runtime')[0]
    static=importlib.import_module(package+'.static_study');configuration=importlib.import_module(package+'.config')
    checks=[];seen={};inputs=[protocol.receipt(plan_path)]
    for bound in plan['development_packets']:
        packet=json.loads(Path(bound['path']).read_text())
        if protocol.receipt(bound['path'])!=bound or not protocol.validate_packet(packet)['valid']:raise ValueError('Frozen packet changed')
        inputs.append(bound)
        for job in packet['jobs']:
            key=protocol.digest(job['policy_contract'])
            if key in seen:
                seen[key]['equivalent_contract_job_ids'].append(job['id']);continue
            target=output/f'text{job["policy_contract"]["text_rate"]:g}-{job["role"]}';target.mkdir()
            base=json.loads(Path(job['base_packet']['path']).read_text());binding=copy.deepcopy(base['jobs'][0])
            binding['plan']=job['nominal_plan'];binding['seed']=job['seed']
            binding['identity'].update(seed=job['seed'],horizon_seconds=600,warmup_seconds=30)
            try:
                config,admission_path,workers=static.bind_native_job(base,binding,target/'native-binding')
                config['catalog']['performance_mode']='task_replica_proxy';config['catalog']['memory_mode']='standalone_proxy'
                config['random_seed']=job['seed'];config['horizon_seconds']=600
                check=admission.preflight(configuration.SystemConfig(**config),job['nominal_plan'],Path(base['inputs'])/'runtime-inputs.jsonl',workers,job['policy_contract'],
                    admission=str(admission_path),require_gpu=True,capacity_evidence=job['capacity_evidence'])
                if (check.get('gpu') or {}).get('uuid')!='GPU-8a262fe0-598c-46b7-c953-41d9da84a47e':
                    check['ready']=False;check['issues'].append('GPU_UUID_NOT_PREDECLARED')
            except Exception as exc:check=dict(ready=False,issues=[repr(exc)])
            row=dict(label=job['label'],role=job['role'],job_id=job['id'],load=job['policy_contract']['text_rate'],
                equivalent_contract_job_ids=[job['id']],check=check)
            seen[key]=row;checks.append(row);protocol.write(target/'report.json',row)
    result=dict(ready=bool(checks) and all(r['check']['ready'] for r in checks),checks=checks,inputs=inputs,
        GPU_workers_started=False,scope='Read-only GPU identity/idle query plus CPU checkpoint, sample, contract and original resource-envelope checks.')
    protocol.write(output/'report.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.plan,a.output);print(json.dumps(dict(ready=result['ready'],contracts=len(result['checks']),issues=[r for r in result['checks'] if not r['check']['ready']]),ensure_ascii=False))
    raise SystemExit(0 if result['ready'] else 1)
