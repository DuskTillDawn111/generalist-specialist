"""Predeclare shared text/ASR pressure without changing models, inputs or utility."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.config import SystemConfig
from gsdeploy import normalization as norm
from experiments.prepare_activation_pressure import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args();reference=args.reference.resolve()
    source=reference/'system-frozen.json';base=json.loads(source.read_text());output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);isolated={};receipts={}
    for task in base['workload']:
        path=reference/('isolated-'+task)/'run.json';raw=json.loads(path.read_text())
        if raw['status']!='completed':raise ValueError('Complete immutable isolated pressure anchors required')
        isolated[task]=sum(r['worker_response']['worker_latency_ms'] for r in raw['request_records'])/len(raw['request_records'])
        receipts[task]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    cases=[]
    for index,asr_rate in enumerate((1.,1.2)):
        directory=output/f'text-1.0-asr-{asr_rate:.1f}';directory.mkdir();data=copy.deepcopy(base)
        data['workload'].update(text=1.,asr=asr_rate);catalog=data['catalog'];deployment={}
        for task,rate in data['workload'].items():
            ms=catalog['models'][task]['service_time_ms'][task];states=catalog['capacity_factors'][task]
            legal=[n for n in sorted(map(int,states)) if n>0 and rate*ms/1000/states[str(n)]<=.8]
            if not legal:raise ValueError('No catalog reference at unchanged rho<=0.8')
            deployment[task]=legal[0]
        spec=catalog[norm.KEY]
        for row in spec['tasks'].values():row.update(deadline_ms=None,reference_mean_e2e_ms=None)
        spec.update(status='pending',workload_rps=copy.deepcopy(data['workload']),episode_reference_joules=None,
                    reference_deployment=deployment,reference_receipts=[],reference_seeds=[])
        spec.pop('episode_protocol',None)
        spec['provenance']=str(spec['provenance'])+'; shared text/ASR pressure predeclared before route search; own denominator required'
        data['latency_reference_ms']={t:None for t in data['workload']}
        config_path=directory/'system-pending.json';save(config_path,data);config=SystemConfig.load(config_path)
        plan=dict(deployment=deployment,routing={t:{t:q} for t,q in config.workload.items()})
        check=config.validate_plan(plan['deployment'],plan['routing']);save(directory/'reference-plan.json',plan)
        save(directory/'measurements-pending.json',dict(schema='gsdeploy.normalization_measurements.v1',
             split='normalization_development',gpu_compatibility=dict(policy='same_model',name='NVIDIA GeForce RTX 4090'),
             confirmation_seeds=spec['confirmation_seeds'],task_runs={t:[receipts[t]['path']] for t in config.workload},
             episode_runs=[str(directory/f'reference-episode-{r}'/'run.json') for r in (1,2)]))
        cases.append(dict(workload_rps=data['workload'],reference_deployment=deployment,config=str(config_path),
             episode_seeds=[6110721+10*index,6110722+10*index],
             nominal_measured_isolated_specialist_rho={t:q*isolated[t]/1000 for t,q in config.workload.items()},
             historical_generalist_total_nominal_rho=sum(q*catalog['models']['generalist_2']['service_time_ms'][t]/1000
                                                      for t,q in config.workload.items()),
             original_constraint_check=check,candidate_runner_ready=check['constraints_valid'],
             pressure_stability_qualified=False,normalization_frozen=False,optimizer_not_run=True))
    save(output/'design.json',dict(schema='gsdeploy.shared_task_pressure_design.v1',cases=cases,
         source_config=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),isolated_receipts=receipts,
         input_pool_changed=False,quality_anchors_changed=False,utility_weights_changed=False,
         pure_specialist_not_disabled=True,replica_cost_not_invented=True,
         motivation='OCR-only stress gives G2 a very slow alternative; shared text and ASR pressure also tests the common-owner FIFO tradeoff',
         selection_scope='predeclared workload contrasts; no promised internal-route advantage or special treatment for the new optimizer',
         previous_low_load_denominator_reused=False))
    print(json.dumps(dict(output=str(output),cases=cases),ensure_ascii=False))


if __name__=='__main__':main()
