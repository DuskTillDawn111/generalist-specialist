"""Predeclare OCR pressure cases with their own pending energy references."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.config import SystemConfig
from gsdeploy import normalization as norm


def save(path,value):
    with path.open('x') as out:json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args();reference=args.reference.resolve()
    source=reference/'system-frozen.json';base=json.loads(source.read_text());output=args.output.resolve()
    isolated_path=reference/'isolated-ocr/run.json';isolated=json.loads(isolated_path.read_text())
    measured_ms=sum(r['worker_response']['worker_latency_ms'] for r in isolated['request_records'])/len(isolated['request_records'])
    output.mkdir(parents=True,exist_ok=False);cases=[]
    for index,rate in enumerate((1.5,1.9)):
        directory=output/f'ocr-rps-{rate:.1f}';directory.mkdir();data=copy.deepcopy(base)
        data['workload']['ocr']=rate;catalog=data['catalog'];spec=catalog[norm.KEY]
        original_ms=catalog['models']['ocr']['service_time_ms']['ocr']
        # Keep the original family-normalized rho/g(n) test. Do not infer g(2)
        # from short probes, nor pretend its catalog values are GPU certification.
        states=catalog['capacity_factors']['ocr'];allowed=sorted(int(n) for n in states if int(n)>0)
        copies=next(n for n in allowed if rate*original_ms/1000/states[str(n)]<=.8)
        deployment=dict(text=1,image=1,ocr=copies,asr=1)
        for task,row in spec['tasks'].items():
            row['deadline_ms']=None;row['reference_mean_e2e_ms']=None
        spec.update(status='pending',workload_rps=copy.deepcopy(data['workload']),episode_reference_joules=None,
            reference_deployment=deployment,reference_receipts=[],reference_seeds=[])
        spec.pop('episode_protocol',None)
        spec['provenance']=str(spec['provenance'])+'; new predeclared OCR pressure workload; fresh episode denominator required'
        data['latency_reference_ms']={t:None for t in data['workload']}
        config_path=directory/'system-pending.json';save(config_path,data);config=SystemConfig.load(config_path)
        plan=dict(deployment=deployment,routing={t:{t:q} for t,q in config.workload.items()})
        check=config.validate_plan(plan['deployment'],plan['routing'])
        # Old low-load calibrated replay must remain UNKNOWN for changed rates.
        # Record this prerequisite, rather than rewriting its workload binding.
        save(directory/'reference-plan.json',plan)
        manifest=dict(schema='gsdeploy.normalization_measurements.v1',split='normalization_development',
            gpu_compatibility=dict(policy='same_model',name='NVIDIA GeForce RTX 4090'),
            confirmation_seeds=spec['confirmation_seeds'],
            task_runs={t:[str(reference/('isolated-'+t)/'run.json')] for t in config.workload},
            episode_runs=[str(directory/f'reference-episode-{i}'/'run.json') for i in (1,2)])
        save(directory/'measurements-pending.json',manifest)
        case=dict(ocr_rps=rate,reference_deployment=deployment,config=str(config_path),
            reference_plan=str(directory/'reference-plan.json'),episode_seeds=[6110581+10*index,6110582+10*index],
            nominal_catalog_rho_single=rate*original_ms/1000,
            nominal_measured_isolated_rho_single=rate*measured_ms/1000,
            original_constraint_check=check,matching_performance_evidence_required=not check['constraints_valid'],
            candidate_runner_ready=check['constraints_valid'],capacity_qualified=False,normalization_frozen=False,optimizer_not_run=True)
        cases.append(case)
    save(output/'design.json',dict(schema='gsdeploy.activation_pressure_design.v1',cases=cases,
        source_config=str(source),source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        isolated_ocr_source=str(isolated_path),isolated_ocr_sha256=hashlib.sha256(isolated_path.read_bytes()).hexdigest(),
        input_pool_changed=False,quality_anchors_changed=False,utility_weights_changed=False,
        baseline_rule='fewest retained specialist OCR states satisfying the unchanged catalog rho<=0.8; measured stability still required',
        selection_scope='predeclared pressure contrasts before optimizer search; no guaranteed interior-routing advantage',
        previous_low_load_denominator_reused=False))
    print(json.dumps(dict(output=str(output),cases=cases),ensure_ascii=False))


if __name__=='__main__':main()
