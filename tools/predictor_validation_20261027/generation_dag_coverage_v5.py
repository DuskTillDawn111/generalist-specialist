"""Public-only empirical work scenarios versus exact known-input DAG support.

Training completions describe frozen templates. No future completion may select
a template; every positive-mass unsupported scenario remains explicit UNKNOWN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from .generation_work_v4 import GenerationWorkProvider,WORK_FIELDS,actual_work
from .kernel_domain_bridge_review_v3 import load_report
from .native_collect import receipt,save


def work_key(work):return tuple(work[field] for field in WORK_FIELDS)


def select_scenarios(provider,public_request,templates):
    """Map positive training-distribution mass without dropping unknown shapes."""
    expected={'alias','pool_id','input_tokens','family','task','protocol_sha256',
              'effective_generation','native_execution_projection_sha256','request_id'}
    if set(public_request)!=expected:raise ValueError('Only explicit public known-input request features accepted')
    request={key:value for key,value in public_request.items() if key not in ('alias','pool_id')}
    result=provider.scenarios(request);outcomes=[]
    for scenario in result['scenarios']:
        matches=[template for template in templates if template['alias']==public_request['alias']
            and template['pool_id']==public_request['pool_id']
            and template['input_tokens']==public_request['input_tokens']
            and template['protocol_sha256']==public_request['protocol_sha256']
            and template['effective_generation']==public_request['effective_generation']
            and work_key(template['work'])==work_key(scenario)]
        if len(matches)>1:raise ValueError('Ambiguous exact known-input work template')
        outcomes.append(dict(scenario=scenario,status='EXACT_TRAINING_TEMPLATE_AVAILABLE' if matches else 'UNKNOWN',
            template=matches[0]['template'] if matches else None,
            reason=None if matches else 'NO_EXACT_KNOWN_INPUT_DAG_FOR_EMPIRICAL_WORK_SCENARIO'))
    complete=all(outcome['status']!='UNKNOWN' for outcome in outcomes)
    return dict(status='COMPLETE_CONDITIONAL_WORK_SUPPORT' if complete else 'UNKNOWN_INCOMPLETE_WORK_SCENARIOS',
        outcomes=outcomes,covered_probability_mass=sum(o['scenario']['weight'] for o in outcomes if o['status']!='UNKNOWN'),
        unconditional_forecast_complete=complete,future_actual_output_used=False,
        unknown_scenario_mass_dropped=False,physical_accuracy_validated=False)


def review(dag_path,generation_path,light_review_path,output):
    dag=json.loads(Path(dag_path).read_text());generation=json.loads(Path(generation_path).read_text())
    report,plan,raw=load_report(light_review_path,'light-kernel')
    if dag.get('training_only') is not True or raw.get('kind')!='measured':raise ValueError('Measured training template evidence required')
    rows={(r['model'],r['task'],r['pool_id'],r['request_id']):r for r in raw['request_records']}
    descriptors=[];ocr=[]
    for profile in dag['profiles']:
        key=profile['alias'],profile['task'],profile['pool_id'],profile['request_id'];row=rows[key]
        if profile['source_sqlite']!=report['owners'][profile['alias']]['sqlite']:
            raise ValueError('DAG does not bind the exact training owner source')
        if row['split']!='training':raise ValueError('Future validation cannot describe training work')
        if row['task']=='ocr':ocr.append(dict(alias=profile['alias'],pool_id=profile['pool_id'],status='NON_AUTOREGRESSIVE_OCR_SEPARATE'));continue
        response=row['worker_response'];work=actual_work(response,row['request_id'])
        descriptors.append(dict(alias=profile['alias'],family=profile['family'],task=profile['task'],pool_id=profile['pool_id'],
            input_tokens=response['input_tokens'],protocol_sha256=response['protocol_sha256'],
            effective_generation=response['effective_generation'],work=work,
            template=dict(dag=receipt(dag_path),alias=profile['alias'],pool_id=profile['pool_id'],task=profile['task'])))
    projections={p['native_execution_projection_sha256'] for p in generation['profiles']}
    if len(projections)!=1:raise ValueError('Explicit unique natural generation projection required')
    projection=projections.pop();provider=GenerationWorkProvider(generation,projection,allow_empirical_length_hypothesis=True)
    results=[]
    for index,descriptor in enumerate(descriptors):
        public={key:descriptor[key] for key in ('alias','pool_id','family','task','input_tokens','protocol_sha256','effective_generation')}
        public.update(native_execution_projection_sha256=projection,request_id=f'public-known-template:{index}')
        try:coverage=select_scenarios(provider,public,descriptors)
        except UnknownEvidence as exc:coverage=dict(status='UNKNOWN',reason=str(exc),future_actual_output_used=False)
        results.append(dict(public_request=public,coverage=coverage))
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'template-descriptors.json',dict(templates=descriptors,non_autoregressive=ocr))
    result=dict(schema='gsdeploy.generation_to_known_input_dag_support_review.v5',results=results,
        generation_model=receipt(generation_path),dag=receipt(dag_path),light_review=receipt(light_review_path),
        template_descriptors=receipt(out/'template-descriptors.json'),sources=[receipt(__file__)],
        template_work_comes_from_training_only=True,future_actual_output_used=False,
        GPU_started=False,independent_prediction_accuracy_validated=False)
    save(out/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dag','generation','light-review','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=review(a.dag,a.generation,a.light_review,a.output)
    print(json.dumps([dict(alias=r['public_request']['alias'],pool_id=r['public_request']['pool_id'],
        status=r['coverage']['status'],mass=r['coverage'].get('covered_probability_mass')) for r in result['results']]))


if __name__=='__main__':main()
