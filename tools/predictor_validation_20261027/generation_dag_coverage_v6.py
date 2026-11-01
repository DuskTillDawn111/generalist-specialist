"""Exact public-content work scenarios and known-input DAG support."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from .generation_dag_coverage_v5 import select_scenarios as select_population
from .generation_work_v4 import input_content_identity
from .generation_work_v5 import ExactContentGenerationWorkProvider
from .kernel_domain_bridge_review_v3 import load_report
from .native_collect import receipt,save


def select_scenarios(provider,public_request,templates):
    request=copy.deepcopy(public_request)
    content=request.pop('input_content_sha256')
    class ContentProvider:
        def scenarios(self,public):return provider.scenarios(dict(public,input_content_sha256=content))
    matching=[t for t in templates if t['input_content_sha256']==content]
    result=select_population(ContentProvider(),request,matching)
    result.update(input_content_sha256=content,scope='EXACT_KNOWN_PUBLIC_CONTENT_WORK_SUPPORT',
                  unseen_input_generalization_claimed=False)
    return result


def review(previous_review,generation_path,output):
    previous=json.loads(Path(previous_review).read_text());ref=previous['template_descriptors']
    if receipt(ref['path'])!=ref:raise ValueError('Changed original training work descriptors')
    descriptor=json.loads(Path(ref['path']).read_text())
    report,plan,raw=load_report(previous['light_review']['path'],'light-kernel')
    context=json.loads(Path(report['context']['path']).read_text())
    refs=[s['frozen'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(refs)!=1 or receipt(refs[0]['path'])!=refs[0]:raise ValueError('Exact immutable public input catalog required')
    pools={p['pool_id']:p for p in map(json.loads,Path(refs[0]['path']).read_text().splitlines())}
    templates=[dict(t,input_content_sha256=input_content_identity(pools[t['pool_id']])) for t in descriptor['templates']]
    generation=json.loads(Path(generation_path).read_text())
    projection={p['native_execution_projection_sha256'] for p in generation['profiles']}
    if len(projection)!=1:raise ValueError('One explicit generation projection required')
    provider=ExactContentGenerationWorkProvider(generation,projection.pop(),allow_exact_content_hypothesis=True)
    results=[]
    for original in previous['results']:
        public=dict(original['public_request']);public['input_content_sha256']=input_content_identity(pools[public['pool_id']])
        results.append(dict(public_request=public,coverage=select_scenarios(provider,public,templates)))
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'template-descriptors.json',dict(templates=templates,non_autoregressive=descriptor['non_autoregressive']))
    result=dict(schema='gsdeploy.exact_content_generation_to_known_input_dag_support_review.v6',
        previous_population_review=receipt(previous_review),generation_model=receipt(generation_path),
        public_input_catalog=refs[0],template_descriptors=receipt(out/'template-descriptors.json'),results=results,
        sources=[receipt(__file__)],future_actual_output_used=False,unseen_input_generalization_claimed=False,
        independent_prediction_accuracy_validated=False,GPU_started=False)
    save(out/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('previous-review','generation','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();r=review(a.previous_review,a.generation,a.output)
    print(json.dumps([dict(alias=x['public_request']['alias'],status=x['coverage']['status'],
        mass=x['coverage']['covered_probability_mass']) for x in r['results']]))


if __name__=='__main__':main()
