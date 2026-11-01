"""Validate deterministic exact-content training work before DAG execution."""
from __future__ import annotations
import json
from pathlib import Path
from gsdeploy.forecast_errors import UnknownEvidence
from .generation_dag_coverage_v6 import select_scenarios
from .generation_work_v4 import input_content_identity
from .generation_work_v5 import ExactContentGenerationWorkProvider
from .native_collect import receipt
from .native_execution_projection_v5 import native_execution_projection


def checked(ref):
    if receipt(ref['path'])!=ref:raise ValueError('Changed exact-content DAG work evidence')
    return json.loads(Path(ref['path']).read_text())


def check(bundle,contract,aliases,rows):
    if bundle.get('allow_exact_content_generation_hypothesis') is not True:
        raise ValueError('Exact-content generation work requires explicit conditioning opt-in')
    context=checked(contract['target']['context']);projection=native_execution_projection(context)['projection_sha256']
    model=checked(bundle['artifacts']['generation_work']);support=checked(bundle['artifacts']['generation_dag_support'])
    if support['generation_model']!=bundle['artifacts']['generation_work']:
        raise ValueError('DAG work support must bind this exact generation model')
    catalog_ref=support['public_input_catalog']
    if receipt(catalog_ref['path'])!=catalog_ref:raise ValueError('Changed public content catalog')
    context_catalog=[s['frozen'] for s in context['execution_sources'] if s['original_path'].endswith('/runtime-inputs.jsonl')]
    if len(context_catalog)!=1 or context_catalog[0]!=catalog_ref:
        raise ValueError('Work support must use the target runtime exact public catalog')
    pools={p['pool_id']:p for p in map(json.loads,Path(catalog_ref['path']).read_text().splitlines())}
    descriptor=checked(support['template_descriptors']);templates=descriptor['templates']
    provider=ExactContentGenerationWorkProvider(model,projection,allow_exact_content_hypothesis=True)
    results=[]
    for row in rows:
        content=input_content_identity(pools[row['pool_id']])
        if row['task']=='ocr':
            if not any(t['alias']==row['alias'] and t['pool_id']==row['pool_id'] for t in descriptor['non_autoregressive']):
                raise UnknownEvidence('NO_EXACT_TRAINING_NON_AUTOREGRESSIVE_OCR_TEMPLATE')
            results.append(dict(request_id=row['request_id'],kind='NON_AUTOREGRESSIVE_OCR_SEPARATE',input_content_sha256=content));continue
        matches=[t for t in templates if t['alias']==row['alias'] and t['task']==row['task'] and t['pool_id']==row['pool_id']
                 and t['input_content_sha256']==content]
        if len(matches)!=1:raise UnknownEvidence('NO_UNIQUE_EXACT_CONTENT_DAG_PUBLIC_FEATURES')
        features=matches[0]
        public={key:features[key] for key in ('alias','pool_id','family','task','input_tokens','protocol_sha256','effective_generation','input_content_sha256')}
        if public['family']!=aliases[row['alias']]['family']:raise ValueError('Exact-content work family differs from route')
        public.update(native_execution_projection_sha256=projection,request_id=row['request_id'])
        result=select_scenarios(provider,public,templates)
        if (not result['unconditional_forecast_complete'] or len(result['outcomes'])!=1
                or result['outcomes'][0]['scenario']['weight']!=1.):
            raise UnknownEvidence('COMPLETE_SINGLE_EXACT_CONTENT_WORK_SCENARIO_REQUIRED_FOR_CURRENT_DAG_ENGINE')
        results.append(dict(request_id=row['request_id'],public_features=public,work=result['outcomes'][0]['scenario'],
            input_token_feature_basis='Immutable exact public content and source-bound training tokenization',
            future_actual_output_used=False))
    return dict(scope='EXACT_KNOWN_PUBLIC_CONTENT_NEW_REQUEST_GENERATION_HYPOTHESIS',rows=results,
        unseen_input_generalization_claimed=False,future_actual_output_used=False,all_declared_requests_retained=True)
