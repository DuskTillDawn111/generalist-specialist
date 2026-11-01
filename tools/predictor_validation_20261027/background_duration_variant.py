"""Full-task duration-weighted fit with the frozen prediction interface."""
from __future__ import annotations
import copy
import json
from .background_statistical import SCHEMA, DEFINITION as CONTEXT_DEFINITION, family_task
from .duration_variant_profiles import variant_definition, fit_traces as fit_base_traces
from .profiling_analysis import digest

def background_definition(variant):
    definition=copy.deepcopy(CONTEXT_DEFINITION)
    definition.update(base_numerics=variant_definition(variant),estimator='duration_variants_WLS.v1',variant=variant,
        scope='Fixed training-only duration variant; no independent validation, causal identification or final gain claim')
    return definition


def fit_traces(traces, *, sources=(), variant):
    definition=background_definition(variant)
    encoded=copy.deepcopy(traces)
    for trace in encoded:
        if trace.get('split')!='training' or trace.get('not_for_training') is True:
            raise ValueError('Explicit training-only traces required')
        for row in trace['requests']:
            row['family']=family_task(row['family'],row['task'])
    base=fit_base_traces(encoded,sources=sources,variant=variant)
    document={'schema':SCHEMA,'definition':copy.deepcopy(definition),'definition_sha256':digest(definition),
        'execution_context_sha256':base['execution_context_sha256'],'source_receipts':list(sources),
        'base_model':base,'coverage':base['coverage'],'core_enabled':False,'GPU_started':False,
        'prediction_status':{'point_prediction_available':bool(base['profiles']),
            'independently_validated':False,'physically_identified':False},
        'training_task_mixtures':[[{'family':json.loads(m['family'])[0],'task':json.loads(m['family'])[1],'count':m['count']}
            for m in members] for members in base['family_contexts'].values()]}
    document['model_sha256']=digest(document)
    return document
