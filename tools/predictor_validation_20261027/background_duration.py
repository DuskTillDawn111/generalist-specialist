"""Full-task duration-weighted fit with the frozen prediction interface."""
from __future__ import annotations
import copy
import json
from .background_statistical import SCHEMA, DEFINITION as CONTEXT_DEFINITION, family_task
from .duration_profiles import DEFINITION as DURATION_DEFINITION, fit_traces as fit_base_traces
from .profiling_analysis import digest

DEFINITION=copy.deepcopy(CONTEXT_DEFINITION)
DEFINITION.update(base_numerics=DURATION_DEFINITION, estimator='duration_weighted_relative_work_WLS.v1',
    scope='Regularized duration-weighted conditional work WLS; no independent validation, exact joint-execution MLE or causal identification')


def fit_traces(traces, *, sources=()):
    encoded=copy.deepcopy(traces)
    for trace in encoded:
        if trace.get('split')!='training' or trace.get('not_for_training') is True:
            raise ValueError('Explicit training-only traces required')
        for row in trace['requests']:
            row['family']=family_task(row['family'],row['task'])
    base=fit_base_traces(encoded,sources=sources)
    document={'schema':SCHEMA,'definition':copy.deepcopy(DEFINITION),'definition_sha256':digest(DEFINITION),
        'execution_context_sha256':base['execution_context_sha256'],'source_receipts':list(sources),
        'base_model':base,'coverage':base['coverage'],'core_enabled':False,'GPU_started':False,
        'prediction_status':{'point_prediction_available':bool(base['profiles']),
            'independently_validated':False,'physically_identified':False},
        'training_task_mixtures':[[{'family':json.loads(m['family'])[0],'task':json.loads(m['family'])[1],'count':m['count']}
            for m in members] for members in base['family_contexts'].values()]}
    document['model_sha256']=digest(document)
    return document
