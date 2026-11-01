"""Explicit public181 training-output quality adapter; never a predictor.

Pure scoring uses original task error/reference totals. Native authentication
belongs to the resource review/intake, not this helper or a caller flag.
"""
from __future__ import annotations
import copy
import math

from gsdeploy.quality import sample_totals
from gsdeploy.tasks import page_ocr_text,score_result
from .twoFIFO_native_quality_labels_v1 import prepared_references
from .twoFIFO_training_manifest_v1 import require
from .stage_response_features_v1 import checked,receipt

SCHEMA='gsdeploy.twoFIFO_public181_actual_training_quality.v1'


def measure(raw,plan,raw_ref):
    from . import twoFIFO_public181_resource_execute_v1 as executor
    require(raw.get('schema')==executor.RAW_SCHEMA and raw.get('split')=='training'
        and raw.get('backend')=='native' and raw.get('kind')=='measured'
        and raw.get('status')=='completed' and plan.get('schema')==executor.SCHEMA+'.attempt'
        and plan.get('split')=='training','Explicit complete native public181 training roles required')
    require(all(raw.get(k)==plan[k] for k in ('attempt_id','execution_id','capture_id')),
            'Exact native quality source identity required')
    require(checked(raw['plan'])==plan,'Exact immutable quality plan required')
    require(receipt(raw_ref['path'])==raw_ref and checked(raw_ref)==raw,
            'Quality output source bytes differ')
    catalog_ref=plan['quality_catalog'];references=prepared_references(plan,catalog_ref)
    contract_ref=plan['public_case']['public_contract'];contract=checked(contract_ref)
    rates=contract['workload_rps']
    require(rates and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in rates.values()),
            'Original positive offered rates required')
    require(len(plan['cases'])==len(raw['measurement_windows'])==len(raw['cases'])==1,
            'One original public181 quality case required')
    case=plan['cases'][0];window=raw['measurement_windows'][0];epoch=window['start_ns']
    require(window['case_id']==case['case_id'] and window['end_ns']==epoch+executor.MEASUREMENT_NS
        and window['fixed_drain_deadline_ns']==epoch+executor.MEASUREMENT_NS+executor.DRAIN_NS,
        'Exact original public181 quality horizon required')
    actual=raw['request_records'];arrivals=case['arrivals']
    require(len(actual)==len(arrivals)==181 and len({r['request_id'] for r in actual})==181
        and [r['request_id'] for r in actual]==[a['request_id'] for a in arrivals],
        'Complete ordered public181 quality denominator required')
    totals={t:dict(numerator=0.,denominator=0.,offered=0,scored=0,missing=[]) for t in rates}
    outputs=[]
    for index,(row,arrival) in enumerate(zip(actual,arrivals)):
        task=arrival['task'];require(task in rates,'Unexpected native quality task')
        require((row['model'],row['task'],row['pool_id'],row['scheduled_time_ns'])==
            (arrival['owner'],task,arrival['pool_id'],epoch+arrival['arrival_ns']),
            'Native output differs from original route/input/arrival')
        target=totals[task];target['offered']+=1
        output=dict(request_id=row['request_id'],task=task,pool_id=row['pool_id'],owner=row['model'],
            native_output_source=dict(raw=raw_ref,json_pointer='/request_records/'+str(index)+'/worker_response'),
            numerator=None,denominator=None,status='UNKNOWN_NATIVE_OUTCOME')
        complete=(row.get('status')=='completed' and type(row.get('completion_time_ns')) is int
            and row['scheduled_time_ns']<=row['completion_time_ns']<=window['fixed_drain_deadline_ns'])
        if complete:
            response=copy.deepcopy(row.get('worker_response',{}))
            require(response.get('task')==task and response.get('request_id')==row['request_id']
                and type(response.get('success')) is bool,'Exact native task/request/success output binding required')
            complete=response['success'] is True
        if complete:
            reference=references[task,row['pool_id']]
            require(row.get('sample_id')==reference['id'],'Native quality sample identity changed')
            if task=='ocr':response['prediction']=page_ocr_text(response['prediction'])
            numerator,denominator=sample_totals(task,score_result(reference,response))
            require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (numerator,denominator)),
                    'Finite nonnegative original quality counts required')
            target['numerator']+=numerator;target['denominator']+=denominator;target['scored']+=1
            output.update(numerator=numerator,denominator=denominator,status='MEASURED_NATIVE_OUTCOME')
        else:target['missing'].append(row['request_id'])
        outputs.append(output)
    reasons=[]
    for task,total in totals.items():
        total['loss']=(total['numerator']/total['denominator'] if total['offered'] and
            not total['missing'] and total['denominator']>0 else None)
        if total['loss'] is None:reasons.append('UNKNOWN_NATIVE_QUALITY:'+task)
    Q=(math.fsum(rates[t]/math.fsum(rates.values())*totals[t]['loss'] for t in rates)
       if not reasons else None)
    require(receipt(raw_ref['path'])==raw_ref and receipt(catalog_ref['path'])==catalog_ref
        and receipt(contract_ref['path'])==contract_ref,'Native quality evidence changed')
    return dict(schema=SCHEMA,status='ACTUAL_TRAINING_OUTPUT_LABEL_NOT_PREDICTION',
        attempt_id=plan['attempt_id'],case_id=case['case_id'],measured_Q=Q,
        per_task=totals,requests=outputs,offered_requests=181,unknown_reasons=reasons,
        original_offered_rates=rates,quality_catalog=catalog_ref,public_contract=contract_ref,
        numerator_denominator_summed_before_division=True,training_expectation_fallback_used=False,
        native_authentication_by_pure_helper=False,quality_model_fitted=False,
        future_outputs_used_for_evaluation_only=True,validation_labels_read=False,
        full_U_known_by_quality_alone=False,final_gain_eligible=False)
