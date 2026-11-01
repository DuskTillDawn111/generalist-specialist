"""Reserve a new public arrival family for future group-disjoint risk validation.

Preparation only. The training collector cannot execute validation plans; its
replacement and prediction/protocol sealing must precede any future GPU run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from gsdeploy.resources import identity
from .background_collect import context_identity, receipt, save
from .multi_native_composite_v5 import checked
from .native_queue_training_v8 import verify_sources


def prepare(template_plan_path,risk_model_path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    template_ref=receipt(template_plan_path);template=checked(template_ref);verify_sources(template)
    model_ref=receipt(risk_model_path);model=checked(model_ref)
    if (template['group_id']!='burst_A' or len(template['trace'])!=24
            or model['provenance']['group_ids']!=['burst_A','burst_B']
            or model['target']!=template['risk_target']):
        raise ValueError('Original public 24-request A template and frozen A/B risk model required')
    context=checked(template['context'])
    if context_identity(context)!=model['execution_context_sha256']:
        raise ValueError('Declared collection context and frozen risk model differ')
    # Fixed asymmetric arrival waves. Selection uses public registry only;
    # no original or future response/label enters this recipe.
    peer_order={'generalist_2#0':0,'text#1':1,'text#0':2}
    waves={10_000_000:(10_000_000,90_000_000),
           2_500_000_000:(1_800_000_000,70_000_000),
           2_550_000_000:(2_300_000_000,110_000_000),
           9_000_000_000:(8_000_000_000,130_000_000),
           9_050_000_000:(8_750_000_000,50_000_000),
           14_000_000_000:(14_000_000_000,160_000_000)}
    nontext={20_000_000:160_000_000,4_000_000_000:5_800_000_000,
             4_100_000_000:6_100_000_000,11_000_000_000:10_800_000_000,
             11_100_000_000:11_400_000_000,11_200_000_000:11_700_000_000}
    trace=[]
    for public in template['trace']:
        row=copy.deepcopy(public)
        if row['task']=='text':
            base,stagger=waves[row['offset_ns']];row['offset_ns']=base+peer_order[row['alias']]*stagger
        else:row['offset_ns']=nontext[row['offset_ns']]
        trace.append(row)
    trace.sort(key=lambda r:(r['offset_ns'],r['task'],r['alias']))
    for i,row in enumerate(trace):row['request_id']='measurement:'+str(i)
    signature=lambda rows:Counter((r['alias'],r['task'],r['pool_id']) for r in rows)
    if signature(trace)!=signature(template['trace']) or trace==template['trace']:
        raise ValueError('All original public contents/routes must survive a distinct arrival recipe')
    run_id='native_queue_validation_C_01';group='burst_C'
    if run_id in model['provenance']['run_ids'] or group in model['provenance']['group_ids']:
        raise ValueError('Prospective run and family overlap frozen model training')
    fields=('aliases','window_seconds','drain_seconds','warmup_seconds','arrival_process',
        'workload_rps','latency_reference_ms','weights','memory_safe_factor','memory_budget_mib',
        'energy_reference_watts','startup_amortization_seconds','cutoff_offsets_ns','risk_target',
        'catalog','profile_plan','input_selection','inherited_runtime_smoke')
    plan={k:copy.deepcopy(template[k]) for k in fields}
    candidate=copy.deepcopy(template['candidates'][0]);candidate['id']='shared_declared_validation_C'
    candidate['routes']={r['request_id']:r['alias'] for r in trace}
    plan.update(schema='gsdeploy.prospective_native_queue_risk_family_reservation.v10',
        split='validation',training_only=False,kind='prospective_reservation_no_observations',
        run_id=run_id,group_id=group,execution_context_sha256=template['execution_context_sha256'],
        context=template['context'],trace=trace,trace_sha256=identity(trace),candidates=[candidate],
        baseline_candidate_id=candidate['id'],frozen_risk_model=model_ref,
        public_recipe_source=template_ref,preparation_source=receipt(__file__),
        recipe=dict(name='burst_C',peer_order=list(peer_order),
            text_waves=[dict(original_offset_ns=k,new_base_ns=v[0],stagger_ns=v[1]) for k,v in waves.items()],
            nontext_offsets=[dict(original_offset_ns=k,new_offset_ns=v) for k,v in nontext.items()],
            selection_uses_public_inputs_and_arrivals_only=True,outcomes_used=False),
        validation_scope='FRESH_RUN_AND_NEW_ARRIVAL_FAMILY_OF_EXACT_TRAINED_PUBLIC_INPUTS',
        reserved_request_count=24,reserved_cutoff_count=17,
        collector_ready=False,protocol_sealed=False,predictions_sealed=False,
        effective_execution_context_verified=False,
        required_before_collection=[
            'Implement and source-bind a validation collector: native_queue_training_v8 rejects validation and families outside A/B',
            'Verify effective native execution/context against frozen risk model, or explicitly resolve any changed collector context',
            'Seal original risk target, all 17 cutoffs, source closure and full 24-request registry before observations',
            'Use cutoff-visible START/READY and full declared suffix for predictions; labels must remain offline and separate',
            'Retain UNKNOWN/out-of-support and censored cutoffs; report support and both classes before rank metrics'],
        six_term_utility_changed=False,stability_rule_changed=False,admission_changed=False,
        observations_available=False,GPU_started=False)
    plan['plan_sha256']=identity(plan);save(output/'plan.json',plan)
    if (len(trace)!=24 or len(plan['cutoff_offsets_ns'])!=17
            or any(not 0<=r['offset_ns']<16_000_000_000 for r in trace)
            or plan['weights']!={k:1/6 for k in 'QTLPME'}):
        raise ValueError('Complete original cohort, cutoff grid, window and objective required')
    report=dict(schema='gsdeploy.prospective_native_queue_risk_family_preparation.v10',
        status='PREPARED_RESERVATION_ONLY_COLLECTOR_AND_SEALS_REQUIRED',plan=receipt(output/'plan.json'),
        frozen_model=model_ref,public_source=template_ref,source=receipt(__file__),
        fresh_group_disjoint_from_model=True,fresh_run_disjoint_from_model=True,
        original_task_owner_input_counts_preserved=True,full_requests=24,full_cutoffs=17,
        planned_group=group,training_groups=model['provenance']['group_ids'],
        training_run_count=len(model['provenance']['run_ids']),
        family_holdout_gap='Existing two families each have two runs; leaving one family out leaves only two runs and one family, below three-run/two-group fitting support.',
        independent_validation_collected=False,physical_accuracy_established=False,
        collector_ready=False,protocol_sealed=False,predictions_sealed=False,GPU_started=False)
    save(output/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template-plan',required=True);parser.add_argument('--risk-model',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.template_plan,args.risk_model,args.output)))


if __name__=='__main__':main()
