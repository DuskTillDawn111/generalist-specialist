"""Train and leave-one-capture-out check exact conditional envelope budgets.

The output is a partial availability-surrogate component. It deliberately
does not invent handoff, initial context, next-context order, unsupported phase
or owner budgets. Those are required before the stateful predictor is numeric.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import statistics

from gsdeploy.forecast_errors import UnknownEvidence
from .native_collect import receipt,save

SCHEMA='gsdeploy.conditional_envelope_budget_fit.v8'


def group_key(row):
    return row['alias'],row['task'],row['pool_id'],row['phase'],tuple(row['continuous_unfinished_kernel_peers'])


def derive(sample_reports):
    if len(sample_reports)!=2:raise ValueError('Two independent training captures required')
    sources=[receipt(__file__)];groups=defaultdict(lambda:defaultdict(list));raw_ids=[];reports=[]
    for path in sample_reports:
        ref=receipt(path);r=json.loads(Path(path).read_text())
        if r.get('schema')!='gsdeploy.context_envelope_training_samples.v8' or r.get('status')!='CONSERVATIVE_TRAINING_SAMPLE_EXTRACTION':
            raise ValueError('Qualified conservative source-bound training samples required')
        for source in r['sources']:
            if receipt(source['path'])!=source:raise ValueError('Changed training sample source')
        raw=[s for s in r['sources'] if Path(s['path']).name=='raw.json']
        if len(raw)!=1:raise ValueError('One original actual capture identity required')
        raw_id=raw[0]['sha256'];raw_ids.append(raw_id);reports.append(ref);sources.extend([ref,*r['sources']])
        for s in r['samples']:
            if not s['eligible_for_conditional_surrogate_budget']:continue
            if s['phase']=='UNRESOLVED_PHASE' or s['envelope_length_ns']<=0:
                raise ValueError('Exact static phase and positive envelope budget required')
            groups[group_key(s)][raw_id].append(s['envelope_length_ns'])
    if len(set(raw_ids))!=2:raise ValueError('Two independent original raw captures required')
    fitted=[];rejected=[]
    for key,captures in sorted(groups.items()):
        row=dict(alias=key[0],task=key[1],pool_id=key[2],phase=key[3],continuous_peer_set=list(key[4]))
        if set(captures)!=set(raw_ids):
            rejected.append(dict(**row,reason='TWO_INDEPENDENT_CAPTURE_SUPPORT_REQUIRED'));continue
        medians={r:statistics.median(values) for r,values in captures.items()}
        # Capture-balanced estimator: more intervals in one run do not add
        # independent-run weight or a claim of additional replication.
        budget=round(statistics.mean(medians.values()));checks=[]
        for held in raw_ids:
            train=next(r for r in raw_ids if r!=held);prediction=medians[train]
            errors=[abs(prediction-v)/v for v in captures[held]]
            checks.append(dict(training_raw_id=train,held_out_training_raw_id=held,
                predicted_envelope_ns=prediction,sample_count=len(errors),
                sample_mean_absolute_relative_error=statistics.mean(errors),
                held_run_median_absolute_relative_error=abs(prediction-medians[held])/medians[held]))
        fitted.append(dict(**row,grant_budget_ns=budget,independent_capture_count=2,
            per_capture=[dict(raw_id=r,sample_count=len(captures[r]),median_ns=medians[r],
                              min_ns=min(captures[r]),max_ns=max(captures[r])) for r in raw_ids],
            leave_one_capture_out_training_checks=checks))
    return dict(schema=SCHEMA,status='PARTIAL_TRAINED_AVAILABILITY_SURROGATE',
        training_only=True,sample_reports=reports,sources=sources,groups=fitted,rejected=rejected,
        supported_group_count=len(fitted),parameter_aggregation='equal capture mean of within-capture medians, rounded to integer ns',
        work_domain='SOLO_ONLY_REFERENCE_NS',predicted_quantity='restore-start to save-end envelope length conditional on continuous unfinished GPU kernel backlog',
        physical_quantum_identified=False,physical_SM_service_identified=False,
        missing_context_events_filled=False,fixed_quantum_for_all_owners_fitted=False,
        handoff_parameters_available=False,next_context_order_available=False,initial_context_parameters_available=False,
        complete_stateful_policy_available=False,independent_prediction_validation=False,
        natural_domain_performance_transfer_qualified=False,validation_labels_used=False,GPU_started=False)


class ConditionalGrantBudgets:
    def __init__(self,model_path):
        self.reference=receipt(model_path);self.model=json.loads(Path(model_path).read_text())
        if self.model.get('schema')!=SCHEMA:raise ValueError('Explicit conditional envelope fit model required')
        original=self.model.get('sample_reports')
        if not original or any(receipt(r['path'])!=r for r in original):raise ValueError('Changed conditional sample evidence')
        if derive([r['path'] for r in original])!=self.model:
            raise ValueError('Conditional grant parameters do not rederive from training sources')
        self.groups={(g['alias'],g['task'],g['pool_id'],g['phase'],tuple(g['continuous_peer_set'])):g for g in self.model['groups']}

    def budget_ns(self,alias,task,pool_id,phase,continuous_peer_set):
        row=self.groups.get((alias,task,pool_id,phase,tuple(sorted(continuous_peer_set))))
        if row is None:raise UnknownEvidence('UNSUPPORTED_EXACT_CONDITIONAL_ENVELOPE_BUDGET')
        return row['grant_budget_ns']

    def provenance(self):return dict(model=self.reference,complete_stateful_policy_available=False,physical_quantum_identified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    model=derive(a.samples);save(out/'model.json',model)
    verified=ConditionalGrantBudgets(out/'model.json')
    for row in model['groups']:
        assert verified.budget_ns(row['alias'],row['task'],row['pool_id'],row['phase'],row['continuous_peer_set'])==row['grant_budget_ns']
    print(json.dumps(dict(status=model['status'],supported_group_count=model['supported_group_count'],groups=model['groups']),indent=2))


if __name__=='__main__':main()
