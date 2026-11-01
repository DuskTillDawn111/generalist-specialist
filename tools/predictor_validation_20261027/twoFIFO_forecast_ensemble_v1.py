"""Finite full-cohort forecast ensembles with explicit unknown probability mass.

Draws and weights are supplied before evaluation. This module neither fits a
law nor selects seeds/candidates, calibrates risk or changes the original score.
"""
from fractions import Fraction
from copy import deepcopy
from dataclasses import dataclass
import math

from gsdeploy.forecast_errors import UnknownEvidence
from .actual_worker_world_forecast_v4 import forecast
from .actual_worker_twoFIFO_query_adapter_v5_2 import canonical_requests,clock
from .macro_response_queue_reference_v1 import encode

SCHEMA='gsdeploy.twoFIFO_finite_full_cohort_forecast_ensemble.v1'
DRAW_FIELDS={'draw_id','block_id','weight','inputs'}
INPUT_FIELDS={'plans','rates','protocol_slacks','transport_draws'}
SCORE_FIELDS=('window_seconds','startup_amortization_seconds','weights','workload_rps',
    'latency_reference_ms','memory_budget_mib','energy_reference_watts','memory_safe_factor')
SHARED_FIELDS={'initial_sender_available_ns','initial_worker_read_release_ns','initial_protocol_state',
    'continuation_state','settled_initial_state','producer_fifo','resource_provider','target_context',
    'resource_source_evidence','resource_memory_initial_state','scales_ns'}


@dataclass(frozen=True)
class UnknownForecastInputs:
    """A scientifically unsupported joint draw, retained at its original weight.

    Callers may construct this only for missing evidence, not invalid contracts
    or programming errors. In particular, compilation catches UnknownEvidence
    only; it must propagate ValueError and all other implementation failures.
    """
    reason: str
    evidence: dict | None = None

    def __post_init__(self):
        if type(self.reason) is not str or not self.reason or (self.evidence is not None and type(self.evidence) is not dict):
            raise ValueError('Explicit nonempty missing-evidence reason and optional evidence mapping required')


def need(value,message):
    if not value:raise ValueError(message)


def distribution(rows,field):
    known=[(row['weight'],field(row['forecast'])) for row in rows if row['status']=='GENERATED_WORLD']
    supported=[(weight,value) for weight,value in known if value is not None]
    mass=sum((weight for weight,value in supported),Fraction(0))
    return dict(known_probability_mass=mass,unknown_probability_mass=1-mass,
        expected_value=(sum(float(weight)*float(value) for weight,value in supported) if mass==1 else None),
        conditional_on_known_mean=(sum(float(weight)*float(value) for weight,value in supported)/float(mass)
                                   if mass else None),conditional_mean_is_not_full_expectation=True)


def evaluate(requests,draws,*,scoring_plan,quality,window_start_ns,window_end_ns,drain_end_ns,**initial_and_resources):
    """Generate every predefined draw; unsupported worlds keep their full mass.

    Every scenario shares the public cohort, routes, score, boundaries and past
    state. Only complete joint law inputs vary. Invalid code/input contracts raise;
    scientifically missing evidence is a retained UNKNOWN world.
    """
    requests=[dict(row) for row in requests];public=canonical_requests(requests)
    need(set(initial_and_resources)<=SHARED_FIELDS,
         'Only explicit forecast shared state/resources are accepted; no admission cap or cutoff override')
    a,b,d=map(clock,(window_start_ns,window_end_ns,drain_end_ns))
    need(a<b<=d and all(at.denominator==1 for at in (a,b,d)),'Original integer measurement/drain required')
    need(all(type(row['arrival_ns']) is int and a<=row['arrival_ns']<b for row in requests),
        'Complete scheduled public integer arrival cohort required')
    need(draws and all(type(row) is dict and set(row)==DRAW_FIELDS for row in draws),'Closed predefined draw catalog required')
    ids=[row['draw_id'] for row in draws]
    need(all(type(i) is str and i for i in ids) and len(set(ids))==len(ids),'Distinct explicit public draw slots required')
    weights=[]
    for draw in draws:
        need(type(draw['block_id']) is str and draw['block_id'],'Explicit training-block latent required')
        need(type(draw['weight']) in (int,Fraction) and draw['weight']>0,'Exact positive predefined draw probability required')
        need(type(draw['inputs']) is UnknownForecastInputs or
             (type(draw['inputs']) is dict and set(draw['inputs'])==INPUT_FIELDS),
             'Only joint work/rates/notices/transport or explicit missing evidence may vary by draw')
        weights.append(Fraction(draw['weight']))
    need(sum(weights,Fraction(0))==1,'Retain all predefined probability mass; no renormalization')
    # Prevent unknown transport in every world from bypassing the score contract.
    from .macro_six_term_bridge_v1 import whole_score
    whole_score(dict(original_window=encode([a,b]),original_drain_end_ns=encode(d),cutoff_ns=encode(d),
        RESULT_receipts={},unfinished_request_ids=[r['request_id'] for r in requests],offered_requests=len(requests)),
        requests,scoring_plan,quality)
    comparison_contract=dict(scoring_plan={k:deepcopy(scoring_plan[k]) for k in SCORE_FIELDS},
        quality=deepcopy(quality))
    rows=[]
    for draw,weight in zip(draws,weights):
        row=dict(draw_id=draw['draw_id'],block_id=draw['block_id'],weight=weight,offered_requests=len(requests))
        if type(draw['inputs']) is UnknownForecastInputs:
            row.update(status='UNKNOWN_WORLD',reason=draw['inputs'].reason,
                evidence=deepcopy(draw['inputs'].evidence),
                all_public_request_ids_retained=[r['request_id'] for r in requests],forecast=None)
            rows.append(row)
            continue
        try:
            result=forecast(requests,scoring_plan=scoring_plan,quality=quality,
                window_start_ns=window_start_ns,window_end_ns=window_end_ns,drain_end_ns=drain_end_ns,
                **draw['inputs'],**initial_and_resources)
            row.update(status='GENERATED_WORLD',forecast=result)
        except UnknownEvidence as exc:
            row.update(status='UNKNOWN_WORLD',reason=str(exc),evidence=getattr(exc,'evidence',None),
                all_public_request_ids_retained=[r['request_id'] for r in requests],forecast=None)
        rows.append(row)
    terms={term:distribution(rows,lambda f,t=term:f['scoring']['original_score']['terms'][t]) for term in 'QTLPME'}
    utility=distribution(rows,lambda f:f['scoring']['original_score']['full_U'])
    p95=distribution(rows,lambda f:None if f['queue_diagnostics']['reader_E2E_P95_ns'] is None else
                     clock(f['queue_diagnostics']['reader_E2E_P95_ns']))
    # A known world with incomplete requests is an observed simulated miss;
    # unknown worlds contribute an interval, never a false success or a deletion.
    risk=[]
    for request in requests:
        misses=Fraction(0);unknown=Fraction(0)
        for row in rows:
            if row['status']=='UNKNOWN_WORLD':unknown+=row['weight']
            elif request['request_id'] in row['forecast']['world']['unfinished_request_ids']:misses+=row['weight']
        risk.append(dict(request_id=request['request_id'],fixed_drain_incomplete_probability_bounds=[misses,misses+unknown]))
    return encode(dict(schema=SCHEMA,status='PREDEFINED_FINITE_ENSEMBLE_NOT_CALIBRATED_PREDICTOR',
        public_requests=public,offered_requests=len(requests),draws=rows,total_probability_mass=Fraction(1),
        score_comparison_contract=comparison_contract,
        world_unknown_probability_mass=sum((r['weight'] for r in rows if r['status']=='UNKNOWN_WORLD'),Fraction(0)),
        six_terms=terms,utility=utility,expected_world_complete_cohort_P95_ns=p95,
        mean_of_world_P95_is_not_mixture_latency_P95=True,fixed_drain_risk=risk,
        original_window=encode([a,b]),original_drain_end_ns=encode(d),
        fixed_public_denominator_and_unknown_world_mass_preserved=True,
        donor_weights_or_seeds_selected_from_error=False,unknown_worlds_renormalized=False,
        calibrated_long_term_risk=False,original_stability_evaluated=False,
        main_queue_provider_eligible=False,independent_prediction_accuracy_validated=False,
        final_gain_eligible=False,validation_labels_read=False,GPU_started=False))


def compare(baseline,candidate):
    """Compare paired draws, preserving unknown ΔU and probability intervals."""
    need(baseline.get('schema')==candidate.get('schema')==SCHEMA,'Explicit new finite ensembles required')
    for document in (baseline,candidate):
        need(type(document.get('public_requests')) is list
             and type(document.get('offered_requests')) is int
             and document['offered_requests']==len(document['public_requests']),
             'Serialized comparison must retain the complete offered denominator')
        draws=document.get('draws')
        need(type(draws) is list and draws,'Serialized comparison requires predefined draws')
        need(all(type(r) is dict and type(r.get('draw_id')) is str and r['draw_id']
             and type(r.get('block_id')) is str and r['block_id'] for r in draws),
             'Distinct explicit draw and block identities required')
        need(len({r['draw_id'] for r in draws})==len(draws),'Distinct explicit draw identities required')
        weights=[clock(r['weight']) for r in draws]
        need(all(w>0 for w in weights) and sum(weights,Fraction(0))==1,
             'Serialized comparison must retain exact positive total probability one')
        need(all(type(r.get('offered_requests')) is int
             and r['offered_requests']==document['offered_requests'] for r in draws),
             'Every draw must retain the complete offered denominator')
        need(all((r.get('status')=='GENERATED_WORLD' and type(r.get('forecast')) is dict)
             or (r.get('status')=='UNKNOWN_WORLD' and r.get('forecast') is None
                 and r.get('all_public_request_ids_retained')==[p['request_id'] for p in document['public_requests']])
             for r in draws),'Explicit generated or full-cohort unknown world status required')
    need(type(baseline.get('score_comparison_contract')) is dict
         and baseline['score_comparison_contract']==candidate.get('score_comparison_contract'),
         'Fair comparison requires identical original score normalization and common quality evidence')
    for field in ('public_requests','original_window','original_drain_end_ns'):
        if field=='public_requests':
            # Routes may change; exact request/input/task/arrival population may not.
            strip=lambda rows:[{k:v for k,v in row.items() if k!='owner'} for row in rows]
            need(strip(baseline[field])==strip(candidate[field]),'Fair comparison requires the same complete arrival/input cohort')
        else:need(baseline[field]==candidate[field],'Fair comparison requires identical original window/drain')
    left=baseline['draws'];right=candidate['draws']
    need([(r['draw_id'],r['block_id'],r['weight']) for r in left]==
         [(r['draw_id'],r['block_id'],r['weight']) for r in right],'Explicit paired joint draw scheme required')
    known=Fraction(0);positive=Fraction(0);mean=0.;deltas=[]
    for a,b in zip(left,right):
        weight=clock(a['weight']);u0=a['forecast']['scoring']['original_score']['full_U'] if a['forecast'] else None
        u1=b['forecast']['scoring']['original_score']['full_U'] if b['forecast'] else None
        delta=None if u0 is None or u1 is None else u1-u0
        if delta is not None:
            need(math.isfinite(delta),'Finite original utility difference required')
            known+=weight;mean+=float(weight)*delta
            if delta>0:positive+=weight
        deltas.append(dict(draw_id=a['draw_id'],weight=weight,delta_U=delta))
    return encode(dict(status='PAIRED_FINITE_ENSEMBLE_UTILITY_COMPARISON',draws=deltas,
        expected_delta_U=mean if known==1 else None,unknown_delta_probability_mass=1-known,
        positive_delta_probability_bounds=[positive,positive+1-known],
        candidate_better=None if known!=1 else mean>0,
        original_equal_six_term_score_reused=True,stability_evaluated=False,final_gain_eligible=False))
