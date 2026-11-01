"""Observable request completion, without a physical context/service gate.

The target is START-reader to RESULT-reader wall time. Peer activity means
reader-visible START without RESULT. It does not identify SM service, worker
availability, or resource throughput. Dynamic peers are covariates, not an
assumed causal intervention. Fixed-peer landmark predictions remain hypotheses.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from gsdeploy.forecast_errors import UnknownEvidence
from .implementation_source_closure_v27 import implementation_sources, strict_references
from .native_collect import receipt
from .profiling import digest
from .unknown_preflight_archive_v1 import save_synced
from .whole57_shared_observable_risk_v31 import OWNERS

SCHEMA = 'gsdeploy.reader_handler_completion_hazard.v46'
DATA_SCHEMA = 'gsdeploy.natural_reader_handler_completion_risk.v45'
ROUTES = ('balanced', 'skewed', 'spill_low', 'spill_high')
AGE_EDGES_NS = tuple(round(s*1e9) for s in (0, .125, .25, .5, 1, 2, 4, 8, 16, 32, 64, 128))
LANDMARK_AGES_NS = tuple(round(s*1e9) for s in (0, .1, .25, .5, 1, 2, 4, 8, 16, 32))
HORIZONS_NS = tuple(round(s*1e9) for s in (.1, 1, 5, 10))
PARAMETERS = dict(target='reader_START_to_reader_RESULT_wall_time',
    age_edges_ns=list(AGE_EDGES_NS), landmarks_ns=list(LANDMARK_AGES_NS), horizons_ns=list(HORIZONS_NS),
    training_population='measurement_only_priming_retained_as_history_not_fitted',
    loss='capture_event_normalized_Poisson_plus_L2_over_training_event_count',
    interactions='alias_by_complete_reader_active_mask',
    baseline='same_fixed_age_bins_and_alias_input_without_peer_state',
    hyperparameters_selected_from_holdout=False,
    landmark_peer_future='current_reader_peer_state_held_fixed_explicit_unvalidated_hypothesis',
    physical_context_execution_gate=False, phase_equals_CUDA_execution=False,
    reader_RESULT_equals_worker_READY=False, independently_validated=False)


def checked(ref):
    if set(ref) != {'path', 'sha256'} or receipt(ref['path']) != ref:
        raise ValueError('Exact immutable source receipt required')
    return json.loads(Path(ref['path']).read_text())


def features(value):
    if (set(value) != {'alias', 'task', 'pool_id', 'attained_reader_handler_age_ns', 'active_handler_by_owner'}
            or value['alias'] not in OWNERS or not isinstance(value['task'], str)
            or not isinstance(value['pool_id'], str)
            or type(value['attained_reader_handler_age_ns']) is not int
            or value['attained_reader_handler_age_ns'] < 0
            or set(value['active_handler_by_owner']) != set(OWNERS)
            or any(type(x) is not bool for x in value['active_handler_by_owner'].values())
            or value['active_handler_by_owner'][value['alias']] is not True):
        raise ValueError('Complete typed reader-visible active state, input and attained age required')
    return value


def tokens(value, include_peers):
    features(value)
    result = ['alias:'+value['alias'], 'alias_input:'+digest([value['alias'], value['task'], value['pool_id']])]
    if include_peers:
        bits = ''.join('1' if value['active_handler_by_owner'][o] else '0' for o in OWNERS)
        result.append('alias_reader_mask:'+value['alias']+':'+bits)
        result += ['reader_active:'+o+':'+str(value['active_handler_by_owner'][o]).lower() for o in OWNERS]
    return tuple(sorted(result))


def split_age(begin, end):
    if type(begin) is not int or type(end) is not int or not 0 <= begin < end:
        raise ValueError('Strictly positive ordered age exposure required')
    while begin < end:
        i = bisect_right(AGE_EDGES_NS, begin)-1
        stop = min(end, AGE_EDGES_NS[i+1]) if i+1 < len(AGE_EDGES_NS) else end
        yield i, stop-begin, stop == end
        begin = stop


def validate_row(row):
    f = features(row['features'])
    if (any(type(row[k]) is not int for k in ('start_ns','end_ns','age_entry_ns','age_exit_ns'))
            or not 0 <= row['age_entry_ns'] < row['age_exit_ns']
            or not row['start_ns'] < row['end_ns']
            or row['end_ns']-row['start_ns'] != row['age_exit_ns']-row['age_entry_ns']
            or f['attained_reader_handler_age_ns'] != row['age_entry_ns']
            or type(row['completion_observed']) is not bool
            or row['cohort'] not in ('measured', 'priming')
            or not isinstance(row.get('capture'), str) or not row['capture']
            or not isinstance(row.get('request_id'), str) or not row['request_id']
            or type(row.get('origin_reader_START_ns')) is not int
            or row['start_ns']-row['origin_reader_START_ns'] != row['age_entry_ns']):
        raise ValueError('Exact reader interval identity, positive exposure and terminal event required')


def merge_intervals(spans):
    result = []
    for lo, hi in sorted(spans):
        if result and lo <= result[-1][1]:
            result[-1][1] = max(result[-1][1], hi)
        else:
            result.append([lo, hi])
    return result


def positive_rate(log_rate):
    try:
        value=math.exp(log_rate)
    except OverflowError as exc:
        raise UnknownEvidence('NONFINITE_READER_COMPLETION_HAZARD') from exc
    if not math.isfinite(value) or value<=0:
        raise UnknownEvidence('NONFINITE_OR_UNDERFLOWED_READER_COMPLETION_HAZARD')
    return value


def aggregate(rows, include_peers):
    bins = defaultdict(lambda: [0,0]); spans = defaultdict(list); counts = Counter(); by_capture = Counter()
    for row in rows:
        validate_row(row)
        if row['cohort'] != 'measured':
            raise ValueError('Explicit measurement training population only')
        fs = tokens(row['features'], include_peers)
        spans[digest(fs)].append([row['age_entry_ns'], row['age_exit_ns']])
        for index, width, final in split_age(row['age_entry_ns'], row['age_exit_ns']):
            cell = bins[row['capture'], index, fs]
            cell[0] += width; cell[1] += int(row['completion_observed'] and final)
        counts['risk_rows'] += 1
        counts['exposure_ns'] += row['end_ns']-row['start_ns']
        counts['completion_events'] += int(row['completion_observed'])
        by_capture[row['capture']] += int(row['completion_observed'])
    return dict(rows=[dict(capture=c, age_bin=i, tokens=list(ts), exposure_ns=v[0], events=v[1])
        for (c,i,ts),v in sorted(bins.items())], counts=dict(counts),
        events_by_capture=dict(by_capture), state_age_support_ns={k:merge_intervals(v) for k,v in spans.items()})


def matrix(rows, columns):
    index = {c:i for i,c in enumerate(columns)}; rr=[]; cc=[]
    for j,row in enumerate(rows):
        for name in ['intercept', 'age_bin:'+str(row['age_bin']), *row['tokens']]:
            if name in index: rr.append(j); cc.append(index[name])
    return csr_matrix((np.ones(len(rr)),(rr,cc)), shape=(len(rows),len(columns)))


def fit(rows, *, include_peers):
    episodes(rows)
    data = aggregate(rows, include_peers); cells = data['rows']; counts = data['events_by_capture']
    if not cells or not counts or any(n <= 0 for n in counts.values()):
        raise ValueError('Positive measured completion support in every capture required')
    vocabulary = sorted({t for row in cells for t in row['tokens']})
    columns = ['intercept', *['age_bin:'+str(i) for i in range(len(AGE_EDGES_NS))], *vocabulary]
    x=matrix(cells,columns); exposure=np.asarray([r['exposure_ns']/1e9 for r in cells]); events=np.asarray([r['events'] for r in cells])
    weights=np.asarray([1/(len(counts)*counts[r['capture']]) for r in cells]); penalty=1/sum(counts.values())
    initial=np.zeros(len(columns));initial[0]=math.log(np.dot(weights,events)/np.dot(weights,exposure))
    def objective(beta):
        eta=x@beta
        with np.errstate(over='ignore'): expected=exposure*np.exp(eta)
        if not np.isfinite(expected).all():return float('inf'),np.zeros_like(beta)
        loss=float(np.dot(weights,expected-events*eta)+.5*penalty*np.dot(beta[1:],beta[1:]))
        gradient=np.asarray(x.T@(weights*(expected-events)));gradient[1:]+=penalty*beta[1:]
        return loss,gradient
    def hessian(beta):
        value=(x.T@x.multiply((weights*exposure*np.exp(x@beta))[:,None])).toarray()
        value[np.diag_indices_from(value)]+=np.r_[0.,np.repeat(penalty,len(columns)-1)]
        return value
    fitted=minimize(objective,initial,jac=True,hess=hessian,method='trust-exact',options={'maxiter':2000,'gtol':1e-7})
    if not fitted.success or not np.isfinite(fitted.x).all():raise ValueError('Declared completion fit did not converge')
    return dict(columns=columns,coefficients=fitted.x.tolist(),include_peers=include_peers,
        vocabulary=vocabulary,age_edges_ns=list(AGE_EDGES_NS),state_age_support_ns=data['state_age_support_ns'],
        training_counts=data['counts'],events_by_capture=counts,regularization=penalty,
        optimizer=dict(method='trust-exact',success=bool(fitted.success),iterations=int(fitted.nit),
            objective=float(fitted.fun),gradient_max_abs=float(np.abs(fitted.jac).max())))


class ReaderCompletionCandidate:
    def __init__(self, model):
        if (model.get('age_edges_ns') != list(AGE_EDGES_NS)
                or len(model['columns']) != len(set(model['columns']))
                or len(model['columns']) != len(model['coefficients'])
                or not np.isfinite(model['coefficients']).all()):
            raise ValueError('Complete finite fixed-age completion candidate required')
        self.model = model
        self.coefs = dict(zip(model['columns'], model['coefficients']))

    def base(self,value):
        fs=tokens(value,self.model['include_peers']); age=value['attained_reader_handler_age_ns']
        flags=[]
        if set(fs)-set(self.model['vocabulary']): flags.append('UNSEEN_READER_STATE_OR_INPUT')
        spans=self.model['state_age_support_ns'].get(digest(fs),[])
        stop=next((hi for lo,hi in spans if lo<=age<hi),age)
        if stop==age: flags.append('ATTAINED_AGE_OUTSIDE_READER_STATE_EXPOSURE')
        return self.coefs['intercept']+sum(self.coefs.get(t,0.) for t in fs),stop,flags

    def rate(self,value):
        base,stop,flags=self.base(value);age=value['attained_reader_handler_age_ns']
        index=bisect_right(AGE_EDGES_NS,age)-1
        return dict(hazard_per_second=positive_rate(base+self.coefs.get('age_bin:'+str(index),0.)),
            OOD_reasons=flags,observed_state_support_end_age_ns=stop,
            target=PARAMETERS['target'],physical_context_execution_gate=False,
            reader_RESULT_equals_worker_READY=False,probability_calibrated=False)

    def predict(self,value,horizon_ns):
        if type(horizon_ns) is not int or horizon_ns<0:raise ValueError('Nonnegative integer horizon required')
        base,stop,flags=self.base(value);age=value['attained_reader_handler_age_ns'];end=age+horizon_ns
        if end>stop:flags.append('FUTURE_AGE_OUTSIDE_CURRENT_READER_STATE_EXPOSURE')
        cumulative=0.
        for i,width,_ in split_age(age,end) if age<end else ():
            cumulative+=positive_rate(base+self.coefs.get('age_bin:'+str(i),0.))*(width/1e9)
            if not math.isfinite(cumulative):raise UnknownEvidence('NONFINITE_COMPLETION_HAZARD_INTEGRAL')
        return dict(candidate_completion_CDF=-math.expm1(-cumulative),OOD_reasons=flags,
            calibrated_probability=False,current_reader_peer_state_held_fixed=True,
            physical_context_execution_gate=False,reader_RESULT_equals_worker_READY=False)

    def quantile(self,value,q):
        if not 0<q<1:raise ValueError('Strict interior quantile required')
        base,support,flags=self.base(value);age=value['attained_reader_handler_age_ns'];clock=age;remaining=-math.log1p(-q)
        while True:
            i=bisect_right(AGE_EDGES_NS,clock)-1
            rate=positive_rate(base+self.coefs.get('age_bin:'+str(i),0.))
            end=AGE_EDGES_NS[i+1] if i+1<len(AGE_EDGES_NS) else None
            increment=rate*(end-clock)/1e9 if end is not None else math.inf
            if increment>=remaining:
                delay=remaining/rate*1e9
                if not math.isfinite(delay):raise UnknownEvidence('NONFINITE_READER_COMPLETION_QUANTILE')
                stop=clock+math.ceil(delay)
                if stop>support:flags.append('QUANTILE_OUTSIDE_CURRENT_READER_STATE_EXPOSURE')
                return dict(candidate_residual_ns=stop-age,OOD_reasons=flags,
                    calibrated_probability=False,current_reader_peer_state_held_fixed=True)
            remaining-=increment;clock=end


def episodes(rows):
    groups=defaultdict(list)
    for row in rows:
        validate_row(row);groups[row['capture'],row['request_id']].append(row)
    for key,values in groups.items():
        values.sort(key=lambda r:r['start_ns'])
        identities={(r['features']['alias'],r['features']['task'],r['features']['pool_id'],
            r['cohort'],r['origin_reader_START_ns']) for r in values}
        if (values[0]['age_entry_ns']!=0 or sum(r['completion_observed'] for r in values)!=1
                or len(identities)!=1
                or not values[-1]['completion_observed']
                or any(a['end_ns']!=b['start_ns'] or a['age_exit_ns']!=b['age_entry_ns'] for a,b in zip(values,values[1:]))):
            raise ValueError('Complete uninterrupted START through one terminal RESULT required')
    return groups


def landmarks(rows):
    result=[]
    for (capture,rid),values in sorted(episodes(rows).items()):
        origin=values[0]['start_ns'];end=values[-1]['end_ns']
        for age in LANDMARK_AGES_NS:
            cutoff=origin+age
            if cutoff>=end:continue
            row=next(r for r in values if r['start_ns']<=cutoff<r['end_ns'])
            value=dict(row['features'],attained_reader_handler_age_ns=age)
            result.append(dict(capture=capture,request_id=rid,cutoff_ns=cutoff,features=value,
                offline_label_residual_ns=end-cutoff))
    return result


def evaluate(model,baseline,test_rows):
    candidate=ReaderCompletionCandidate(model);reference=ReaderCompletionCandidate(baseline)
    predictions=[];horizon_values={h:[] for h in HORIZONS_NS};common={h:[] for h in HORIZONS_NS}
    median=[];coverage=[];ood=Counter()
    for row in landmarks(test_rows):
        f=row['features'];actual=row['offline_label_residual_ns'];horizons=[]
        for h in HORIZONS_NS:
            p=candidate.predict(f,h);b=reference.predict(f,h);y=int(actual<=h)
            horizon_values[h].append((y,p['candidate_completion_CDF'],b['candidate_completion_CDF']))
            if not p['OOD_reasons'] and not b['OOD_reasons']:common[h].append(horizon_values[h][-1])
            horizons.append(dict(horizon_ns=h,offline_label_completion_within_h=y,candidate=p,baseline=b))
        p50=candidate.quantile(f,.5);p95=candidate.quantile(f,.95)
        median.append(abs(p50['candidate_residual_ns']-actual));coverage.append(actual<=p95['candidate_residual_ns'])
        for flag in set(p50['OOD_reasons']+p95['OOD_reasons']):ood[flag]+=1
        predictions.append(dict(**row,horizons=horizons,p50=p50,p95=p95))
    def metric(cells):
        return dict(n=len(cells),candidate_brier=None if not cells else float(np.mean([(y-p)**2 for y,p,b in cells])),
            baseline_brier=None if not cells else float(np.mean([(y-b)**2 for y,p,b in cells])))
    return dict(landmarks=len(predictions),full_population_quantile_diagnostic=dict(
        median_absolute_error_ns=None if not median else float(np.median(median)),
        mean_absolute_error_ns=None if not median else float(np.mean(median)),
        nominal95_actual_coverage=None if not coverage else float(np.mean(coverage)),
        OOD_counts=dict(ood),includes_flagged_extrapolation=True),
        per_horizon={str(h):dict(all_landmarks_including_flagged=metric(horizon_values[h]),
            both_models_observed_state_age_support=metric(common[h])) for h in HORIZONS_NS},
        predictions=predictions,independent_GPU_validation=False,
        empirical_task_P95_predicted=False,six_term_utility_produced=False,
        frozen_current_peers_are_not_future_true_peer_trajectory=True)
