"""Trainable, explicitly unvalidated SAVE-envelope hazard; CPU training only.

Fixed piecewise-exponential Poisson likelihood, not physical SM service.
Frontier changes censor and restore ages survive delayed entry. The conditional
exchangeability / noninformative censoring hypotheses remain unverified.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from .context_survival_v12_5 import ContextEnvelopeSurvival
from .multi_native_composite_v5 import checked
from .native_collect import receipt, save
from .profiling import digest

SCHEMA = 'gsdeploy.age_conditioned_SAVE_hazard_candidate.v13'
# Declared before reading fitted outcomes. The final bin is right-unbounded,
# but its prediction is supported only through observed training exposure.
EDGES_NS = (0, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
            200_000, 500_000, 1_000_000, 2_000_000, 5_000_000,
            10_000_000, 20_000_000, 50_000_000)
L2 = 1.0
MODES = ('constant', 'holder_only', 'age_additive')


def split_exposure(row):
    """Positive exposure on (entry, exit]; an exit-edge event goes left."""
    entry, exit = row['age_entry_ns'], row['age_exit_ns']
    if (type(entry) is not int or type(exit) is not int or not 0 <= entry < exit
            or type(row['save_observed']) is not bool):
        raise ValueError('Integer positive delayed-entry exposure required')
    first = bisect_right(EDGES_NS, entry)-1
    final = bisect_left(EDGES_NS, exit)-1
    for index in range(first, final+1):
        left = max(entry, EDGES_NS[index])
        right = min(exit, EDGES_NS[index+1]) if index+1 < len(EDGES_NS) else exit
        if right > left:
            yield index, right-left, int(row['save_observed'] and right == exit)


def frontier_features(holder, frontier):
    if not isinstance(holder, str) or not holder or not frontier:
        raise ValueError('Visible holder and nonempty pending frontier required')
    features = {'holder:'+holder}
    for row in frontier:
        if (not isinstance(row, (list, tuple)) or len(row) != 4
                or any(not isinstance(x, str) or not x for x in row[:3])
                or not isinstance(row[3], (list, tuple)) or not row[3]
                or any(not isinstance(x, str) or not x for x in row[3])):
            raise ValueError('Visible owner/task/pool/phase frontier required')
        owner, task, pool, phases = row
        features.update(('pending_owner:'+owner, 'task:'+task, 'pool:'+pool))
        features.update('phase:'+phase for phase in phases)
    return tuple(sorted(features))


def aggregate(episodes):
    """Loss-sufficient rows; retain every input interval and every ns/event."""
    bins = defaultdict(lambda: [0, 0, 0])
    counts = Counter(); envelopes = defaultdict(set)
    exact = set(); vocab = set()
    for row in episodes:
        capture = row['capture']; envelopes[capture].add(row['envelope_id'])
        fs = frontier_features(row['held_owner'], row['frontier'])
        exact.add(digest([row['held_owner'], row['frontier']]))
        vocab.update(fs)
        pieces = list(split_exposure(row))
        assert sum(n for _, n, _ in pieces) == row['age_exit_ns']-row['age_entry_ns']
        assert sum(e for _, _, e in pieces) == int(row['save_observed'])
        for index, exposure, event in pieces:
            target = bins[capture, index, fs]
            target[0] += exposure; target[1] += event; target[2] += 1
        counts['episodes'] += 1
        counts['events'] += int(row['save_observed'])
        counts['exposure_ns'] += row['age_exit_ns']-row['age_entry_ns']
        counts['censored_episodes'] += int(not row['save_observed'])
    if not counts['events']:
        raise ValueError('Positive observed SAVE evidence required for hazard fit')
    rows = [dict(capture=c, age_bin=i, features=list(fs), exposure_ns=v[0],
                 events=v[1], episode_bin_intersections=v[2])
            for (c, i, fs), v in sorted(bins.items())]
    # Capture-wide support is a union, not a min/max fill over absent exposure.
    support = []
    for entry, exit in sorted((r['age_entry_ns'], r['age_exit_ns']) for r in episodes):
        if support and entry <= support[-1][1]:
            support[-1][1] = max(support[-1][1], exit)
        else:
            support.append([entry, exit])
    return dict(rows=rows, counts=dict(counts),
                capture_envelopes={c:len(s) for c,s in sorted(envelopes.items())},
                vocabulary=sorted(vocab), exact_frontier_keys=sorted(exact),
                age_support_ns=support,
                age_bin_exposure_ns={str(i):sum(r['exposure_ns'] for r in rows if r['age_bin']==i)
                                     for i in range(len(EDGES_NS))})


def selected_features(row, mode):
    if mode == 'constant':
        return ()
    fs = row['features']
    if mode == 'holder_only':
        return tuple(f for f in fs if f.startswith('holder:'))
    if mode == 'age_additive':
        return (*fs, 'age_bin:'+str(row['age_bin']))
    raise ValueError('Predeclared model mode required')


def design(rows, columns, mode):
    lookup = {f:i for i,f in enumerate(columns)}; ii=[]; jj=[]; unseen=Counter()
    for i,row in enumerate(rows):
        ii.append(i); jj.append(0)
        for feature in selected_features(row, mode):
            if feature in lookup:
                ii.append(i); jj.append(lookup[feature])
            else:
                unseen[feature] += 1
    return csr_matrix((np.ones(len(ii)), (ii,jj)), shape=(len(rows),len(columns))), dict(unseen)


def capture_weights(data):
    captures = data['capture_envelopes']
    return {c:1/(len(captures)*n) for c,n in captures.items()}


def objective(beta, x, exposure, events, weights):
    eta = x @ beta
    with np.errstate(over='ignore'):
        expected = exposure*np.exp(eta)
    if not np.all(np.isfinite(expected)):
        return float('inf'), np.zeros_like(beta)
    # Full parameter-dependent exposure likelihood, not event-only loss.
    value = float(np.sum(weights*(expected-events*eta)) + .5*L2*np.dot(beta[1:],beta[1:]))
    gradient = np.asarray(x.T @ (weights*(expected-events)))
    gradient[1:] += L2*beta[1:]
    return value, gradient


def fit(data, mode):
    began = time.monotonic();rows = data['rows']
    columns = ['intercept', *sorted({f for r in rows for f in selected_features(r, mode)})]
    x,_ = design(rows, columns, mode)
    exposure = np.array([r['exposure_ns']/1e6 for r in rows])
    events = np.array([r['events'] for r in rows])
    cw = capture_weights(data);weights = np.array([cw[r['capture']] for r in rows])
    initial = np.zeros(len(columns));initial[0] = math.log(float(np.dot(weights,events)/np.dot(weights,exposure)))
    result = minimize(objective, initial, args=(x,exposure,events,weights), jac=True,
                      method='L-BFGS-B', options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
    return dict(schema=SCHEMA, mode=mode, columns=columns, coefficients=result.x.tolist(),
                l2_lambda=L2, intercept_penalized=False, exposure_unit='ms',
                capture_weights=cw, capture_envelopes=data['capture_envelopes'],
                training_counts=data['counts'], training_vocabulary=data['vocabulary'],
                exact_frontier_keys=data['exact_frontier_keys'],age_support_ns=data['age_support_ns'],
                age_bin_exposure_ns=data['age_bin_exposure_ns'], fixed_age_edges_ns=list(EDGES_NS),
                optimizer=dict(method='SciPy L-BFGS-B',success=bool(result.success),message=str(result.message),
                               iterations=int(result.nit),objective=float(result.fun),
                               gradient_max_abs=float(np.max(np.abs(result.jac)))),
                elapsed_seconds=time.monotonic()-began,training_only=True,
                physical_SM_service=False,probability_calibration_established=False)


def evaluate(model, data):
    rows = data['rows'];x,unseen = design(rows,model['columns'],model['mode'])
    eta = x @ np.array(model['coefficients'])
    exposure = np.array([r['exposure_ns']/1e6 for r in rows]);events=np.array([r['events'] for r in rows])
    cw=capture_weights(data);weights=np.array([cw[r['capture']] for r in rows])
    loss=weights*(exposure*np.exp(eta)-events*eta)
    return dict(full_exposure_poisson_loss_without_parameter_independent_constants=float(np.sum(loss)),
                expected_events_capture_envelope_weighted=float(np.sum(weights*exposure*np.exp(eta))),
                observed_events_capture_envelope_weighted=float(np.sum(weights*events)),
                counts=data['counts'],aggregate_rows=len(rows),unseen_features=unseen,
                unseen_feature_aggregate_rows=sum(1 for row in rows if any(f not in model['columns'] for f in selected_features(row,model['mode']))),
                unknown_rows_removed=0, probability_calibration=False)


class SaveHazardCandidate:
    def __init__(self, model):
        if model.get('schema') != SCHEMA or model.get('mode') != 'age_additive':
            raise ValueError('Explicit age-additive SAVE candidate required')
        if model.get('fixed_age_edges_ns') != list(EDGES_NS):
            raise ValueError('Predeclared age bins required')
        self.model=model;self.coefs=dict(zip(model['columns'],model['coefficients']))

    def query(self, held_owner, pending_frontier, context_age_ns, horizon_ns=0):
        if any(type(t) is not int or t < 0 for t in (context_age_ns,horizon_ns)):
            raise ValueError('Nonnegative integer actual restore age/horizon required')
        fs=frontier_features(held_owner,pending_frontier);m=self.model
        unseen=sorted(set(fs)-set(m['training_vocabulary']))
        exact=digest([held_owner,pending_frontier]) in m['exact_frontier_keys']
        index=bisect_right(EDGES_NS,context_age_ns)-1
        def hazard(i):
            eta=self.coefs['intercept']+sum(self.coefs.get(f,0.) for f in fs)+self.coefs.get('age_bin:'+str(i),0.)
            return math.exp(eta)
        segments=m['age_support_ns'];support_end=context_age_ns
        for entry,exit in segments:
            if entry <= context_age_ns < exit:
                support_end=exit;break
        end=context_age_ns+horizon_ns;restricted=min(end,support_end)
        total=0.;clock=context_age_ns
        while clock < restricted:
            i=bisect_right(EDGES_NS,clock)-1
            stop=min(restricted,EDGES_NS[i+1]) if i+1 < len(EDGES_NS) else restricted
            total += hazard(i)*(stop-clock)/1e6;clock=stop
        flags=[]
        if unseen:flags.append('UNSEEN_VISIBLE_CATEGORIES')
        if support_end == context_age_ns:flags.append('AGE_OUTSIDE_OBSERVED_EXPOSURE')
        if end > support_end:flags.append('HORIZON_OUTSIDE_OBSERVED_EXPOSURE')
        value=-math.expm1(-total)
        return dict(status='UNVALIDATED_TRAINED_HAZARD_CANDIDATE',hazard_per_ms=hazard(index),
                    conditional_CDF=value if not flags else None,
                    restricted_candidate_CDF=value,restricted_horizon_ns=restricted-context_age_ns,
                    requested_horizon_ns=horizon_ns,context_age_ns=context_age_ns,
                    OOD=bool(flags),OOD_reasons=flags,unseen_features=unseen,
                    exact_frontier_seen=exact,novel_combination_allowed=True,
                    candidate_uses_zero_effect_for_unseen_categories=bool(unseen),
                    complete_finite_distribution=False,tail_extrapolation_supported=False,
                    probability_calibration=False,physical_SM_service=False,
                    conditional_on_frontier_remaining_fixed=True,
                    capture_union_support_does_not_establish_category_conditional_support=True)


def run(model_path, output):
    began=time.monotonic();out=Path(output).resolve();out.mkdir(exist_ok=False,parents=True)
    source=receipt(model_path);original=checked(source)
    declaration=dict(schema=SCHEMA,source_model=source,fixed_age_edges_ns=list(EDGES_NS),
                     l2_lambda=L2,intercept_penalized=False,modes=list(MODES),
                     objective='sum_c 1/(C*N_envelopes_c) * sum_episode_bin(exposure_ms*exp(eta)-event*eta) + 0.5*lambda*sum_nonintercept(beta^2)',
                     exposure_rule='entry < t <= exit; exit-edge SAVE belongs left bin',
                     capture_split='two whole-capture folds; all rows preserved',
                     category_features='holder, pending owner/task/pool/phase set indicators, age-bin indicators; no interactions',
                     training_only=True,GPU_started=False)
    save(out/'declaration.json',declaration)
    # This path verifies the original extractor against ALL original sources.
    checked_provider=ContextEnvelopeSurvival(original,allow_unvalidated_survival_hypothesis=True,
                                           allow_v12_4_source_model=True)
    del checked_provider
    verification_seconds=time.monotonic()-began
    episodes=original['episodes'];captures=sorted({r['capture'] for r in episodes})
    if len(captures)!=2:
        raise ValueError('Two-capture diagnostic may not choose a subset')
    full=aggregate(episodes);save(out/'sufficient-statistics.json',full)
    fits={};training={}
    for mode in MODES:
        fits[mode]=fit(full,mode);training[mode]=evaluate(fits[mode],full)
        save(out/(mode+'-model.json'),fits[mode])
    folds=[]
    for held in captures:
        train=aggregate([r for r in episodes if r['capture']!=held])
        test=aggregate([r for r in episodes if r['capture']==held])
        fold=dict(held_out_training_capture=held,training_captures=list(train['capture_envelopes']),models={})
        for mode in MODES:
            fitted=fit(train,mode)
            name='fold-'+str(len(folds))+'-'+mode+'-model.json';save(out/name,fitted)
            fold['models'][mode]=dict(model=receipt(out/name),optimizer=fitted['optimizer'],
                                      held_training_capture=evaluate(fitted,test),fit_seconds=fitted['elapsed_seconds'])
        folds.append(fold)
    candidate=SaveHazardCandidate(fits['age_additive'])
    queries=[]
    seen=set()
    for row in episodes:
        key=digest([row['held_owner'],row['frontier']])
        if key in seen:continue
        seen.add(key)
        for age in EDGES_NS:
            queries.append(dict(held_owner=row['held_owner'],frontier=row['frontier'],
                                prediction=candidate.query(row['held_owner'],row['frontier'],age,100_000)))
    save(out/'candidate-queries.json',queries)
    # Frozen episode model verifies its complete original lineage. Recheck every
    # declared source again at completion; no diagnostic/validation labels used.
    sources=[source,receipt(__file__),receipt(Path(__file__).with_name('context_survival_v12_5.py')),
             *original['sources']]
    for ref in sources:
        if receipt(ref['path'])!=ref:raise ValueError('Source changed during hazard training')
    report=dict(schema=SCHEMA,status='TRAINABLE_CANDIDATE_FITTED_TRAINING_DIAGNOSTIC_COMPLETE',
                sources=sources,source_rederivation='v12_5 explicit full-source v12_4 episode reuse',
                source_verification_seconds=verification_seconds,counts=full['counts'],
                capture_envelopes=full['capture_envelopes'],sufficient_statistics=receipt(out/'sufficient-statistics.json'),
                models={k:receipt(out/(k+'-model.json')) for k in MODES},training=training,folds=folds,
                candidate_queries=receipt(out/'candidate-queries.json'),query_count=len(queries),
                all_episodes_preserved=True,validation_labels_used=False,independent_validation=False,
                capture_equally_weighted=True,same_envelope_episodes_independent=False,
                probability_calibration=False,physical_SM_service_identified=False,
                complete_dispatch_oracle=False,complete_finite_distribution=False,
                training_only=True,GPU_started=False,elapsed_seconds=time.monotonic()-began)
    save(out/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=run(args.model,args.output)
    print(json.dumps({k:report[k] for k in ('status','counts','elapsed_seconds')}))


if __name__=='__main__':main()
