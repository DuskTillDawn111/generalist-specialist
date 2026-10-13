"""Planning-only probe selection and a single deployment-local log residual."""
import math
import numpy as np

from performance_model.generalization.features import planning_vector, FoldPreprocessor


SELECTION_POLICY = dict(
    max_budget=3, max_trials=3,
    single_resident_max_model_demand=.20,
    multi_resident_max_model_demand=.40,
    safety_rationale='Conservative planning surrogate only: prior isolated Qwen backlog begins near u=.3. Prefer <=.2 with a single resident; cap every model at .4 with multiple residents. Residency does not guarantee safety. Reveal selected probes and reject non-steady configurations; never replace them using test labels.',
    first_anchor='Up to three distinct lowest planning-risk anchors; deterministic sort by numeric features, not observed labels.',
    risk='max model standalone demand + .05 * max(0, request-rate-weighted global predicted log amplification)',
    distance='Euclidean RMS of request-rate-weighted planning feature vectors standardized ONLY on global training configurations.',
    later_points='Greedy farthest-first distance from previously selected configurations; numeric-feature tie break.',
    fixed_test='Reserve full nested calibration chain up to supported max K; all other configurations form a fixed test set for every K in that trial.',
    unsupported='Never split repetitions to obtain test data; K unavailable if insufficient safe candidates or independent test configurations.',
    reveal_gate='A chosen configuration contributes residuals only if ALL its runs are steady under the current threshold setting. Rejected probes still consume budget and are never moved to test or replaced.',
    repetitions='Use all available repetitions of each selected configuration; record run count and measurement-seconds cost. K counts configurations, not single runs.',
)


def planning_descriptor(row, ex, resource_keys, predictor):
    x, w, r = row['deployment'], row['workload'], row['routing']
    f = ex.extract(x, w, r)
    paths, weights = [], []
    for task, route in f['routing'].items():
        for model, rate in route.items():
            if rate > 0:
                v, _, _ = planning_vector(dict(deployment=x, workload=w, routing=r, task=task, target_model=model), ex, 'C', resource_keys)
                paths.append(v)
                weights.append(rate)
    weights = np.array(weights) / sum(weights)
    vector = {k: float(sum(w * p[k] for w, p in zip(weights, paths))) if all(p[k] is not None for p in paths) else None
              for k in paths[0]}
    predicted_y = float(np.dot(weights, predictor(paths)))
    u = max(p['standalone_service_demand'] for p in f['per_model'].values())
    cap = SELECTION_POLICY['single_resident_max_model_demand'] if sum(x.values()) == 1 else SELECTION_POLICY['multi_resident_max_model_demand']
    return dict(vector=vector, max_model_demand=u, planning_safe_candidate=u <= cap + 1e-12,
                safety_cap=cap, predicted_log_amplification=predicted_y,
                risk=u + .05 * max(0., predicted_y))


def selection_plan(descriptors, training_descriptors):
    ids = sorted(descriptors)
    safe = [c for c in ids if descriptors[c]['planning_safe_candidate']]
    train_vectors = [v['vector'] for v in training_descriptors.values()]
    scaler = FoldPreprocessor().fit(train_vectors, list(train_vectors[0]))
    vectors = {c: scaler.transform([descriptors[c]['vector']])[0] for c in ids}
    def tie(c):
        return tuple(-1e100 if v is None else v for v in descriptors[c]['vector'].values())
    anchors = sorted(safe, key=lambda c: (descriptors[c]['risk'], tie(c)))[:SELECTION_POLICY['max_trials']]
    max_k = min(SELECTION_POLICY['max_budget'], len(safe), max(0, len(ids) - 1))
    trials = []
    if max_k:
        for i, anchor in enumerate(anchors):
            chain, distances = [anchor], [None]
            while len(chain) < max_k:
                available = [c for c in safe if c not in chain]
                distances_to_selected = {c: min(float(np.sqrt(np.mean((vectors[c] - vectors[s]) ** 2))) for s in chain) for c in available}
                nxt = max(available, key=lambda c: (distances_to_selected[c], tie(c)))
                chain.append(nxt)
                distances.append(distances_to_selected[nxt])
            test = [c for c in ids if c not in chain]
            if not test or set(test) & set(chain): raise ValueError('Invalid calibration/test partition')
            trials.append(dict(trial_id=f'anchor_{i}', calibration_chain=chain, acquisition_distances=distances,
                               test_configurations=test, max_supported_K=max_k))
    return dict(candidates=descriptors, safe_candidates=safe, max_supported_K=max_k,
        unavailable_budgets={str(k): 'insufficient planning-safe configurations while reserving at least one independent test configuration'
                             for k in (1, 2, 3) if k > max_k},
        trials=trials, distance_training_configuration_ids=sorted(training_descriptors),
        distance_preprocessing=scaler.audit([descriptors[c]['vector'] for c in ids]))


def bias_from_probes(rows, calibration_ids, predicted_y, labels):
    selected = [r for r in rows if r['configuration_id'] in calibration_ids]
    accepted, rejected = [], []
    for cfg in calibration_ids:
        rs = [r for r in selected if r['configuration_id'] == cfg]
        if rs and all(labels[r['run_id']]['stability_state_v2'] == 'steady' for r in rs): accepted.append(cfg)
        else: rejected.append(cfg)
    used = [r for r in selected if r['configuration_id'] in accepted and '_y' in r]
    residuals = [math.log(r['_y'] / r['_s0']) - predicted_y[r['_id']] for r in used]
    bias = float(np.median(residuals)) if residuals else 0.
    return dict(bias_log_ratio=bias, multiplier=math.exp(bias), accepted_configurations=accepted,
        rejected_configurations=rejected, calibration_row_ids=[r['_id'] for r in used],
        residuals=residuals, effective_configuration_count=len(accepted),
        status='calibrated' if residuals else 'no_accepted_probe_global_fallback',
        measurement_run_ids=sorted({r['run_id'] for r in selected}),
        calibration_measurement_seconds=sum({r['run_id']: r['measurement_seconds'] for r in selected}.values()))
