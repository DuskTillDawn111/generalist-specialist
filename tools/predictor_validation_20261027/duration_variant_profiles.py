"""Duration-weighted relative-work WLS for statistical mixed-phase prediction.

Independent fixed-candidate training engine; frozen duration sources unchanged. Contexts, baselines,
priors, bounds and UNKNOWN behavior are preserved. Only regression row weights
change to baseline_ms/observed_target_duration_ms, without renormalization.
This is conditional work WLS, not a joint runtime likelihood or causal speed.
"""
from __future__ import annotations
import collections
import copy
import statistics
import numpy as np
from scipy.optimize import lsq_linear
from .profiling_analysis import digest
from .statistical_profiles import SCHEMA, DEFINITION as ORIGINAL_DEFINITION, full_count, phase_count, StatisticalProvider

DEFINITION = copy.deepcopy(ORIGINAL_DEFINITION)
DEFINITION.update({
    "estimator": "duration_weighted_relative_work_WLS.v1",
    "objective": "mean_i[(1/sum_j X_ij)*(X_i speed-1)^2] + alpha/p * sum_j(speed_j-prior_j)^2",
    "observation_weight": "1/sum(X_i)=arithmetic_solo_baseline_ms/observed_target_duration_ms; no renormalization",
    "single_context_alpha0": "Inactive bounds imply predicted duration = arithmetic sample mean duration; finite fixed ridge changes this solution",
    "interpretation": "Conditional work WLS; observed context fractions depend on actual boundaries. No exact joint-execution MLE or causal identification claim",
    "diagnostics": "Training weighted/unweighted work reconstruction only; mean duration bias/MSE and whole-seed rollout require separate diagnostics",
})


VARIANTS=('ocr_handler','phase_ridge1')

def variant_definition(variant):
    if variant not in VARIANTS:raise ValueError('Unknown prospectively fixed duration variant')
    definition=copy.deepcopy(DEFINITION)
    definition.update(estimator='duration_variants_WLS.v1',variant=variant,
        handler_alpha=.1,semantic_alpha=1. if variant=='phase_ridge1' else .1,
        OCR_service='One ocr_native stage, exact handler fulltaskcount speeds; no OCR phase fit' if variant=='ocr_handler' else 'Original mixed semantic phases',
        alpha_scope='Separate handler and semantic alpha; alpha inherited key .1 applies handler only')
    return definition


def bounded_ridge(matrix, prior, *, alpha=.1):
    matrix, prior = np.asarray(matrix, dtype=float), np.asarray(prior, dtype=float)
    if matrix.ndim != 2 or not matrix.size or matrix.shape[1] != len(prior):
        raise ValueError("Nonempty observation matrix and matching prior required")
    n,p = matrix.shape
    total = np.sum(matrix, axis=1)
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(total <= 0):
        raise ValueError("Finite nonnegative context durations and positive target duration required")
    weights = 1. / total
    scale = np.sqrt(weights / n)
    penalty = np.sqrt(alpha/p)
    design = np.vstack([matrix * scale[:, None], penalty*np.eye(p)])
    response = np.concatenate([scale, penalty*prior])
    fit = lsq_linear(design, response, bounds=tuple(DEFINITION["progress_bounds"]), tol=1e-10)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("Bounded ridge optimization did not converge")
    errors = matrix@fit.x-1.
    return fit.x, {"observations": n, "columns": p, "matrix_rank": int(np.linalg.matrix_rank(matrix)),
        "mean_absolute_relative_work_error": float(np.mean(abs(errors))),
        "max_absolute_relative_work_error": float(np.max(abs(errors))),
        "rms_relative_work_error": float(np.sqrt(np.mean(errors**2))),
        "optimizer_cost": float(fit.cost),
        "row_weights": {"count": n, "sum": float(np.sum(weights)), "min": float(np.min(weights)), "max": float(np.max(weights)), "normalization": "w=1/sum(X); no mean-weight normalization"},
        "mean_weighted_squared_work_error": float(np.mean(weights * errors**2)), "physically_identified": False,
        "independently_validated": False, "bound_active_count": int(np.count_nonzero(fit.active_mask)),
        "parameter_scope": "Regularized statistical coefficients, not uniquely identified physical speeds"}


def _fit_groups(groups, prior_for, *, alpha=.1):
    fits = {}
    for target, rows in sorted(groups.items()):
        keys = sorted({key for row in rows for key in row["durations"]})
        matrix = np.array([[row["durations"].get(key,0.) for key in keys] for row in rows])
        prior = [prior_for(target,key) for key in keys]
        if any(value is None for value in prior):
            continue
        speeds, diagnostics = bounded_ridge(matrix, prior, alpha=alpha)
        errors = abs(matrix@speeds-1.)
        diagnostics["per_alias"] = {a:{"observations": len(ix), "mean_absolute_relative_work_error":float(np.mean(errors[ix])),
            "max_absolute_relative_work_error":float(max(errors[ix]))}
            for a in sorted({r["alias"] for r in rows}) for ix in [[i for i,r in enumerate(rows) if r["alias"]==a]]}
        fits[target] = {"target":list(target), "progress":dict(zip(keys,map(float,speeds))),
                        "priors":dict(zip(keys,map(float,prior))), "fit_diagnostics": diagnostics}
    return fits


def fit_traces(traces, *, sources=(), variant):
    definition=variant_definition(variant)
    traces=copy.deepcopy(traces)
    if variant=="ocr_handler":
        for trace in traces:
            for row in trace["requests"]:
                if row["task"]=="ocr":
                    row["segments"]=[{"phase":"ocr_native","start_ns":row["handler_start_ns"],"end_ns":row["handler_end_ns"]}]
    training = [t for t in traces if t["split"]=="training"]
    if not training:
        raise ValueError("Explicit training traces required")
    identities = {t["execution_context_sha256"] for t in training}
    if len(identities)!=1:
        raise ValueError("Mixed execution identities")
    bases, handler_bases, solo_alias = {}, {}, {}
    for trace in training:
        for row in trace["requests"]:
            if any(o is not row and o["handler_start_ns"]<row["handler_end_ns"] and row["handler_start_ns"]<o["handler_end_ns"] for o in trace["requests"]):
                continue
            target = row["family"],row["task"],row["pool_id"]
            handler_ms = (row["handler_end_ns"]-row["handler_start_ns"])/1e6
            handler_bases.setdefault(target,[]).append(handler_ms)
            solo_alias.setdefault((*target,row["alias"]),[]).append(handler_ms)
            for segment in row["segments"]:
                bases.setdefault((*target,segment["phase"]),[]).append((segment["end_ns"]-segment["start_ns"])/1e6)
    valid_handler = {k:statistics.mean(v) for k,v in handler_bases.items() if len(v)>=2}
    valid_phase = {k:statistics.mean(v) for k,v in bases.items() if len(v)>=2}
    family_contexts, phase_contexts, phase_to_family, hg, sg = {}, {}, {}, {}, {}
    for trace in training:
        rows = trace["requests"]
        segments = [(r,s) for r in rows for s in r["segments"]]
        cuts = sorted({s[k] for _,s in segments for k in ("start_ns","end_ns")})
        intervals = []
        for left,right in zip(cuts,cuts[1:]):
            active = [(r,s) for r,s in segments if s["start_ns"]<=left and right<=s["end_ns"]]
            if not active:
                continue
            if len({r["alias"] for r,_ in active})!=len(active):
                raise ValueError("Overlapping same-alias requests")
            fk,fm = full_count([r for r,_ in active]);pk,pm = phase_count(active)
            family_contexts[fk],phase_contexts[pk],phase_to_family[pk] = fm,pm,fk
            intervals.append((left,right,fk,pk))
        for row in rows:
            target = row["family"],row["task"],row["pool_id"]
            if target not in valid_handler:
                continue
            hd = {}
            for left,right,fk,pk in intervals:
                if row["handler_start_ns"]<=left and right<=row["handler_end_ns"]:
                    hd[fk] = hd.get(fk,0.)+(right-left)/1e6/valid_handler[target]
            hg.setdefault(target,[]).append({"alias":row["alias"],"durations":hd})
            for segment in row["segments"]:
                phase_target = (*target,segment["phase"])
                if phase_target not in valid_phase or (variant=="ocr_handler" and row["task"]=="ocr"):
                    continue
                sd = {}
                for left,right,fk,pk in intervals:
                    if segment["start_ns"]<=left and right<=segment["end_ns"]:
                        sd[pk] = sd.get(pk,0.)+(right-left)/1e6/valid_phase[phase_target]
                sg.setdefault(phase_target,[]).append({"alias":row["alias"],"durations":sd})
    handler = _fit_groups(hg,lambda target,key:1.)
    semantic = _fit_groups(sg,lambda target,key:handler.get(target[:3],{}).get("progress",{}).get(phase_to_family[key]),alpha=1. if variant=="phase_ridge1" else .1)
    if variant=="ocr_handler":
        for target,model in handler.items():
            if target[1]!="ocr":continue
            progress={key:model["progress"][full] for key,full in phase_to_family.items() if full in model["progress"]}
            semantic[(*target,"ocr_native")]={"target":[*target,"ocr_native"],"progress":progress,"priors":copy.deepcopy(progress),
                "fit_diagnostics":{"estimator":"Exactly tied to duration-weighted handler fulltaskcontext model; no OCR phase fit",
                    "handler_fit_diagnostics":copy.deepcopy(model["fit_diagnostics"]),"phase_parameters_fitted":0}}

    profiles = []
    for target in sorted(handler):
        ordered = []
        # All input observations must agree on phase order; no future output length feature.
        orders = {tuple(s["phase"] for s in r["segments"]) for t in training for r in t["requests"]
                  if (r["family"],r["task"],r["pool_id"])==target}
        if len(orders)!=1:
            continue
        for phase in next(iter(orders)):
            key = (*target,phase)
            if key not in valid_phase or key not in semantic:
                ordered = [];break
            ordered.append({"id":phase,"kind":"mixed","duration_ms":valid_phase[key],
                            "demand":{"sm":None,"dram":None,"l2":None}})
        if ordered:
            profiles.append({"id":digest(list(target)),"family":target[0],"task":target[1],"pool_id":target[2],
                             "segments":ordered,"execution":{"batch_size":1,"weight_reuse":False,"backend":"native_semantic_statistical_mixed"}})
    document = {"schema":SCHEMA,"definition":copy.deepcopy(definition),"definition_sha256":digest(definition),
        "execution_context_sha256":next(iter(identities)),"source_receipts":list(sources),
        "profiles":profiles,"family_contexts":family_contexts,"phase_contexts":phase_contexts,
        "phase_to_family":phase_to_family,"handler_models":list(handler.values()),"semantic_models":list(semantic.values()),
        "solo_baselines":[{"target":list(k),"count":len(v),"mean_ms":statistics.mean(v),"samples_ms":v} for k,v in sorted(bases.items())],
        "handler_solo_alias_diagnostics":[{"target":list(k[:3]),"alias":k[3],"count":len(v),"mean_ms":statistics.mean(v)} for k,v in sorted(solo_alias.items())],
        "coverage":{"training_batches":len(training),"validation_batches_excluded":len(traces)-len(training),
                    "handler_targets":len(handler),"semantic_targets":len(semantic),"profiles":len(profiles)},
        "prediction_status":{"point_prediction_available":bool(profiles),"independently_validated":False,"physically_identified":False},
        "core_enabled":False,"GPU_started":False}
    document["model_sha256"] = digest(document)
    return document


