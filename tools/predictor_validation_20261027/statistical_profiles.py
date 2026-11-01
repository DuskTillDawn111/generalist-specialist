"""Training-only two-level bounded ridge statistical mixed-phase predictor.

No coefficient is a physical interference identification or capacity bound.
Regularization permits statistical prediction despite deficient design rank.
Independent endpoint validation, uncertainty calibration and E/M are separate.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import lsq_linear

from gsdeploy.forecast_errors import UnknownEvidence
from .profiling_analysis import digest
from .semantic_profiles import semantic_traces, DEFINITION as SEMANTIC_DEFINITION

SCHEMA = "gsdeploy.statistical_semantic_progress.v1"
DEFINITION = {"schema": SCHEMA, "alpha": .1, "progress_bounds": [.05, 8.],
    "bounds_scope": "Empirical numerical regularization only; not hardware or safety bounds",
    "objective": "mean((X speed - 1)^2) + alpha * mean((speed-prior)^2)",
    "handler_prior": 1., "minimum_solo_observations": 2,
    "baseline": "Independent family/task/pool[/phase] arithmetic solo mean; aliases pooled",
    "level1": "Per-target complete family-count context handler speeds, ridge toward 1",
    "level2": "Per-target-phase full family/phase context speeds, ridge toward level1 matching fullcount speed",
    "interpolation": "Unseen phase context uses matching observed target-handler fullcount speed, explicitly interpolated",
    "domain": "Exact input solo baseline and full family-count observed for target handler; no pairwise multiway synthesis",
    "phase_contract": SEMANTIC_DEFINITION,
    "diagnostics": "Training work reconstruction only; no max-residual acceptance gate or confidence guarantee",
    "physical_identification": False, "independent_validation": False,
    "power_memory": "Separate providers; UNKNOWN here", "scenario": "mean only",
    "online_position": "Event simulator predicted semantic position; no actual future boundary/tokens",
    "core_enabled": False}


def full_count(rows):
    members = [{"family": f, "count": n} for f,n in sorted(collections.Counter(r["family"] for r in rows).items())]
    return digest(members), members


def phase_count(active):
    members = [{"family": f, "phase": p, "count": n} for (f,p),n in
               sorted(collections.Counter((r["family"],s["phase"]) for r,s in active).items())]
    return digest(members), members


def bounded_ridge(matrix, prior):
    matrix, prior = np.asarray(matrix, dtype=float), np.asarray(prior, dtype=float)
    if matrix.ndim != 2 or not matrix.size or matrix.shape[1] != len(prior):
        raise ValueError("Nonempty observation matrix and matching prior required")
    n,p = matrix.shape
    penalty = np.sqrt(DEFINITION["alpha"]/p)
    design = np.vstack([matrix/np.sqrt(n), penalty*np.eye(p)])
    response = np.concatenate([np.ones(n)/np.sqrt(n), penalty*prior])
    fit = lsq_linear(design, response, bounds=tuple(DEFINITION["progress_bounds"]), tol=1e-10)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("Bounded ridge optimization did not converge")
    errors = matrix@fit.x-1.
    return fit.x, {"observations": n, "columns": p, "matrix_rank": int(np.linalg.matrix_rank(matrix)),
        "mean_absolute_relative_work_error": float(np.mean(abs(errors))),
        "max_absolute_relative_work_error": float(np.max(abs(errors))),
        "rms_relative_work_error": float(np.sqrt(np.mean(errors**2))),
        "optimizer_cost": float(fit.cost), "physically_identified": False,
        "independently_validated": False, "bound_active_count": int(np.count_nonzero(fit.active_mask)),
        "parameter_scope": "Regularized statistical coefficients, not uniquely identified physical speeds"}


def _fit_groups(groups, prior_for):
    fits = {}
    for target, rows in sorted(groups.items()):
        keys = sorted({key for row in rows for key in row["durations"]})
        matrix = np.array([[row["durations"].get(key,0.) for key in keys] for row in rows])
        prior = [prior_for(target,key) for key in keys]
        if any(value is None for value in prior):
            continue
        speeds, diagnostics = bounded_ridge(matrix, prior)
        errors = abs(matrix@speeds-1.)
        diagnostics["per_alias"] = {a:{"observations": len(ix), "mean_absolute_relative_work_error":float(np.mean(errors[ix])),
            "max_absolute_relative_work_error":float(max(errors[ix]))}
            for a in sorted({r["alias"] for r in rows}) for ix in [[i for i,r in enumerate(rows) if r["alias"]==a]]}
        fits[target] = {"target":list(target), "progress":dict(zip(keys,map(float,speeds))),
                        "priors":dict(zip(keys,map(float,prior))), "fit_diagnostics": diagnostics}
    return fits


def fit_traces(traces, *, sources=()):
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
                if phase_target not in valid_phase:
                    continue
                sd = {}
                for left,right,fk,pk in intervals:
                    if segment["start_ns"]<=left and right<=segment["end_ns"]:
                        sd[pk] = sd.get(pk,0.)+(right-left)/1e6/valid_phase[phase_target]
                sg.setdefault(phase_target,[]).append({"alias":row["alias"],"durations":sd})
    handler = _fit_groups(hg,lambda target,key:1.)
    semantic = _fit_groups(sg,lambda target,key:handler.get(target[:3],{}).get("progress",{}).get(phase_to_family[key]))
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
    document = {"schema":SCHEMA,"definition":copy.deepcopy(DEFINITION),"definition_sha256":digest(DEFINITION),
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


def fit_statistical(raw_path):
    traces,mapping = semantic_traces(raw_path)
    paths = [Path(raw_path).resolve(),Path(__file__).resolve(),Path(__file__).with_name("semantic_profiles.py"),Path(__file__).with_name("profiling_analysis.py")]
    sources = [{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    document = fit_traces(traces,sources=sources)
    document["mapping_report"] = mapping
    document.pop("model_sha256")
    document["model_sha256"] = digest(document)
    return document


class StatisticalProvider:
    """Mean statistical predictions; observed context support != validation."""
    def __init__(self,document,execution_context_sha256):
        if document.get("schema")!=SCHEMA or document.get("execution_context_sha256")!=execution_context_sha256:
            raise ValueError("Statistical model execution identity mismatch")
        copied = copy.deepcopy(document);fingerprint = copied.pop("model_sha256",None)
        if fingerprint!=digest(copied):
            raise ValueError("Changed statistical model")
        self.document = copy.deepcopy(document)
        self.execution_context_sha256 = execution_context_sha256
        self.profiles = {(p["family"],p["task"],p["pool_id"]):p for p in document["profiles"]}
        self.lookup = self.profiles
        self.handler = {tuple(m["target"]):m for m in document["handler_models"]}
        self.semantic = {tuple(m["target"]):m for m in document["semantic_models"]}

    def plan(self,row,aliases,scenario):
        if scenario!="mean":
            raise UnknownEvidence("UNCALIBRATED_STATISTICAL_SCENARIO:"+scenario)
        target = aliases[row["alias"]]["family"],row["task"],row["pool_id"]
        if target not in self.profiles:
            raise UnknownEvidence("UNKNOWN_STATISTICAL_INPUT_SOLO_PROFILE")
        profile = self.profiles[target]
        return {"profile_id":profile["id"],"execution":copy.deepcopy(profile["execution"]),
                "segments":copy.deepcopy(profile["segments"])}

    def service(self,row,active,aliases,scenario):
        if scenario!="mean":
            return {"known":False,"reason":"UNCALIBRATED_STATISTICAL_SCENARIO"}
        target = aliases[row["alias"]]["family"],row["task"],row["pool_id"]
        try:
            members = [{"family":aliases[a]["family"]} for a in active]
            fk,_ = full_count(members)
            pooled = self.handler[target]["progress"].get(fk)
            if pooled is None:
                return {"known":False,"domain_supported":False,"reason":"UNSEEN_TARGET_FULL_FAMILY_COUNT"}
            phases = [({"family":aliases[a]["family"]},{"phase":r["segment_id"]}) for a,r in active.items()]
            pk,_ = phase_count(phases)
            model = self.semantic[(*target,row["segment_id"])]
            speed = model["progress"].get(pk,pooled)
            interpolated = pk not in model["progress"]
            profile = self.profiles[target]
            baseline = next(s["duration_ms"] for s in profile["segments"] if s["id"]==row["segment_id"])
            if row.get("profile_id")!=profile["id"]:
                raise ValueError("Active statistical profile identity mismatch")
            return {"known":True,"context_supported":True,"domain_supported":True,"point_prediction_available":True,
                "service_ms":baseline/speed,"progress":speed,"interpolated":interpolated,
                "evidence":self.document["model_sha256"],"scope":"Regularized mixed-wall statistical prediction",
                "physically_identified":False,"independently_validated":False,
                "regularized":True,"uncertainty_calibrated":False}
        except KeyError:
            return {"known":False,"domain_supported":False,"reason":"UNKNOWN_STATISTICAL_TARGET_OR_PHASE"}

    def power(self,*args):
        return {"known":False,"watts":None,"reason":"SEPARATE_WHOLE_BOARD_PROVIDER_REQUIRED"}

    def memory(self,*args):
        return {"known":False,"peak_memory_mib":None,"reason":"SEPARATE_MEMORY_PROVIDER_REQUIRED"}


StatisticalProfileProvider = StatisticalProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    document = fit_statistical(args.raw)
    with args.output.open("x") as stream:
        json.dump(document,stream,indent=2,allow_nan=False);stream.write("\n")


if __name__=="__main__":
    main()
