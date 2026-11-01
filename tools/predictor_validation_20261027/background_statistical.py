"""New all-task statistical model: full task mixtures have separate coefficients.

The immutable v6 statistical model is reused as a numerical engine with a
bijective internal (family,task) identity. Thus every handler context is a full
family/task multiset and every semantic context a full family/task/phase
multiset. No background task can silently share a competitor coefficient.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .background_profiles import background_semantic_traces
from .profiling_analysis import digest
from .statistical_profiles import (
    DEFINITION as BASE_DEFINITION, StatisticalProvider, fit_traces as fit_base_traces,
)

SCHEMA = "gsdeploy.background_task_statistical_progress.v1"
DEFINITION = {"schema":SCHEMA,"base_numerics":BASE_DEFINITION,
    "handler_context":"Full active (original family, actual task, count) multiset",
    "semantic_context":"Full active (original family, actual task, semantic phase, count) multiset",
    "target":"Original family/task/input/phase with independent solo baseline",
    "task_identity_encoding":"Canonical JSON [family,task] bijection; no task pooling",
    "fallback":"Only unseen phase combinations inside target-observed full family/task multiset; explicitly interpolated",
    "unseen_task_mixture":"UNKNOWN even if family counts were observed",
    "input_provenance":"New all-task collection context only; no v5/v6 model transfer",
    "scope":"Regularized statistical predictions; no independent validation or physical identification"}


def family_task(family,task):
    return json.dumps([family,task],ensure_ascii=False,separators=(",",":"))


def fit_traces(traces, *, sources=()):
    encoded = copy.deepcopy(traces)
    for trace in encoded:
        for row in trace["requests"]:
            row["family"] = family_task(row["family"],row["task"])
    base = fit_base_traces(encoded,sources=sources)
    document = {"schema":SCHEMA,"definition":copy.deepcopy(DEFINITION),"definition_sha256":digest(DEFINITION),
        "execution_context_sha256":base["execution_context_sha256"],"source_receipts":list(sources),
        "base_model":base,"coverage":base["coverage"],"core_enabled":False,"GPU_started":False,
        "prediction_status":{"point_prediction_available":bool(base["profiles"]),
                             "independently_validated":False,"physically_identified":False},
        "training_task_mixtures":[[{"family":json.loads(m["family"])[0],"task":json.loads(m["family"])[1],"count":m["count"]}
                                   for m in members] for members in base["family_contexts"].values()]}
    document["model_sha256"] = digest(document)
    return document


def fit_background(raw_path):
    traces,mapping = background_semantic_traces(raw_path)
    paths = [Path(raw_path).resolve(),Path(__file__).resolve(),Path(__file__).with_name("statistical_profiles.py"),
             Path(__file__).with_name("background_profiles.py"),Path(__file__).with_name("semantic_profiles.py")]
    sources = [{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    document = fit_traces(traces,sources=sources)
    document["mapping_report"] = mapping
    document.pop("model_sha256")
    document["model_sha256"] = digest(document)
    return document


class BackgroundStatisticalProvider:
    def __init__(self,document,execution_context_sha256):
        if document.get("schema")!=SCHEMA or document.get("execution_context_sha256")!=execution_context_sha256:
            raise ValueError("Task-aware model execution identity mismatch")
        unsigned = copy.deepcopy(document);fingerprint = unsigned.pop("model_sha256",None)
        if fingerprint!=digest(unsigned):
            raise ValueError("Changed task-aware model")
        self.document = copy.deepcopy(document)
        self.execution_context_sha256 = execution_context_sha256
        self.base = StatisticalProvider(document["base_model"],execution_context_sha256)
        self.lookup = {}
        for (encoded_family,task,pool),profile in self.base.lookup.items():
            family,encoded_task = json.loads(encoded_family)
            if encoded_task!=task:
                raise ValueError("Task-aware model internal identity mismatch")
            public_profile = copy.deepcopy(profile);public_profile["family"] = family
            self.lookup[(family,task,pool)] = public_profile
        self.profiles = self.lookup

    def _aliases(self,aliases,active):
        encoded = copy.deepcopy(aliases)
        for alias,row in active.items():
            encoded[alias]["family"] = family_task(aliases[alias]["family"],row["task"])
        return encoded

    def plan(self,row,aliases,scenario):
        return self.base.plan(row,self._aliases(aliases,{row["alias"]:row}),scenario)

    def service(self,row,active,aliases,scenario):
        result = self.base.service(row,active,self._aliases(aliases,active),scenario)
        if result.get("reason")=="UNSEEN_TARGET_FULL_FAMILY_COUNT":
            result["reason"] = "UNSEEN_TARGET_FULL_FAMILY_TASK_MIXTURE"
        result.update(task_aware=True,competitor_task_pooling=False,
            scope="Full family/task/phase statistical coefficients; mixed wall progress")
        if result.get("known"):
            result["evidence"] = self.document["model_sha256"]
        return result

    def power(self,*args):
        return {"known":False,"watts":None,"reason":"SEPARATE_TASK_AWARE_BOARD_PROVIDER_REQUIRED"}

    def memory(self,*args):
        return {"known":False,"peak_memory_mib":None,"reason":"SEPARATE_TASK_AWARE_MEMORY_PROVIDER_REQUIRED"}


TaskAwareStatisticalProvider = BackgroundStatisticalProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    document = fit_background(args.raw)
    with args.output.open("x") as stream:
        json.dump(document,stream,indent=2,allow_nan=False);stream.write("\n")


if __name__=="__main__":main()
