#!/usr/bin/env python3
"""CPU-only finite text backup screening; no energy, utility or certification."""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsdeploy.calibration_v2 import replay_input_ids
from gsdeploy.quality import predict_quality
from gsdeploy.routing import Router
from gsdeploy.trace import expand_routes, make_trace, trace_digest

SOURCE = ROOT / "artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json"
OUTPUT = ROOT / "artifacts/static-spillover-study-20261022/cpu-discovery/text-backup-readonly-v1"
STATUS = "EXPLORATORY_STANDALONE_REPLAY_NOT_VALIDATED"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    original = json.loads(SOURCE.read_text())
    data = original["catalog"]
    baselines = data["calibrated_replay_v2"]["input_baselines"]
    protocol = dict(status=STATUS, source=dict(path=str(SOURCE), sha256=sha(SOURCE)),
        gpu_started=False, production_math_modified=False, production_U=None, E=None,
        validated_utility=False, optimality_certified=False,
        workloads=[dict(text=1.8, image=.05, ocr=.05, asr=.05),
                   dict(text=1.8, image=.01, ocr=.01, asr=.01),
                   dict(text=1.9, image=.01, ocr=.01, asr=.01)],
        seeds=[51, 52, 53], horizon_seconds=1800,
        finite_routes=[dict(text_replicas=n, generalist_2_replicas=1, text_to_G2=s)
                       for n, s in [(2, 0), (2, .125), (2, .25), (1, .25), (1, .5), (3, 0)]],
        background_routing=dict(image="generalist_2", ocr="ocr", asr="generalist_2"),
        service_model="Historical median per input and n-state; no old global service multiplier; no colocation/load model.",
        memory_model="Original safe_factor times standalone residency sum; old learned multiplier shown separately, not treated as observed memory.",
        quality_model="Unchanged public historical per-sample replay; actual mixed outputs unverified.",
        limitations=["No energy or total utility comparison", "No GPU interference model", "No open-loop validation",
                     "Limited finite alternatives, not full-pool optimization", "Historical request equivalence caveat retained"])
    save(OUTPUT / "protocol.json", protocol)
    table = []
    for m, tasks in data["task_replica_profiles"].items():
        for t, states in tasks.items():
            for ns, entry in states.items():
                n = int(ns)
                table.append(dict(model=m, task=t, replicas=n,
                    measured_mean_service_ms=entry["mean_service_ms"],
                    standalone_mean_rho08_rps=.8 * data["capacity_factors"][m][ns] * 1000 / entry["mean_service_ms"],
                    residency_mib=data["memory_table_mib"][m][ns], source=entry["source"]))
    memory = []
    for nt in (2, 3, 4):
        for ng in (1, 2):
            for image_specialist in (False, True):
                x = dict(text=nt, generalist_2=ng, ocr=1)
                if image_specialist:
                    x["image"] = 1
                raw = data["safe_factor"] * math.fsum(data["memory_table_mib"][m][str(n)] for m, n in x.items())
                transferred = raw * data["calibrated_replay_v2"]["memory_multiplier"]
                memory.append(dict(deployment=x, safe_residency_proxy_mib=raw,
                    safe_residency_proxy_fits=raw <= original["gpu_memory_mib"],
                    old_extra_multiplier_proxy_mib=transferred,
                    old_extra_multiplier_proxy_fits=transferred <= original["gpu_memory_mib"]))
    shapes = []
    for m, ns in [("text", [1, 2, 3, 4]), ("generalist_2", [1, 2])]:
        for n in ns:
            entry = data["task_replica_profiles"][m]["text"][str(n)]
            sample_table = baselines[m]["text"][str(n)]["samples"]
            values = [z["median_service_ms"] for z in sample_table.values()]
            records = [json.loads(line) for f in entry["raw_service_samples"] for line in Path(f).read_text().splitlines() if line.strip()]
            fields = {}
            for key in ("input_tokens", "generated_tokens"):
                valid = [r for r in records if type(r.get(key)) in (int, float)]
                fields[key] = dict(count=len(valid), min_p50_p95_max=np.percentile([r[key] for r in valid], [0, 50, 95, 100]).tolist(),
                    pearson_with_service_ms=float(np.corrcoef([r[key] for r in valid], [r["service_ms"] for r in valid])[0, 1]))
            shapes.append(dict(model=m, replicas=n, input_pool_size=len(values), pool_mean_service_ms=float(np.mean(values)),
                pool_service_min_p50_p95_max_ms=np.percentile(values, [0, 50, 95, 100]).tolist(), raw_records=len(records),
                fields=fields, generation_controls=records[0]["effective_generation"],
                raw_sources=[dict(path=f, sha256=sha(f)) for f in entry["raw_service_samples"]]))
    results = []
    for rates in protocol["workloads"]:
        for seed in protocol["seeds"]:
            trace = make_trace(rates, protocol["horizon_seconds"], seed, original["arrival_process"])
            ids = replay_input_ids(data, trace, seed)
            for route in protocol["finite_routes"]:
                n, share = route["text_replicas"], route["text_to_G2"]
                x = dict(text=n, generalist_2=1, ocr=1)
                textrow = dict(text=rates["text"] * (1 - share))
                if share:
                    textrow["generalist_2"] = rates["text"] * share
                routing = dict(text=textrow, image={"generalist_2": rates["image"]},
                               ocr={"ocr": rates["ocr"]}, asr={"generalist_2": rates["asr"]})
                aliases, routes = expand_routes(x, routing)
                router, clocks = Router(routes), {a: 0. for a in aliases}
                lat = {t: [] for t in rates}
                inwindow = late = 0
                for (arrival, t), sample in zip(trace, ids):
                    alias = router.choose(t)
                    model = aliases[alias]
                    svc = baselines[model][t][str(x[model])]["samples"][sample]["median_service_ms"]
                    svc *= x[model] / data["capacity_factors"][model][str(x[model])]
                    finish = max(arrival, clocks[alias]) + svc / 1000
                    clocks[alias] = finish
                    lat[t].append((finish - arrival) * 1000)
                    inwindow += finish <= protocol["horizon_seconds"]
                    late += finish > protocol["horizon_seconds"] + original["drain_seconds"]
                rho = {m: math.fsum(row.get(m, 0) * np.mean([z["median_service_ms"] for z in
                       baselines[m][t][str(x[m])]["samples"].values()]) / 1000
                       for t, row in routing.items() if m in row) / data["capacity_factors"][m][str(x[m])] for m in x}
                q, qe = predict_quality(data, x, routing, rates, trace, seed)
                assert q is not None
                mem = data["safe_factor"] * math.fsum(data["memory_table_mib"][m][str(nn)] for m, nn in x.items())
                results.append(dict(workload=copy.deepcopy(rates), seed=seed, deployment=x, routing=routing,
                    text_to_G2_share=share, status=STATUS, production_U=None, E=None,
                    historical_quality_loss=q, historical_text_EM=1 - qe["per_task"]["text"]["loss"],
                    raw_service_rho=rho, raw_service_max_rho=max(rho.values()), safe_residency_proxy_mib=mem,
                    proxy_constraints_pass=max(rho.values()) <= .8 and mem <= original["gpu_memory_mib"] and late == 0,
                    per_task={t:dict(mean_ms=float(np.mean(v)), p95_ms=float(np.percentile(v, 95)), count=len(v)) for t, v in lat.items()},
                    throughput_loss=1 - inwindow / len(trace), beyond_drain_requests=late, trace_sha256=trace_digest(trace)))
    save(OUTPUT / "report.json", dict(protocol=protocol, measured_profile_table=table,
        limited_memory_matrix=memory, text_service_shapes=shapes, finite_replays=results))
    save(OUTPUT / "manifest.json", dict(script=dict(path=str(Path(__file__)), sha256=sha(__file__)),
        files={p.name:sha(p) for p in OUTPUT.iterdir() if p.is_file()}, gpu_started=False))
    print(json.dumps(dict(output=str(OUTPUT), finite_replays=len(results), measured_profile_rows=len(table))))


if __name__ == "__main__":
    main()
