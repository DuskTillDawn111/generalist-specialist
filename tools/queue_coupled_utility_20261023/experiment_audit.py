"""Independent read-only audit of the isolated controlled calibration.

No quality answers, references or per-input evaluation scores are extracted.
This verifies evidence identity and arithmetic, not physical counterfactuals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

PRIMARY = {"text": 2, "generalist_2": 1, "ocr": 1}
CONTEXTS = {"S": ("text#0",), "G": ("generalist_2#0",),
            "SS": ("text#0", "text#1"), "SG": ("text#0", "generalist_2#0"),
            "SSG": ("text#0", "text#1", "generalist_2#0")}


def sha(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _schedule(pool_ids):
    result = []
    for seed in (301, 302):
        cases = [(context, pool_id) for context in CONTEXTS for pool_id in sorted(pool_ids)]
        random.Random(f"coupled-controlled-v1:{seed}").shuffle(cases)
        for index, (context, pool_id) in enumerate(cases):
            aliases = list(CONTEXTS[context])
            random.Random(f"coupled-controlled-v1:{seed}:{index}:aliases").shuffle(aliases)
            identity = dict(seed=seed, index=index, context=context, pool_id=pool_id, aliases=aliases)
            result.append(dict(id=canonical(identity), **identity))
    return result


def _integral(samples, start, end):
    points = sorted((sample["time_ns"], sample["watts"]) for sample in samples)
    if (not finite(start) or not finite(end) or end <= start or len(points) < 2
            or points[0][0] > start or points[-1][0] < end
            or any(type(t) is not int or not finite(watts) or watts < 0 for t, watts in points)):
        return None
    value = 0.
    for (left, a), (right, b) in zip(points, points[1:]):
        lo, hi = max(left, start), min(right, end)
        if hi <= lo:
            continue
        if right <= left or right-left > 2_000_000_000:
            return None
        pa = a+(b-a)*(lo-left)/(right-left)
        pb = a+(b-a)*(hi-left)/(right-left)
        value += .5*(pa+pb)*(hi-lo)/1e9
    return value


def _largest(item, task):
    sample = item["sample"]
    if task != "text":
        return Path(sample["input"]).stat().st_size
    value = "\n".join(str(sample[key]) for key in ("context", "question") if key in sample)
    return len((value or str(sample.get("input", sample.get("text", "")))).encode())


def audit_controlled(packet_path, raw_path):
    packet_path, raw_path = Path(packet_path).resolve(), Path(raw_path).resolve()
    packet, raw = json.loads(packet_path.read_bytes()), json.loads(raw_path.read_bytes())
    checks = []; errors = []; diagnostics = {}

    def check(condition, name, detail=None):
        checks.append({"name": name, "passed": bool(condition), **({"detail": detail} if detail is not None else {})})
        if not condition:
            errors.append(name)

    try:
        check(packet.get("schema") == "gsdeploy.coupled_controlled_calibration_packet.v1", "PACKET_SCHEMA")
        check(raw.get("schema") == "gsdeploy.coupled_controlled_calibration.v1", "RAW_SCHEMA")
        check(raw.get("source_packet_digest") == canonical(packet), "SOURCE_PACKET_DIGEST")
        check(raw.get("source_receipts") == packet["sources"], "SOURCE_RECEIPT_DECLARATION")
        check(raw.get("input_pool") == packet["input_pool"], "INPUT_RECEIPT_DECLARATION")
        bindings = packet["sources"]+packet["bindings"]+[packet["input_pool"], packet["base_packet"]]
        changed = [source["path"] for source in bindings if sha(source["path"]) != source]
        check(not changed, "ALL_SOURCE_INPUT_BINDING_HASHES", changed)
        check(len(packet["sources"]) == len({source["path"] for source in packet["sources"]}), "UNIQUE_SOURCE_PATHS")
        check(packet.get("no_retries") is True and raw.get("no_retries") is True, "NO_RETRY_DECLARATION")
        check(raw.get("backend") == "native" and raw.get("status") == "completed" and raw.get("errors") == [], "SUCCESSFUL_NATIVE_TERMINAL")
        check(raw.get("calibration_usable") is True, "CALIBRATION_USABLE_DECLARATION")
        check(raw.get("U") is None and raw.get("original_stability_evaluated") is False, "NO_FORMAL_UTILITY_OR_STABILITY_CLAIM")
        check(packet.get("seeds") == [301, 302] and raw.get("seeds") == [301, 302], "CONTROL_PHASE_SEEDS")
        check(raw.get("calibration_phase_seeds_are_arrival_seeds") is False, "NO_ARRIVAL_SEED_FICTION")
        check(raw.get("deployment") == PRIMARY and packet.get("deployment") == PRIMARY, "EXACT_PRIMARY_DEPLOYMENT")
        contract = packet["policy_contract_for_preflight_only"]
        check(raw.get("policy_contract") == contract, "PREFLIGHT_CONTRACT_IDENTITY")
        check(raw.get("aliases") == contract["aliases"], "RESIDENT_ALIAS_METADATA")
        aliases = {f"{family}#{index}": family for family, count in PRIMARY.items() for index in range(count)}
        check(raw.get("replica_to_family") == aliases, "RESIDENT_ALIAS_EXPANSION")
        check({alias: spec["family"] for alias, spec in raw["aliases"].items()} == aliases, "ALIAS_FAMILY_IDENTITY")
        check(all(worker.get("env", {}).get(key) == "1" for worker in raw["worker_registry"].values()
                  for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "WORKER_THREAD_LIMITS")
        check(set(raw["worker_registry"]) == set(aliases), "EXACT_WORKER_REGISTRY")
        check(bool(raw.get("preflight", {}).get("ready")), "UNCHANGED_NATIVE_PREFLIGHT_READY")
        check(set(raw["shutdown"]) == set(aliases) and all(not status["forced"] and status["exit_code"] == 0
                  and status["bye_received"] for status in raw["shutdown"].values()), "GRACEFUL_ALL_WORKER_SHUTDOWN")

        pool = [json.loads(line) for line in Path(packet["input_pool"]["path"]).read_text().splitlines() if line.strip()]
        text_ids = [item["pool_id"] for item in pool if item["task_type"] == "text"]
        check(len(text_ids) == 32 and len(set(text_ids)) == 32, "EXACT_32_TEXT_INPUTS")
        expected = _schedule(text_ids)
        check(packet["batches"] == expected and packet["batch_schedule_sha256"] == canonical(expected), "EXACT_RANDOMIZED_SCHEDULE")
        check(packet.get("planned_batches") == 320 and packet.get("planned_text_requests") == 576, "PLANNED_320_BATCH_576_REQUESTS")
        check(len(raw["batches"]) == 320 and len(raw["request_records"]) == 576, "ACTUAL_320_BATCH_576_REQUESTS")
        check(len(raw["batches"]) == len(expected), "NO_MISSING_OR_EXTRA_BATCHES")
        by_id = {row["request_id"]: row for row in raw["request_records"]}
        check(len(by_id) == len(raw["request_records"]), "UNIQUE_REQUEST_IDS")
        check([row["request_id"] for row in raw["request_records"]] == [f"controlled:{index}" for index in range(576)], "EXACT_REQUEST_ID_SEQUENCE")
        all_ids = []; offsets = []; service_starts = []; service_ends = []; previous_end = None
        contexts = Counter(); seeds = Counter()
        for index, (batch, declaration) in enumerate(zip(raw["batches"], expected)):
            check(all(batch.get(key) == value for key, value in declaration.items()), f"BATCH_IDENTITY_{index}")
            check(batch.get("all_completed") is True, f"BATCH_COMPLETED_{index}")
            start, end = batch["start_time_ns"], batch["end_time_ns"]
            check(type(start) is int and type(end) is int and end > start, f"BATCH_TIME_ORDER_{index}")
            check(previous_end is None or start >= previous_end, f"BATCH_DRAIN_ISOLATION_{index}")
            previous_end = end
            rows = [by_id[rid] for rid in batch["request_ids"]]
            check([row["model"] for row in rows] == declaration["aliases"], f"BATCH_ALIAS_ORDER_{index}")
            all_ids += batch["request_ids"]; contexts[batch["context"]] += 1; seeds[batch["seed"]] += 1
            for row in rows:
                rid = row["request_id"]
                check(row["phase"] == "controlled" and row["task"] == "text" and row["pool_id"] == declaration["pool_id"]
                      and row["calibration_seed"] == declaration["seed"] and row["calibration_context"] == declaration["context"]
                      and row["calibration_batch_id"] == declaration["id"] and row["status"] == "completed", "REQUEST_BATCH_IDENTITY_"+rid)
                check(row["scheduled_time_ns"] == start and start <= row["enqueued_time_ns"] <= row["actual_submit_time_ns"]
                      <= row["completion_time_ns"] <= end, "HOST_TIME_AND_DRAIN_"+rid)
                response = row["worker_response"]; service = response["service_ms"]
                check(finite(service) and service > 0, "ACTUAL_POSITIVE_SERVICE_"+rid)
                value = row["actual_service_interval"]
                expected_end = response["end_ns"]
                expected_start = expected_end-round(service*1e6)
                offset = row["completion_unix_ns"]-row["completion_time_ns"]
                check(value["end_unix_ns"] == expected_end and value["start_unix_ns"] == expected_start
                      and value["start_time_ns"] == expected_start-offset and value["end_time_ns"] == expected_end-offset,
                      "EXACT_WORKER_SERVICE_CLOCK_CONVERSION_"+rid)
                check(response["start_ns"] <= expected_start < expected_end <= row["completion_unix_ns"], "WORKER_HANDLER_TIME_ORDER_"+rid)
                check(response.get("request_id") == rid and response.get("pool_id") == row["pool_id"]
                      and response.get("task_type") == "text" and response.get("arrival_index") == row["arrival_index"]
                      and response.get("success") is not False, "WORKER_RESPONSE_IDENTITY_"+rid)
                offsets.append(offset); service_starts.append(value["start_time_ns"]); service_ends.append(value["end_time_ns"])
        check(sorted(all_ids) == sorted(by_id), "EVERY_REQUEST_EXACTLY_ONE_BATCH")
        check(dict(contexts) == {name: 64 for name in CONTEXTS} and dict(seeds) == {301: 160, 302: 160}, "CONTEXT_AND_SEED_COVERAGE")
        diagnostics.update(context_batches=dict(contexts), seed_batches=dict(seeds), request_count=len(by_id),
                           clock_anchor_offset_range_ns=max(offsets)-min(offsets),
                           worker_interval_scope="handler service reconstructed from end and measured service, not GPU kernel observation")

        base = json.loads(Path(packet["base_packet"]["path"]).read_bytes())
        config = base["config"]; tasks = config["workload"]
        largest = {task: [item["pool_id"] for item in sorted((r for r in pool if r["task_type"] == task),
                    key=lambda item: (-_largest(item, task), item["pool_id"]))[:3]] for task in tasks}
        expected_prime = Counter((alias, task, pool_id) for alias, family in aliases.items()
            for task in config["catalog"]["models"][family]["supported_tasks"] if tasks.get(task, 0) > 0
            for pool_id in largest[task])
        observed_prime = Counter((row["model"], row["task"], row["pool_id"]) for row in raw["priming_request_records"])
        check(observed_prime == expected_prime and all(row["status"] == "completed" for row in raw["priming_request_records"]), "EXACT_LARGEST_THREE_ALL_PATH_PRIMING")
        check(max(row["completion_time_ns"] for row in raw["priming_request_records"]) <= raw["batches"][0]["start_time_ns"], "PRIMING_DRAIN_BEFORE_CALIBRATION")

        memory = raw["memory_evidence"]
        peak = max(sample["owned_used_mib"] for sample in memory["samples"])
        safe = peak*config["catalog"]["safe_factor"]
        check(memory.get("measured") is True and memory.get("resource_valid") is True
              and all(not sample["foreign_pids"] and finite(sample["owned_used_mib"]) and sample["owned_used_mib"] >= 0
                      for sample in memory["samples"]), "MEASURED_OWNED_MEMORY_NO_FOREIGN_PROCESSES")
        check(math.isclose(safe, memory["safe_memory_mib"], rel_tol=1e-12, abs_tol=1e-6)
              and safe <= config["gpu_memory_mib"], "ORIGINAL_SAFE_MEMORY_BUDGET")
        energy, window = raw["energy_evidence"], raw["measurement_window"]
        check(energy.get("kind") == "measured" and energy.get("scope") == "whole_gpu_board"
              and energy.get("coverage_valid") is True and energy.get("sampling_errors") == [], "MEASURED_WHOLE_BOARD_ENERGY")
        integrated = _integral(energy["samples"], window["start_time_ns"], window["end_time_ns"])
        check(integrated is not None and finite(energy.get("run_joules"))
              and math.isclose(integrated, energy["run_joules"], rel_tol=1e-9, abs_tol=1e-6), "ORIGINAL_2_SECOND_GAP_ENERGY_INTEGRAL")
        covered = _integral(energy["samples"], min(service_starts), max(service_ends))
        check(covered is not None, "ALL_MEASURED_SERVICE_INTERVALS_POWER_COVERED")
        check(window["start_time_ns"] <= min(service_starts) < max(service_ends) <= window["end_time_ns"]
              and math.isclose(window["duration_seconds"], (window["end_time_ns"]-window["start_time_ns"])/1e9,
                               rel_tol=1e-12, abs_tol=1e-9), "ACTUAL_CALIBRATION_WINDOW_NOT_600S_FICTION")
        diagnostics.update(calibration_joules=integrated, measured_service_span_joules=covered,
                           sampled_safe_memory_mib=safe, calibration_elapsed_seconds=window["duration_seconds"])
    except (KeyError, ValueError, TypeError, OSError, OverflowError) as exc:
        errors.append("AUDIT_EXCEPTION:"+type(exc).__name__+":"+str(exc))
    return dict(schema="gsdeploy.controlled_calibration_independent_audit.v1", verified=not errors,
                checks=checks, errors=errors, diagnostics=diagnostics,
                packet=sha(packet_path), raw=sha(raw_path), audit_source=sha(__file__),
                answers_or_quality_references_extracted=False, GPU_started=False,
                claim="EVIDENCE_IDENTITY_AND_ARITHMETIC_ONLY_NOT_NATIVE_UTILITY_OR_CAUSAL_CERTIFICATE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True); parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_controlled(args.packet, args.raw)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/"audit.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(dict(verified=result["verified"], checks=len(result["checks"]), errors=result["errors"],
                          audit=sha(args.output/"audit.json")), ensure_ascii=False))
    raise SystemExit(0 if result["verified"] else 1)
