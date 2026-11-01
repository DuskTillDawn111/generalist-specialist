"""CPU-only, read-only import of retained measurements into a static study.

Checkpoint hashes are configuration/receipt identities, not a new weight audit.
Standalone closed-loop profiles are explicitly a transfer proxy for mixed load.
"""

from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import types

from ..tasks import page_ocr_text, score_result
from ..trace import make_trace, trace_digest

TASKS = ("text", "image", "ocr", "asr")
SLOTS = ("generalist_1", "generalist_2", "generalist_3", "text", "image", "ocr", "asr")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def score_sample(sample, output):
    """Use existing native task arithmetic; PP output needs the frozen CER view."""
    output = dict(output)
    if sample["task"] == "ocr":
        output["prediction"] = page_ocr_text(output["prediction"])
    score = score_result(sample, output)
    if sample["task"] == "text":
        score["correct"] = bool(score["EM"])
    return score


def shared_protocol(specs_by_task):
    """Preserve each task's exact request policy and native decoding preset."""
    policies = {task: read(spec["resources"]["protocol"]) for task, spec in specs_by_task.items()}
    combined = copy.deepcopy(next(iter(policies.values())))
    for task, policy in policies.items():
        combined["tasks"][task] = copy.deepcopy(policy["tasks"][task])
        for model, presets in policy.get("native_decoding_presets", {}).items():
            if task in presets:
                combined.setdefault("native_decoding_presets", {}).setdefault(model, {})[task] = presets[task]
    return combined


def frozen_compact_adapter(source_root):
    """Import the measured adapter under an isolated CPU namespace."""
    source_root = Path(source_root)
    name = "_static_frozen_" + digest(str(source_root))[:12]
    module_name = name + ".backends.compact"
    if module_name not in sys.modules:
        for suffix, path in (("", source_root / "gsdeploy"), (".backends", source_root / "gsdeploy/backends")):
            package = types.ModuleType(name + suffix)
            package.__path__ = [str(path)]
            sys.modules[name + suffix] = package
        definition = importlib.util.spec_from_file_location(module_name, source_root / "gsdeploy/backends/compact.py")
        module = importlib.util.module_from_spec(definition)
        sys.modules[module_name] = module
        definition.loader.exec_module(module)
    return sys.modules[module_name].CompactAdapter


def validate_shared_requests(specs_by_task, merged, cohorts, source_root):
    CompactAdapter = frozen_compact_adapter(source_root)
    merged_adapter = CompactAdapter(merged, read(next(iter(specs_by_task.values()))["resources"]["classmap"]))
    count = 0
    for task, spec in specs_by_task.items():
        original = CompactAdapter(read(spec["resources"]["protocol"]), read(spec["resources"]["classmap"]))
        for sample in cohorts[task]:
            old = original.build(spec["candidate"], sample)
            new = merged_adapter.build(spec["candidate"], sample)
            old.pop("protocol_sha256")
            new.pop("protocol_sha256")
            if old != new:
                raise ValueError("Shared policy changed native request: " + task + "/" + sample["id"])
            count += 1
    return count


def checkpoint_inventory(checkpoint):
    """Check file sizes and safetensors headers without reading gigabyte tensors."""
    folder = Path(checkpoint["path"] if isinstance(checkpoint, dict) else checkpoint)
    expected = {r["name"]: r for r in checkpoint.get("files", [])} if isinstance(checkpoint, dict) else {}
    # The PP root contains other rejected OCR candidates; bind only the two retained ones.
    if not isinstance(checkpoint, dict):
        files = [folder / f"PaddlePaddle--PP-OCRv6_medium_{part}_safetensors/model.safetensors"
                 for part in ("det", "rec")]
    else:
        files = sorted(folder.glob("*.safetensors"))
    observations = []
    for path in files:
        try:
            stat = path.stat()
            with path.open("rb") as stream:
                raw = stream.read(8)
                length = struct.unpack("<Q", raw)[0] if len(raw) == 8 else 0
                header_valid = 1 <= length <= min(stat.st_size - 8, 16 * 1024 * 1024)
                if header_valid:
                    header_valid = isinstance(json.loads(stream.read(length)), dict)
            size_expected = expected.get(path.name, {}).get("size")
            valid = bool(header_valid and (size_expected is None or stat.st_size == size_expected))
            observations.append(dict(path=str(path), size=stat.st_size,
                allocated_bytes=stat.st_blocks * 512, expected_size=size_expected,
                header_valid=bool(header_valid), ready=valid,
                historical_sha256=expected.get(path.name, {}).get("sha256")))
        except (OSError, ValueError, struct.error) as error:
            observations.append(dict(path=str(path), ready=False, error=str(error)))
    return dict(path=str(folder), files=observations, ready=bool(observations) and all(r["ready"] for r in observations),
                verification="Size and safetensors JSON header only; historical receipt SHA retained, weights not rehashed.")


def prepare_static_study(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    base = root / "artifacts/retained-replica-profiles-20261021"
    combined = read(base / "combined-summary.json")
    completed = [r for r in combined["rows"] if r["status"] == "COMPLETE"]
    failed = [r for r in combined["rows"] if r["status"] != "COMPLETE"]
    if len(completed) != 61 or len(failed) != 6:
        raise ValueError("Unexpected retained resource grid identity")
    entries = {}
    source_bindings = {}

    def bind(path):
        path = Path(path).resolve()
        source_bindings[str(path)] = sha(path)
        return str(path)

    for path in (Path(__file__), root / "scripts/prepare_static_study.py", root / "scripts/static_study_worker.py",
                 root / "gsdeploy/tasks.py", root / "gsdeploy/runtime/protocol.py"):
        bind(path)

    for package in (base, base / "protocol-binding-correction-v2"):
        release = read(package / "release.json")
        bind(package / "release.json")
        for entry in release["entries"]:
            entries[entry["id"]] = entry
            for path in Path(entry["source"]).glob("gsdeploy/**/*.py"):
                bind(path)
    bind(base / "combined-summary.json")
    bind(base / "output-equivalence.json")
    equivalence = read(base / "output-equivalence.json")["rows"]

    # References must match the actual resource cohort and active adapter, not older QA200.
    qa_reference = root / "artifacts/qa-development-revision-20261020/tydiqa-new-native-confirmation80-v2/confirmation80.jsonl"
    ocr_reference = root / "artifacts/ocr-reference-order-audit-20261021/completed-source-review-v3/references200.jsonl"
    ocr_freeze = ocr_reference.parent / "freeze.json"
    for path, expected in read(ocr_freeze)["source_bindings"].items():
        if sha(path) != expected:
            raise ValueError("OCR reference release mismatch: " + path)
    for path in (qa_reference, ocr_reference, ocr_freeze):
        bind(path)
    references = {"text": {r["id"]: dict(r, answers=r["native_answers"]) for r in rows(qa_reference)},
                  "ocr": {r["id"]: dict(r, task="ocr") for r in rows(ocr_reference)}}
    for task in ("image", "asr"):
        origin = Path(entries[f"{task}-{task}"]["origin_job"])
        references[task] = {r["id"]: r for r in read(origin)["samples"]}
        bind(origin)

    cohorts, ordered_inputs, selected_references = {}, {}, {}
    pool_rows = []
    for task in TASKS:
        cohort_path = base / "cohorts" / (task + ".json")
        cohorts[task] = read(cohort_path)
        bind(cohort_path)
        ordered_inputs[task] = [r["id"] for r in cohorts[task]]
        selected_references[task] = {}
        for sample in cohorts[task]:
            sid = sample["id"]
            ref = references[task].get(sid)
            if ref is None:
                raise ValueError(f"Missing reference: {task}/{sid}")
            if sample.get("input_sha256"):
                if ref.get("input_sha256") != sample["input_sha256"] or sha(sample["input"]) != sample["input_sha256"]:
                    raise ValueError("Reference/media identity mismatch: " + sid)
                bind(sample["input"])
            elif sample.get("context") != ref.get("context") or sample.get("question") != ref.get("question"):
                raise ValueError("QA reference/input identity mismatch: " + sid)
            selected_references[task][sid] = ref
            # Runtime receives no labels or answer aliases (including native_answers).
            clean = {k: v for k, v in sample.items() if k not in
                     {"label", "answers", "answer", "reference", "references", "native_answers", "is_impossible"}}
            pool_rows.append(dict(pool_id=sid, task_type=task, sample=clean))
    output.mkdir(parents=True)
    pool_path = output / "runtime-inputs.jsonl"
    pool_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool_rows))
    write(output / "references.json", selected_references)

    models, registry, quality_paths, curves = {}, {}, {}, {}
    for slot in SLOTS:
        tasks = list(TASKS) if slot.startswith("generalist") else [slot]
        paths, workers, all_specs, specs_by_task = {}, {}, [], {}
        for task in tasks:
            entry = entries[f"{slot}-{task}"]
            spec_path = Path(entry["folder"]) / "worker.json"
            spec = read(spec_path)
            bind(spec_path)
            all_specs.append(spec)
            specs_by_task[task] = spec
            protocol = spec["resources"].get("protocol")
            if protocol:
                bind(protocol)
            classmap = spec["resources"].get("classmap")
            if classmap:
                bind(classmap)
            source = spec["resources"].get("inference_source")
            if source:
                bind(source)
            selected = [r for r in completed if r["slot"] == slot and r["task"] == task]
            n1 = next(r for r in selected if r["replicas"] == 1)
            folder = Path(n1["profile"]).parent
            observed, variants, score_cache = {}, {}, {}
            for raw_path in sorted(folder.glob(task + "-round*.jsonl")):
                bind(raw_path)
                for result in rows(raw_path):
                    sid = result["sample_id"]
                    if sid not in selected_references[task]:
                        raise ValueError("Unexpected measured sample " + sid)
                    score_key = (sid, result["prediction"], result["success"])
                    if score_key not in score_cache:
                        score_cache[score_key] = score_sample(selected_references[task][sid], result)
                    score = score_cache[score_key]
                    normalized = page_ocr_text(result["prediction"]) if task == "ocr" else result["prediction"]
                    item = dict(score, prediction=normalized, success=result["success"], source=str(raw_path))
                    observed.setdefault(sid, item)
                    variants.setdefault(sid, set()).add(digest(dict(score=score, prediction=normalized)))
            if set(observed) != set(ordered_inputs[task]):
                raise ValueError("Incomplete measured quality cohort: " + slot + "/" + task)
            changed_ids = {v["sample_id"] for audit in equivalence if audit["id"] == entry["id"]
                           and audit["package"] == n1["package"] for v in audit["changes_vs_n1"]}
            for sid, item in observed.items():
                item["variability"] = dict(n1_output_variants=len(variants[sid]),
                    cross_replica_change_observed=sid in changed_ids,
                    deterministic=len(variants[sid]) == 1 and sid not in changed_ids)
            metric = {"text": "em", "image": "accuracy", "ocr": "cer", "asr": "wer"}[task]
            if task in ("text", "image"):
                value = sum(r["correct"] for r in observed.values()) / len(observed)
            else:
                value = sum(r["errors"] for r in observed.values()) / sum(r["reference_units"] for r in observed.values())
            identity = dict(checkpoint=spec["resources"]["checkpoint"], execution=spec["execution"],
                            adapter=spec.get("adapter"), peft_adapter=spec["execution"].get("peft_adapter"))
            paths[task] = dict(metric=metric, unit="fraction", value=value,
                checkpoint_identity=dict(kind="configuration_manifest", sha256=digest(identity), source=str(spec_path)),
                source=str(spec_path), samples=observed, evidence_status="OBSERVED_N1_COHORT_REPLAY",
                protocol_sha256=sha(protocol) if protocol else sha(source),
                reference_sha256=digest(selected_references[task]),
                applicability="First observed n1 result per input; output variation disclosed, mixed-run determinism unverified.")
            workers[task] = dict(spec=str(spec_path), spec_sha256=sha(spec_path), python=entry["python"],
                source=entry["source"], protocol=protocol,
                output_variation=sum(v["changed_requests_vs_n1"] for v in equivalence
                                     if v["id"] == entry["id"] and v["package"] == n1["package"]))
            curves.setdefault(slot, {})[task] = {}
            for point in selected:
                bind(point["profile"])
                raw_sources = [bind(p) for p in Path(point["profile"]).parent.glob(task + "-round*.jsonl")]
                curves[slot][task][str(point["replicas"])] = dict(mean_service_ms=point["mean_service_ms"],
                    median_rps=point["median_rps"], capacity_ratio_vs_n1=point["median_rps_ratio_vs_n1"],
                    resident_process_mib=point["process_warm_sum_GiB"] * 1024,
                    sampled_board_peak_mib=point["board_peak_including_load_GiB"] * 1024,
                    source=point["profile"], raw_service_samples=raw_sources,
                    scope="standalone_same_task_closed_loop_transfer_proxy")
        if any(s["execution"] != all_specs[0]["execution"] for s in all_specs):
            raise ValueError("Cannot share weights across differing executions: " + slot)
        quality_paths[slot] = paths
        registry[slot] = dict(model=all_specs[0]["candidate"]["model"], tasks=workers,
                             checkpoint=checkpoint_inventory(all_specs[0]["resources"]["checkpoint"]),
                             execution=all_specs[0]["execution"],
                             shared_worker_status="TASK_CONFIGS_MATCH_SHARED_PROTOCOL_PREPARATION_REQUIRED" if len(tasks) > 1 else "SINGLE_TASK")
        runtime_spec = copy.deepcopy(all_specs[0])
        runtime_spec["task_candidates"] = {t: spec["candidate"] for t, spec in specs_by_task.items()}
        if slot != "ocr":
            merged = shared_protocol(specs_by_task)
            checked = validate_shared_requests(specs_by_task, merged, cohorts, entries[f"{slot}-{tasks[0]}"]["source"])
            protocol_path = output / "workers" / slot / "protocol.json"
            write(protocol_path, merged)
            runtime_spec["resources"]["protocol"] = str(protocol_path)
            registry[slot].update(shared_worker_status="CPU_REQUEST_EQUIVALENCE_VERIFIED", cpu_checked_requests=checked,
                                  protocol_sha256=sha(protocol_path))
        else:
            runtime_spec["adapter_source"] = bind(root / "scripts/retained_ppocrv6_profile_adapter.py")
            runtime_spec["adapter_source_sha256"] = sha(runtime_spec["adapter_source"])
        runtime_spec["resources"]["request_output"] = str(output / "worker-native-output" / slot)
        spec_path = output / "workers" / slot / "worker.json"
        write(spec_path, runtime_spec)
        worker_script = root / "scripts/static_study_worker.py"
        registry[slot].update(spec=str(spec_path), spec_sha256=sha(spec_path),
            command=[workers[tasks[0]]["python"], str(worker_script), "--spec", str(spec_path),
                     "--source-root", entries[f"{slot}-{tasks[0]}"]["source"], "--execute"],
            tasks_supported=tasks,
            cpu_preflight_command=[workers[tasks[0]]["python"], str(worker_script), "--spec", str(spec_path),
                "--source-root", entries[f"{slot}-{tasks[0]}"]["source"], "--check",
                "--pool", str(pool_path)])
        checkpoint = all_specs[0]["resources"]["checkpoint"]
        registry[slot]["env"] = dict(CUDA_VISIBLE_DEVICES="0",
            PYTORCH_CUDA_ALLOC_CONF=runtime_spec["execution"].get("allocator_conf", ""),
            GS_MODEL_PATH=checkpoint["path"] if isinstance(checkpoint, dict) else checkpoint,
            GS_PROMPT_FILE=runtime_spec["resources"].get("protocol", runtime_spec["resources"].get("inference_source")))
        registry[slot]["cwd"] = str(root)
        models[slot] = dict(role="generalist" if slot.startswith("generalist") else "specialist",
            name=registry[slot]["model"], supported_tasks=tasks, executable_tasks=tasks,
            service_time_ms={t: curves[slot][t]["1"]["mean_service_ms"] for t in tasks})

    quality = dict(schema_version=1, mode="per_sample_replay", paths=quality_paths,
        input_order=ordered_inputs, references=selected_references,
        identity=dict(input_pool_sha256=sha(pool_path), reference_sha256=digest(selected_references),
                      protocol_sha256=digest({m: {t: p["protocol_sha256"] for t, p in ts.items()} for m, ts in quality_paths.items()}),
                      source=str(output)),
        scope="Observed resource cohort, not fresh independent confirmation; no absolute quality gate.",
        ocr_reference=dict(source=str(ocr_reference), freeze_sha256=sha(ocr_freeze), external_gold_certified=False,
                           uncertainty_pages_full200=52))
    memory, factors = {}, {}
    domain, common_domain = {}, {}
    for slot, task_curves in curves.items():
        # A model at n may serve only tasks measured at n; missing positive paths remain unknown.
        states = [set(int(n) for n in c) for c in task_curves.values()]
        counts = sorted(set.union(*states))
        domain[slot] = [0] + counts
        common_domain[slot] = [0] + sorted(set.intersection(*states))
        factors[slot] = {str(n): float(n) for n in [0] + counts}
        memory[slot] = {"0": 0.0, **{str(n): max(c[str(n)]["resident_process_mib"] for c in task_curves.values() if str(n) in c) for n in counts}}
    catalog = dict(capability_contract="verified_executable_v1", performance_mode="task_replica_proxy",
        safe_factor=1.110854, models=models, capacity_factors=factors, memory_table_mib=memory,
        task_replica_profiles=curves, quality=quality,
        measurement_scope="Standalone closed-loop service/memory transfer proxy; no mixed-model or open-loop certification.",
        memory_scope="Per-state maximum measured process residency across tasks; mixed warmup/peak unknown.",
        energy_profiles={})
    mini_sum = sum(models["generalist_2"]["service_time_ms"].values()) / 1000
    rate = 0.7 / mini_sum
    workload = {t: rate for t in TASKS}
    seeds = {"development": [42], "confirmation": [43, 44, 45]}
    trace_manifest = []
    for phase, values in seeds.items():
        for seed in values:
            trace = make_trace(workload, 1800, seed, "poisson")
            trace_path = output / "traces" / f"{phase}-{seed}.json"
            write(trace_path, trace)
            trace_manifest.append(dict(phase=phase, seed=seed, path=str(trace_path), sha256=sha(trace_path),
                trace_digest=trace_digest(trace), counts=dict(Counter(t for _, t in trace))))
    config = dict(catalog=str(output / "catalog.json"), utility_schema="six_term_quality_v1",
        workload=workload, gpu_memory_mib=49140, latency_reference_ms={"text": 1000, "image": 250, "ocr": 10000, "asr": 3000},
        weights={k: 1 / 6 for k in ("Q", "T", "L", "P", "M", "E")}, route_grid=4,
        horizon_seconds=1800, random_seed=42, arrival_process="poisson", drain_seconds=300,
        energy_reference_watts=450, startup_amortization_seconds=3600, solver="milp",
        milp_capacity_witness=False, milp_time_limit_seconds=60, search_max_states=100000,
        milp_max_iterations=1000, milp_max_columns=50000, milp_absolute_gap=1e-6)
    pools = dict(schema="gsdeploy.static_pool_manifest.v1", decision_sources=[bind(root / "docs/current_progress_20261021.md"),
        bind(root / "artifacts/ocr-small-candidate-screen-20261021/ppocrv6-user-retention-20261021.json"), str(base / "release.json")],
        pools={"full_retained": dict(members=list(SLOTS), status="CURRENT_RETAINED_SEVEN_MAIN_DOMAIN"),
               "generalist_only": dict(members=list(SLOTS[:3]), status="STRUCTURAL_CONTROL_NOT_USER_NAMED_POOL"),
               "specialist_only": dict(members=list(SLOTS[3:]), status="STRUCTURAL_CONTROL_NOT_USER_NAMED_POOL")},
        historical_abc_mapping="UNRESOLVED_OLD_CATALOGS_NOT_REUSED")
    blockers = ["Mixed-model/multi-task service and whole-board open-loop energy evidence not yet measured.",
                "Shared multitask workers passed CPU request equivalence; GPU multi-task warmup/memory unverified.",
                "Historical named A/B/C pool mapping is unresolved; main seven-model pool and structural controls are explicit.",
                "GPU and runtime environment readiness require execution-host preflight."]
    contract = dict(schema="gsdeploy.static_study_contract.v6", phase="INPUTS_FROZEN_GPU_NOT_STARTED",
        workload=workload, workload_rule=dict(reference_model="generalist_2", sum_single_task_service_seconds=mini_sum,
            target_rho_proxy=0.7, rule="equal_task_lambda=0.7/sum(mean_service_seconds)", mixed_capacity_certified=False),
        runtime_input_pool=str(pool_path), input_pool_sha256=sha(pool_path), references_sha256=digest(selected_references),
        distinct_inputs={t: len(c) for t, c in cohorts.items()}, formal_horizon_seconds=1800,
        sample_count_rule="H fixed before search; >=100 expected requests per task; repetitions do not add independent quality samples.",
        expected_requests_per_task=1800 * rate, traces=trace_manifest, seeds=seeds,
        warmup=dict(seconds=30, policy="Largest three inputs per task before arrivals; shared worker warms all supported tasks."),
        calibration_horizon_seconds=600, calibration_identity_separate=True, drain_seconds=300,
        replica_domain=domain, common_task_subdomain=common_domain,
        missing_task_replica_policy="Positive task paths without that n profile remain unknown; no whole-model exclusion.",
        route_domain="union of rational simplex grids with denominators 1..K, K=4; no private witnesses",
        utility=config["weights"], utility_schema=config["utility_schema"], absolute_quality_gate=False,
        quality_missing_paths=[], resource_grid=dict(complete=61, measured_failures=6, model_task_paths=16),
        blockers=blockers, gpu_started=False)
    write(output / "study-contract.json", contract)
    write(output / "pool-manifest.json", pools)
    write(output / "worker-registry.json", registry)
    write(output / "profile-index.json", dict(rows=combined["rows"], stop_receipts=combined["stop_receipts"],
        initial_protocol_binding_failures=combined["initial_protocol_binding_failures"], scope=catalog["measurement_scope"]))
    write(output / "quality-index.json", quality)
    write(output / "catalog.json", catalog)
    write(output / "system-config.json", config)
    write(output / "source-bindings.json", source_bindings)
    artifact_hashes = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}
    write(output / "manifest.json", dict(schema="gsdeploy.static_input_bundle.v1", files=artifact_hashes,
        source_bindings_sha256=sha(output / "source-bindings.json"), gpu_started=False))
    return contract
