#!/usr/bin/env python3
"""Independent arithmetic audit of five explicitly named completed raw runs.

Reads no asr1p9 run directory, fits no predictor, changes no gate, starts no GPU.
Shared task text normalization is retained; request scores, edit distances,
aggregation, cohort, percentiles, memory and power integration are recomputed.
"""
from collections import Counter
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gsdeploy.calibration import _import_source
from gsdeploy.config import SystemConfig
from gsdeploy.evaluation import evaluate_run
from gsdeploy.quality import quality_digest
from gsdeploy.tasks import qa_normalize, asr_normalize, page_ocr_text
from gsdeploy.trace import make_trace, trace_digest, expand_routes
from gsdeploy.routing import Router

STUDY = ROOT / "artifacts/static-spillover-study-20261022"
OUTPUT = STUDY / "interim-data-audit-v1"
ROSTER = (
    ("dev_seed51_G2share0", "development", "development-asr18-v1/experiment-packet.json", "development-asr18-v1/runs/asr18_n2_specialist_share_1p0-attempt1/run.json"),
    ("dev_seed51_G2share0p25", "development", "development-asr18-v1/experiment-packet.json", "development-asr18-v1/runs/asr18_n2_specialist_share_0p75-attempt1/run.json"),
    ("dev_seed51_G2share0p125", "development_interpolation", "development-small-spillover-v2/experiment-packet-asr1p8.json", "development-small-spillover-v2/runs-asr1p8/asr1p8_specialist_share_0p875-attempt1/run.json"),
    ("validation_seed61_G2share0", "validation", "transfer-validation-asr18-v1/experiment-packet.json", "transfer-validation-asr18-v1/runs/validation_asr18_seed61-attempt1/run.json"),
    ("validation_seed62_G2share1of3", "validation", "transfer-validation-asr18-v1/experiment-packet.json", "transfer-validation-asr18-v1/runs/validation_asr18_seed62-attempt1/run.json"),
)
TERMS = "QTLPME"


def receipt(path):
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_write(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, label, absolute=1e-9):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=absolute):
        raise ValueError(f"Arithmetic mismatch {label}: {a!r} vs {b!r}")


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def score(task, prediction, sample):
    if task == "text":
        normalized = qa_normalize(prediction)
        return 1 - int(any(normalized == qa_normalize(ref) for ref in sample["answers"] or [""])), 1
    if task == "image":
        return int(prediction != sample["label"]), 1
    if task == "ocr":
        reference, prediction = page_ocr_text(sample["label"]), page_ocr_text(prediction)
    else:
        # The worker's prediction is already protocol-normalized. Match the
        # frozen scorer: normalize reference and split the actual prediction.
        reference, prediction = asr_normalize(sample["label"]).split(), prediction.split()
    if not reference:
        raise ValueError("Empty quality reference")
    return edit_distance(prediction, reference), len(reference)


def percentile(values, percent):
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lower = math.floor(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def integrate(samples, start, end):
    points = sorted((row["time_ns"], row["watts"]) for row in samples)
    if points[0][0] > start or points[-1][0] < end or end <= start:
        raise ValueError("Power samples do not cover the interval")
    energy, max_gap = [], 0
    for (a, pa), (b, pb) in zip(points, points[1:]):
        left, right = max(a, start), min(b, end)
        if right <= left:
            continue
        if not b > a or (b - a) > 2e9 or min(pa, pb) < 0:
            raise ValueError("Invalid power sample gap/value")
        max_gap = max(max_gap, (b-a)/1e9)
        lpower = pa + (pb-pa)*(left-a)/(b-a)
        rpower = pa + (pb-pa)*(right-a)/(b-a)
        energy.append((lpower+rpower)*(right-left)/2e9)
    return math.fsum(energy), max_gap


def audit_one(label, role, packet_path, path):
    packet, raw_disk = read(packet_path), read(path)
    job = next(j for j in packet["jobs"] if j["id"] == raw_disk["experiment_job_id"])
    source = dict(id=label, **receipt(path), packet_path=str(packet_path), packet_sha256=receipt(packet_path)["sha256"])
    raw, strict, strict_receipt = _import_source(source, {job["identity"]["runtime_code_sha256"]})
    config = SystemConfig(**raw["config"])
    evaluation = evaluate_run(config, copy.deepcopy(raw))
    records, window = raw["request_records"], raw["measurement_window"]
    if raw["status"] != "completed" or not strict["eligible"]:
        raise ValueError("Only completed strictly imported native evidence may enter audit")
    assert config.workload == dict(text=.05, image=.05, ocr=.05, asr=1.8)
    assert window["duration_seconds"] == 600
    trace = make_trace(config.workload, 600, config.random_seed, config.arrival_process)
    if len(trace) != len(records) or trace_digest(trace) != window["trace_sha256"]:
        raise ValueError("Trace cohort mismatch")
    aliases, routes = expand_routes(raw["plan"]["deployment"], raw["plan"]["routing"])
    router = Router(routes)
    quality = config.catalog["quality"]
    if quality_digest(quality["references"]) != quality["identity"]["reference_sha256"]:
        raise ValueError("Reference SHA mismatch")
    orders = copy.deepcopy(quality["input_order"])
    for task, order in orders.items():
        random.Random(f"{config.random_seed}:samples:{task}").shuffle(order)
    totals, latencies, rows, indices = {}, {}, [], Counter()
    for i, ((offset, task), record) in enumerate(zip(trace, records)):
        if (record["task"] != task or record["model"] != router.choose(task)
                or record["request_id"] != f"measurement:{i}" or record["status"] != "completed"
                or record["scheduled_time_ns"] != window["start_time_ns"] + round(offset*1e9)
                or record["pool_id"] != orders[task][indices[task] % len(orders[task])]):
            raise ValueError("Request/route/input-order cohort mismatch")
        response = record["worker_response"]
        if response.get("success") is not True:
            raise ValueError("Unsuccessful model output")
        indices[task] += 1
        numerator, denominator = score(task, response["prediction"], quality["references"][task][record["pool_id"]])
        a, b = totals.get(task, (0, 0))
        totals[task] = (a + numerator, b + denominator)
        e2e = (record["completion_time_ns"] - record["scheduled_time_ns"])/1e6
        runtime = (record["completion_time_ns"] - record["actual_submit_time_ns"])/1e6
        delay = (record["actual_submit_time_ns"] - record["scheduled_time_ns"])/1e6
        latencies.setdefault(task, []).append(e2e)
        for key, value in (("e2e_latency_ms", e2e), ("runtime_latency_ms", runtime), ("schedule_delay_ms", delay)):
            close(value, record[key], label + "/" + key)
        rows.append(dict(run_id=label,role=role,seed=config.random_seed,request_id=record["request_id"],
            task=task,sample_id=record["pool_id"],model_alias=record["model"],model_family=aliases[record["model"]],
            scheduled_offset_seconds=offset,status=record["status"],e2e_latency_ms=e2e,
            runtime_latency_ms=runtime,schedule_delay_ms=delay,worker_handler_ms=record["worker_latency_ms"],
            worker_queue_wait_ms=record.get("worker_queue_wait_ms"),quality_error_numerator=numerator,
            quality_reference_denominator=denominator,completed_within_window=record["completion_time_ns"] <= window["end_time_ns"]))
    task_weight = {t: q/math.fsum(config.workload.values()) for t,q in config.workload.items()}
    per_task = {t: dict(requests=indices[t],unique_inputs=len({r["pool_id"] for r in records if r["task"]==t}),
        error_numerator=totals[t][0],reference_denominator=totals[t][1],quality_loss=totals[t][0]/totals[t][1],
        mean_ms=math.fsum(latencies[t])/len(latencies[t]),p95_ms=percentile(latencies[t],95),
        actual_route_counts=dict(Counter(r["model"] for r in records if r["task"]==t))) for t in config.workload}
    within = sum(row["completed_within_window"] for row in rows)
    memory = raw["memory_evidence"]
    peaks = [memory["before"],memory["after"],*memory["samples"]]
    safe = config.catalog["safe_factor"]*max(row["owned_used_mib"] for row in peaks)
    close(safe,memory["safe_memory_mib"],label+"/safe_memory")
    energy = raw["energy_evidence"]
    run_j, run_gap = integrate(energy["samples"],window["start_time_ns"],window["end_time_ns"])
    startup_j, startup_gap = integrate(energy["samples"],energy["startup_begin_time_ns"],window["start_time_ns"])
    close(run_j,energy["run_joules"],label+"/run_joules",1e-6)
    close(startup_j,energy["startup_joules"],label+"/startup_joules",1e-6)
    terms = dict(Q=math.fsum(task_weight[t]*per_task[t]["quality_loss"] for t in config.workload),
        T=1-within/len(records),
        L=math.fsum(task_weight[t]*per_task[t]["mean_ms"]/config.latency_reference_ms[t] for t in config.workload),
        P=math.fsum(task_weight[t]*per_task[t]["p95_ms"]/config.latency_reference_ms[t] for t in config.workload),
        M=safe/config.gpu_memory_mib,
        E=(run_j/600 + startup_j/(config.startup_amortization_seconds or 600))/config.energy_reference_watts)
    contributions = {k:-config.weights[k]*terms[k] for k in TERMS}
    utility = math.fsum(contributions.values())
    cache = raw_disk["evaluation"]["utility"]
    differences = {}
    for name, other in (("strict_import",strict),("evaluate_run",evaluation["utility"]),("cached_raw_utility",cache)):
        differences[name]={k:terms[k]-other[k+"_norm"] for k in TERMS}
        differences[name]["U"]=utility-other["U"]
        for key, difference in differences[name].items():
            close(difference,0,label+"/"+name+"/"+key)
    for task, values in per_task.items():
        measured = strict["quality_evidence"]["per_task"][task]
        close(values["error_numerator"],measured["error_numerator"],label+"/quality_numerator")
        close(values["reference_denominator"],measured["reference_denominator"],label+"/quality_denominator")
    condition = dict(id=label,role=role,source=source,strict_import_receipt=strict_receipt,
        executed_measurement_id=raw["executed_measurement_id"],plan=raw["plan"],workload=config.workload,
        seed=config.random_seed,horizon_seconds=600,weights=config.weights,latency_reference_ms=config.latency_reference_ms,
        task_weights=task_weight,terms=terms,contributions=contributions,U=utility,per_task=per_task,
        cohort=dict(verified=True,trace_sha256=window["trace_sha256"],offered=len(records),completed=len(records),
            within_window=within,during_drain=len(records)-within,input_sequence_sha256=quality_digest([(r["task"],r["pool_id"]) for r in records])),
        memory=dict(safe_factor=config.catalog["safe_factor"],peak_owned_mib=max(row["owned_used_mib"] for row in peaks),
            safe_memory_mib=safe,budget_mib=config.gpu_memory_mib,samples=len(memory["samples"]),resource_valid=memory["resource_valid"]),
        energy=dict(run_joules=run_j,startup_joules=startup_j,run_max_gap_seconds=run_gap,startup_max_gap_seconds=startup_gap,
            samples=len(energy["samples"]),reference_watts=config.energy_reference_watts,startup_amortization_seconds=config.startup_amortization_seconds,
            scope=energy["scope_note"]),arithmetic_crosschecks=differences,recomputed_evaluation=evaluation,
        strict_utility=strict,cached_utility=cache,quality_direction="loss: lower is better; text 1-EM/image 1-accuracy/OCR CER/ASR WER")
    return condition,rows,raw


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    sources,conditions,requests,raws=[],[],[],{}
    for label,role,packet_name,run_name in ROSTER:
        packet,path=STUDY/packet_name,STUDY/run_name
        sources.extend((receipt(packet),receipt(path)))
        condition,rows,raw=audit_one(label,role,packet,path)
        conditions.append(condition);requests.extend(rows);raws[label]=raw
    development=[row for row in conditions if row["seed"]==51]
    if len({row["cohort"]["trace_sha256"] for row in development}) != 1 or len({row["cohort"]["input_sequence_sha256"] for row in development}) != 1:
        raise ValueError("Development trio does not share trace and input cohort")
    comparisons=[]
    for a,b in itertools.combinations(development,2):
        ar,br=raws[a["id"]]["request_records"],raws[b["id"]]["request_records"]
        delta=[(y["completion_time_ns"]-y["scheduled_time_ns"]-x["completion_time_ns"]+x["scheduled_time_ns"])/1e6 for x,y in zip(ar,br)]
        comparisons.append(dict(a=a["id"],b=b["id"],delta_U_b_minus_a=b["U"]-a["U"],
            delta_terms_b_minus_a={k:b["terms"][k]-a["terms"][k] for k in TERMS},
            delta_contributions_b_minus_a={k:b["contributions"][k]-a["contributions"][k] for k in TERMS},
            paired_request_mean_e2e_delta_ms=math.fsum(delta)/len(delta),paired_request_p95_e2e_delta_ms=percentile(delta,95),
            scope="One shared seed51 trace/input sequence; descriptive paired requests, not independent replications or causal significance",p_value=None))
    validation_path=STUDY/"transfer-validation-asr18-v1/validation-report.json"
    validation=read(validation_path);sources.append(receipt(validation_path))
    for held in validation["conditions"]:
        condition=next(row for row in conditions if row["seed"]==held["seed"])
        for key in ["U",*[k+"_norm" for k in TERMS]]:
            manual=condition["U"] if key=="U" else condition["terms"][key[0]]
            close(manual,held["measured"][key],"validation_report/"+key)
    if validation["status"]!="TRANSFER_VALIDATION_FAILED":
        raise ValueError("Unexpected validation state; audit cannot change or relabel it")
    report=dict(schema="gsdeploy.interim_independent_arithmetic_audit.v1",status="FIVE_COMPLETED_RUNS_ARITHMETIC_VERIFIED",
        conditions=conditions,development_descriptive_pairs=comparisons,validation_failure=validation,
        validation_scope="Seeds61/62 differ; no paired causal deployment comparison. Failed zero-refit transfer remains unvalidated.",
        formulas=dict(Q="sum(rate/total * actual-output error numerator/reference denominator)",T="1-completed_within_window/offered",
            L="sum(rate/total * task mean e2e/reference latency)",P="sum(rate/total * task p95 e2e/reference latency)",
            M="safe_factor * sampled owned-process peak / GPU budget",E="(window joules/H + startup joules/amortization)/reference watts",U="-sum(weight * normalized loss)"),
        limitations=["Fixed input pool; repeated requests are not independent quality samples", "Only one seed for three development routes; no significance test",
            "Shared frozen text normalizers retained; scoring/edit-distance aggregation otherwise independently recomputed",
            "Only completed explicitly listed five runs; no ASR1.9 run directory read", "No fitting, gate change, formal search certification or GPU execution"],gpu_started=False)
    write(OUTPUT/"audit.json",report)
    csv_write(OUTPUT/"plans-six-terms.csv",[dict(id=c["id"],role=c["role"],seed=c["seed"],asr_rps=c["workload"]["asr"],
        G2_asr_share=c["plan"]["routing"]["asr"].get("generalist_2",0)/c["workload"]["asr"],
        offered=c["cohort"]["offered"],during_drain=c["cohort"]["during_drain"],
        **{k+"_norm":c["terms"][k] for k in TERMS},**{k+"_contribution":c["contributions"][k] for k in TERMS},U=c["U"],
        asr_WER=c["per_task"]["asr"]["quality_loss"],text_EM=1-c["per_task"]["text"]["quality_loss"],
        image_accuracy=1-c["per_task"]["image"]["quality_loss"],ocr_CER=c["per_task"]["ocr"]["quality_loss"],
        strict_import_verified=True,raw_sha256=c["source"]["sha256"]) for c in conditions])
    csv_write(OUTPUT/"request-latencies.csv",requests)
    lines=["# 五个已完成 GPU 运行的独立算术审计","", "五条原始记录均通过严格导入；从实际输出重算质量分子/分母、时间戳重算延迟和吞吐损失、采样峰值重算内存、功率采样积分重算能耗，六项及总 U 与缓存/公共评价器一致（数值容差 1e-9）。质量采用损失，越小越好；U 越大越好。","",
        "| 方案 | seed | G2 分流 | Q损失 | T | L | P | M | E | U |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in conditions:
        share=c["plan"]["routing"]["asr"].get("generalist_2",0)/1.8
        lines.append(f"| {c['id']} | {c['seed']} | {share:.3f} | "+" | ".join(f"{c['terms'][k]:.6f}" for k in TERMS)+f" | {c['U']:.9f} |")
    lines.extend(["", "seed51 三条运行的到达 trace、输入序列、负载与时长完全一致，可作单次共同 trace 下的描述性配对；实际执行时刻和 GPU 干扰仍可能不同，不把请求条数当独立重复，不计算显著性。"])
    for pair in comparisons:
        lines.append(f"- {pair['b']} 相对 {pair['a']}：ΔU={pair['delta_U_b_minus_a']:+.9f}。")
    lines.extend(["", "seed61 与 seed62 是不同 trace/输入顺序的两个留出点，路由也不同，本审计不把二者作配对部署因果比较。旧零重拟合预测在新负载上的验证状态仍为 TRANSFER_VALIDATION_FAILED；两点都未通过，排序也不一致，不能转为 validated。", "", "原始功耗范围为 GPU 整板，包含启动/预热摊销，不含 CPU 与窗口后的排空；内存为采样峰值。完整分子分母、每任务延迟、六权重贡献、导入收据及验证误差见 audit.json；逐请求延迟及评分分子分母见 CSV。未读取未结束的 ASR1.9 运行目录，未改变任何数据/门禁，未启动 GPU。"])
    (OUTPUT/"report.zh.md").write_text("\n".join(lines)+"\n")
    unique={row["path"]:row for row in sources}
    if any(receipt(row["path"])!=row for row in unique.values()):
        raise ValueError("Input bytes changed during audit")
    write(OUTPUT/"manifest.json",dict(schema="gsdeploy.interim_audit_manifest.v1",inputs=list(unique.values()),
        source_code=[receipt(ROOT/"gsdeploy"/name) for name in ("evaluation.py","quality.py","tasks.py","metrics.py","energy.py","prediction.py","calibration.py","trace.py","routing.py")],
        script=receipt(__file__),outputs=[receipt(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()],
        input_bytes_unchanged=True,gpu_started=False))
    print(json.dumps(dict(status=report["status"],runs=len(conditions),request_rows=len(requests),output=str(OUTPUT),gpu_started=False)))


if __name__=="__main__":
    main()
