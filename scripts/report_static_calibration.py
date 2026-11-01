"""Read-only native calibration observations; never fit a partial predictor."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.calibration import _baseline, _import_source
from gsdeploy.config import SystemConfig
from gsdeploy.evidence import digest
from gsdeploy.metrics import mean, percentile
from gsdeploy.prediction import predict
from gsdeploy.quality import METRICS, quality_digest


def distribution(values):
    return {"observations": len(values), "mean_ms": mean(values), "p95_ms": percentile(values, 95)}


def error(predicted, observed):
    if predicted is None or observed is None:
        return {"predicted": predicted, "observed": observed, "signed_error": None,
                "absolute_error": None, "relative_error": None}
    return {"predicted": predicted, "observed": observed, "signed_error": predicted - observed,
            "absolute_error": abs(predicted - observed),
            "relative_error": abs(predicted - observed) / abs(observed) if observed else None}


def observation(source, packet):
    versions = {job["identity"]["runtime_code_sha256"] for job in packet["jobs"]}
    raw, measured, provenance = _import_source(source, versions)
    config = SystemConfig(**raw["config"])
    job = next(j for j in packet["jobs"] if j["id"] == raw["experiment_job_id"])
    x, routing = raw["plan"]["deployment"], raw["plan"]["routing"]
    proxy_data = copy.deepcopy(config.catalog)
    proxy_data["performance_mode"], proxy_data["memory_mode"] = "task_replica_proxy", "standalone_proxy"
    # No newly measured energy may leak into an uncalibrated proxy diagnostic.
    proxy_data.pop("energy_profiles", None)
    predicted = predict(proxy_data, x, routing, config.workload, config.gpu_memory_mib,
        config.latency_reference_ms, config.weights, horizon=config.horizon_seconds,
        seed=config.random_seed, arrival_process=config.arrival_process,
        drain_seconds=config.drain_seconds, energy_reference_watts=config.energy_reference_watts,
        startup_amortization_seconds=config.startup_amortization_seconds)
    per_task = {}
    quality = measured["quality_evidence"]["per_task"]
    for task in config.workload:
        rows = [r for r in raw["request_records"] if r["task"] == task]
        service = [r["worker_latency_ms"] for r in rows]
        e2e = [(r["completion_time_ns"] - r["scheduled_time_ns"]) / 1e6 for r in rows]
        residual = [latency - svc for latency, svc in zip(e2e, service)]
        baseline = [_baseline(config.catalog, raw["replica_to_family"][r["model"]], task,
                    x[raw["replica_to_family"][r["model"]]]) for r in rows]
        task_weight = config.workload[task] / math.fsum(config.workload.values())
        native_metric = 1 - quality[task]["loss"] if task in {"text", "image"} else quality[task]["loss"]
        service_stats, e2e_stats = distribution(service), distribution(e2e)
        proxy_stats = predicted.get("per_task", {}).get(task, {})
        per_task[task] = {"requests": len(rows), "service": service_stats, "end_to_end": e2e_stats,
            "worker_queue_wait": distribution([r["worker_queue_wait_ms"] for r in rows
                                                if r.get("worker_queue_wait_ms") is not None]),
            "waiting_plus_transport": distribution(residual),
            "quality": {"metric": METRICS[task], "value": native_metric,
                        "loss": quality[task]["loss"], **quality[task]},
            "uncalibrated_baseline_service": distribution(baseline),
            "baseline_service_mean_error": error(mean(baseline), mean(service)),
            "baseline_service_p95_error": error(percentile(baseline, 95), percentile(service, 95)),
            "baseline_e2e_mean_error": error(proxy_stats.get("mean_ms"), e2e_stats["mean_ms"]),
            "baseline_e2e_p95_error": error(proxy_stats.get("p95_ms"), e2e_stats["p95_ms"]),
            "L_cost": config.weights["L"] * task_weight * e2e_stats["mean_ms"] / config.latency_reference_ms[task],
            "P_cost": config.weights["P"] * task_weight * e2e_stats["p95_ms"] / config.latency_reference_ms[task],
            "L_waiting_plus_transport_cost": config.weights["L"] * task_weight * mean(residual) / config.latency_reference_ms[task],
            "P_uplift_over_service_only_cost": config.weights["P"] * task_weight
                * (e2e_stats["p95_ms"] - service_stats["p95_ms"]) / config.latency_reference_ms[task]}
    energy = {k: v for k, v in raw["energy_evidence"].items() if k != "samples"}
    energy["formal_window_average_board_watts"] = energy["run_joules"] / raw["measurement_window"]["duration_seconds"]
    memory = {k: v for k, v in raw["memory_evidence"].items() if k != "samples"}
    key_fields = ["U", "eligible", "utility_complete", "diagnostic_partial_U", "reason", "safe_memory_mib",
                  "max_rho", "offered_requests", "within_window_requests", "end_backlog_requests"]
    key_fields += [field for term in "QTLPME" for field in (term + "_norm", term + "_contribution")]
    roles = {config.catalog["models"][m]["role"] for m in x}
    return {"id": raw["experiment_job_id"], "labels": job["labels"], "phase": job["phase"],
        "deployment_kind": "pure_generalist" if roles == {"generalist"} else "pure_specialist" if roles == {"specialist"} else "mixed",
        "source": provenance, "strict_native_import": "PASS", "plan": raw["plan"],
        "executed_measurement_id": raw["executed_measurement_id"],
        "context": {"workload": config.workload, "weights": config.weights,
                    "latency_reference_ms": config.latency_reference_ms, "seed": config.random_seed,
                    "horizon_seconds": config.horizon_seconds, "arrival_process": config.arrival_process,
                    "gpu": raw["preflight"]["gpu"], "gpu_memory_budget_mib": config.gpu_memory_mib,
                    "energy_reference_watts": config.energy_reference_watts,
                    "startup_amortization_seconds": config.startup_amortization_seconds,
                    "input_pool_sha256": config.catalog["quality"]["identity"]["input_pool_sha256"]},
        "measured_utility": measured, "cohort": raw["system_metrics"], "per_task": per_task,
        "energy": energy, "memory": memory,
        "uncalibrated_proxy": {"summary": {k: predicted.get(k) for k in key_fields},
            "scope": "Original standalone task-replica means, constant-service FIFO and additive memory; E and total U remain UNKNOWN",
            "errors": {term: error(predicted.get(term + "_norm"), measured.get(term + "_norm"))
                       for term in "QTLPME"}, "total_U_error": error(predicted.get("U"), measured["U"])},
        "latency_decomposition": {"L_waiting_plus_transport_cost": math.fsum(p["L_waiting_plus_transport_cost"] for p in per_task.values()),
            "P_uplift_over_service_only_cost": math.fsum(p["P_uplift_over_service_only_cost"] for p in per_task.values()),
            "interpretation": "L residual is an exact mean decomposition. P is empirical marginal-quantile uplift, not p95(queue) or a causal isolated rerun."}}


def markdown(report):
    rows = report["observations"]
    lines = ["# 静态校准条件的只读观察", "", "所有结果通过严格原生导入复核；本报告不拟合代理，不选择全池最优。", "",
        "当前只比较已完成条件；效用越大越好，六项都是加权损失，Q 是质量损失而不是准确率。", ""]
    if rows:
        c = rows[0]["context"]
        lines.extend([f'共同条件：H={c["horizon_seconds"]} 秒、seed={c["seed"]}、{c["arrival_process"]} 到达；任务速率 {json.dumps(c["workload"], ensure_ascii=False)}；六项权重 {json.dumps(c["weights"], ensure_ascii=False)}。', "",
            f'GPU：{c["gpu"]["name"]}，可用预算 {c["gpu_memory_budget_mib"]} MiB；能耗参考 {c["energy_reference_watts"]} W，启动能耗摊销 {c["startup_amortization_seconds"]} 秒。', ""])
    lines.extend([
        "| 条件 | 实测 U | Q | T | L | P | M | E |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    )
    for row in rows:
        utility = row["measured_utility"]
        lines.append("| " + " | ".join([row["labels"][0], f'{utility["U"]:.6f}']
                     + [f'{utility[t + "_norm"]:.6f}' for t in "QTLPME"]) + " |")
    lines.extend(["", "## 各任务实测（毫秒）", "",
        "| 条件 | 任务 | 数量 | service mean / p95 | e2e mean / p95 | 等待+传输 mean | 原生质量 |",
        "|---|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        for task, stats in row["per_task"].items():
            s, e, q = stats["service"], stats["end_to_end"], stats["quality"]
            lines.append(f'| {row["labels"][0]} | {task} | {stats["requests"]} | {s["mean_ms"]:.2f} / {s["p95_ms"]:.2f} | {e["mean_ms"]:.2f} / {e["p95_ms"]:.2f} | {stats["waiting_plus_transport"]["mean_ms"]:.2f} | {q["metric"]}={q["value"]:.6f} |')
    lines.extend(["", "## 按任务分解的加权时延成本", "",
        "| 条件 | 任务 | L 成本 | P 成本 |", "|---|---|---:|---:|"])
    for row in rows:
        for task, stats in row["per_task"].items():
            lines.append(f'| {row["labels"][0]} | {task} | {stats["L_cost"]:.6f} | {stats["P_cost"]:.6f} |')
    lines.extend(["", "## 整板能耗和显存", "",
        "| 条件 | 正式运行 kJ | 启动 kJ | 正式窗口平均 W | 安全显存 MiB | 显存比例 |",
        "|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        en, mem = row["energy"], row["memory"]
        lines.append(f'| {row["labels"][0]} | {en["run_joules"] / 1000:.3f} | {en["startup_joules"] / 1000:.3f} | {en["formal_window_average_board_watts"]:.3f} | {mem["safe_memory_mib"]:.3f} | {row["measured_utility"]["M_norm"]:.6f} |')
    lines.extend(["", "## 原 standalone proxy 同条件误差", "",
        "以下预测未使用这些条件的实测服务或能耗。E 未知，所以总 U 和总 U 误差保持未知。", "",
        "| 条件 | 项 | proxy | 实测 | 相对误差（绝对值） |", "|---|---|---:|---:|---:|"])
    for row in rows:
        for term, values in row["uncalibrated_proxy"]["errors"].items():
            fmt = lambda v: "UNKNOWN" if v is None else f"{v:.6f}"
            rel = "—" if values["relative_error"] is None else f'{100 * values["relative_error"]:.2f}%'
            lines.append(f'| {row["labels"][0]} | {term} | {fmt(values["predicted"])} | {fmt(values["observed"])} | {rel} |')
    if report.get("pure_role_comparison"):
        comparison = report["pure_role_comparison"]
        lines.extend(["", "## 这两个条件的差异解释", "",
            f'全专用相对纯通用的实测效用增加 {comparison["specialist_U_minus_generalist_U"]:.6f}。以下是各项对该差值的贡献，正数有利于全专用：', "",
            "| Q | T | L | P | M | E |", "|---:|---:|---:|---:|---:|---:|"])
        lines.append("| " + " | ".join(f'{comparison["contribution_delta"][term]:+.6f}' for term in "QTLPME") + " |")
        share = lambda value: "未定义（效用差为零）" if value is None else f"{100 * value:.2f}%"
        lines.extend(["", f'质量项贡献 {share(comparison["quality_gain_share"])} 的效用差，mean/p95 时延两项合计贡献 {share(comparison["latency_gain_share"])}。',
            "纯通用将四类请求放进同一 worker FIFO；长服务请求会推迟后面的短任务。全专用为不同任务使用独立 worker，保留 GPU 共享，但移除了任务间的同一 worker 队头阻塞。此观察由每任务 service 与 e2e 的差距支持；不能仅凭两点把全部差值归因为队列，因为模型自身服务速度和原生质量也不同。", ""])
        g = next(row for row in rows if row["id"] == comparison["generalist_id"])
        s = next(row for row in rows if row["id"] == comparison["specialist_id"])
        if all(s["per_task"][task]["quality"]["loss"] < stats["quality"]["loss"] for task, stats in g["per_task"].items()):
            lines.extend(["四项原生质量指标均是专用更好（EM/accuracy 更高、CER/WER 更低），满足预设质量方向。它们对当前 U 差值的贡献较小，是因为固定 reference 下的 L/P 数值远大于其他归一项；本观察不调整 reference 或权重。", ""])
        if ("text" in g["per_task"] and s["per_task"]["text"]["service"]["mean_ms"] > g["per_task"]["text"]["service"]["mean_ms"]
                and s["per_task"]["text"]["end_to_end"]["mean_ms"] < g["per_task"]["text"]["end_to_end"]["mean_ms"]):
            lines.extend(["文本专用的平均 service 比通用更慢，但平均 e2e 明显更快，说明队列与部署分离带来的收益不能仅用单模型 service 排名解释。", ""])
    lines.extend(["## 排队分解的边界", "",
        "L 的等待+传输项按每个请求 e2e-service 计算，可精确分解均值。P 的 e2e p95 与 service p95 差只表示两个边际分位数的差，不能当成队列等待的 p95。", "",
        "| 条件 | L 等待+传输加权成本 | P 相对 service-only 的加权增量 |", "|---|---:|---:|"])
    for row in rows:
        d = row["latency_decomposition"]
        lines.append(f'| {row["labels"][0]} | {d["L_waiting_plus_transport_cost"]:.6f} | {d["P_uplift_over_service_only_cost"]:.6f} |')
    lines.extend(["", "尚未完成混合部署的独立校准/验证，也未执行全池搜索、DFS 对照或正式 confirmation；不能将本报告中的较好条件称为全池最优。", ""])
    return "\n".join(lines)


def report_observations(packet_path, run_paths):
    packet_path = Path(packet_path).resolve()
    packet_sha = digest(packet_path)
    packet = json.loads(packet_path.read_text())
    observations = []
    for path in run_paths:
        path = Path(path).resolve()
        raw = json.loads(path.read_text())
        source = {"id": raw["experiment_job_id"], "path": str(path), "sha256": digest(path),
                  "packet_path": str(packet_path), "packet_sha256": packet_sha}
        observations.append(observation(source, packet))
    if len({row["executed_measurement_id"] for row in observations}) != len(observations):
        raise ValueError("Observation inputs repeat an executed measurement condition")
    report = {"schema": "gsdeploy.static_calibration_observation.v1", "fitted": False, "gpu_started": False,
        "scope": "Descriptive comparison of completed native conditions only; no all-pool winner claim",
        "execution_packet": {"path": str(packet_path), "sha256": packet_sha}, "observations": observations}
    generals = [o for o in observations if o["deployment_kind"] == "pure_generalist"]
    specialists = [o for o in observations if o["deployment_kind"] == "pure_specialist"]
    if len(generals) == len(specialists) == 1:
        g, s = generals[0], specialists[0]
        if g["context"] != s["context"]:
            raise ValueError("Pure-role observations do not share scoring/workload/hardware context")
        gu, su = g["measured_utility"], s["measured_utility"]
        delta = su["U"] - gu["U"]
        contributions = {term: su[term + "_contribution"] - gu[term + "_contribution"] for term in "QTLPME"}
        report["pure_role_comparison"] = {"generalist_id": g["id"], "specialist_id": s["id"],
            "specialist_U_minus_generalist_U": delta, "contribution_delta": contributions,
            "decomposition_residual": delta - math.fsum(contributions.values()),
            "quality_gain_share": contributions["Q"] / delta if delta else None,
            "latency_gain_share": (contributions["L"] + contributions["P"]) / delta if delta else None}
    report["sha256"] = quality_digest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-packet", type=Path, required=True,
                        help="Explicitly accept this immutable historical execution packet/runtime")
    parser.add_argument("--runs", type=Path, required=True, nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report = report_observations(args.execution_packet, args.runs)
    out = args.output_prefix
    paths = [Path(str(out) + ".json"), Path(str(out) + ".zh.md")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Use a fresh output prefix to preserve observation revisions")
    out.parent.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    paths[1].write_text(markdown(report))
    print(json.dumps({"observations": len(report["observations"]), "strict_import": "PASS",
                      "fitted": False, "gpu_started": False, "outputs": list(map(str, paths))}))


if __name__ == "__main__":
    main()
