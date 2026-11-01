"""Read-only Chinese study report; never runs inference or imports GPU runtime.

Only terminal raw paths registered in progress.json are read. Output must be a
new directory outside the running workflow and all other input directories.
Missing evidence stays missing; diagnostic_U is never substituted for native U.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

TERMS = "QTLPME"
VERIFIED = ("cohort_verified", "constraints_verified", "quality_identity_verified",
            "memory_verified", "energy_verified")
PHASES = (("development", "开发筛选", (113, 114)),
          ("independent-validation", "独立验证", (81, 82)),
          ("final-confirmation", "最终确认", (71, 72, 73)))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def candidate(contract):
    return dict(deployment=contract["deployment"], strategy=contract["strategy"],
                policy_contract_sha256=digest(contract))


def deployment_text(value):
    return ", ".join(f"{k}×{v}" for k, v in sorted(value.items())) or "缺失"


def number(value):
    return f"{value:.8g}" if finite(value) else "未完成/不可评价"


def table(headers, rows):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(escape, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(escape, row)) + " |" for row in rows)
    return "\n".join(lines) + "\n"


class Reader:
    """Bind exactly the bytes used, including a point-in-time progress snapshot."""
    def __init__(self):
        self.sources = {}

    def read(self, path, optional=False):
        path = Path(path).resolve()
        if optional and not path.exists():
            return None
        content = path.read_bytes()
        self.sources[str(path)] = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                                      bytes=len(content))
        return json.loads(content)


def native_observation(raw, path):
    """Read producer-audited native utility; fail closed on unsupported claims."""
    contract = raw["policy_contract"]
    symbolic = candidate(contract)
    utility = raw.get("evaluation", {}).get("utility", {})
    weights = raw.get("config", {}).get("weights", {})
    terms = {t: utility.get(t + "_norm") for t in TERMS}
    verified = all(utility.get(k) is True for k in VERIFIED)
    arithmetic = (set(weights) == set(TERMS)
                  and all(finite(weights[t]) and weights[t] >= 0 for t in TERMS)
                  and all(finite(terms[t]) and terms[t] >= 0 for t in TERMS)
                  and finite(utility.get("U")))
    if arithmetic:
        arithmetic = math.isclose(utility["U"], -math.fsum(weights[t] * terms[t] for t in TERMS),
                                  rel_tol=0, abs_tol=1e-10)
    eligible = (raw.get("backend") == "native" and raw.get("status") == "completed"
                and utility.get("eligible") is True and utility.get("utility_complete") is True
                and verified and arithmetic)
    counts = Counter()
    alias_counts = Counter()
    summaries = defaultdict(lambda: dict(service=[], queue=[]))
    spill_examples = []
    text_spill_inputs = Counter()
    text_inputs = Counter()
    aliases = raw.get("replica_to_family", {})
    for row in raw.get("request_records", []):
        alias = row.get("model", "UNKNOWN")
        family = aliases.get(alias, "UNKNOWN")
        counts[(row["task"], family)] += 1
        if row["task"] == "text":
            pool_id = row.get("pool_id", "UNKNOWN")
            text_inputs[pool_id] += 1
            if family.startswith("generalist"):
                text_spill_inputs[pool_id] += 1
        alias_counts[alias] += 1
        for name, field in (("service", "worker_latency_ms"), ("queue", "worker_queue_wait_ms")):
            if finite(row.get(field)):
                summaries[alias][name].append(row[field])
        decision = row.get("routing_decision", {}).get("decision", {})
        if row["task"] == "text" and family.startswith("generalist") and len(spill_examples) < 3:
            audit = decision.get("audit", [])
            spill_examples.append(dict(request_id=row.get("request_id"), alias=alias,
                audit=[{k: c.get(k) for k in ("alias", "family", "score", "predicted_waiting_ms",
                                            "predicted_own_service_ms", "predicted_completion_ms")}
                       for c in audit]))
    routing = defaultdict(dict)
    for (task, family), count in sorted(counts.items()):
        routing[task][family] = count
    summary = {}
    for alias, values in summaries.items():
        summary[alias] = dict(requests=alias_counts[alias],
            mean_service_ms=statistics.mean(values["service"]) if values["service"] else None,
            mean_worker_queue_wait_ms=statistics.mean(values["queue"]) if values["queue"] else None,
            max_worker_queue_wait_ms=max(values["queue"]) if values["queue"] else None)
    return dict(candidate_id=digest(symbolic), candidate=symbolic,
        seed=raw.get("transport_config", {}).get("random_seed", raw.get("config", {}).get("random_seed")),
        phase=raw.get("experiment_phase"), status=raw.get("status"), eligible=bool(eligible),
        U=utility.get("U") if eligible else None, terms=terms,
        contributions={t: -weights[t] * terms[t] if finite(terms[t]) and finite(weights.get(t)) else None for t in TERMS},
        routing_counts=dict(routing), alias_counts=dict(alias_counts), alias_summaries=summary,
        text_unique_inputs=len(text_inputs), text_spill_unique_inputs=len(text_spill_inputs),
        text_spill_input_counts=dict(text_spill_inputs),
        spill_examples=spill_examples, stability=raw.get("evaluation", {}).get("stability"),
        quality_evidence=utility.get("quality_evidence", {}),
        error=raw.get("errors"), hardware=raw.get("preflight", {}).get("gpu"), raw=str(Path(path).resolve()),
        evidence_scope="Native producer evaluation with required verification flags and arithmetic checked; no independent inference rerun.")


def rank_candidates(declared, observations, seeds):
    groups = defaultdict(dict)
    for row in observations:
        if row["seed"] not in seeds:
            raise ValueError("Native observation outside declared phase seeds")
        if row["seed"] in groups[row["candidate_id"]]:
            raise ValueError("Duplicate native candidate/seed")
        groups[row["candidate_id"]][row["seed"]] = row
    if set(groups) - set(declared):
        raise ValueError("Native observation outside declared candidate domain")
    ranking = []
    for cid, symbolic in declared.items():
        rows = groups.get(cid, {})
        complete = all(seed in rows and rows[seed]["eligible"] for seed in seeds)
        ranking.append(dict(candidate_id=cid, candidate=symbolic, complete=complete,
            completed_runs=len(rows), eligible_runs=sum(r["eligible"] for r in rows.values()),
            mean_U=statistics.mean(rows[s]["U"] for s in seeds) if complete else None,
            mean_contributions={t: statistics.mean(rows[s]["contributions"][t] for s in seeds)
                                if complete else None for t in TERMS},
            per_seed_U={str(s): rows[s]["U"] if s in rows else None for s in seeds}))
    ranking.sort(key=lambda r: (r["mean_U"] is None, -r["mean_U"] if r["mean_U"] is not None else 0,
                                r["candidate_id"]))
    if ranking and ranking[0]["mean_U"] is not None:
        best = ranking[0]["mean_U"]
        ties = [r for r in ranking if r["mean_U"] is not None and abs(r["mean_U"] - best) <= 1e-12]
        ranking = sorted(ties, key=lambda r: r["candidate_id"]) + [r for r in ranking if r not in ties]
    return ranking


def paired(observations, winner_id, other_id, seeds):
    by = {(r["candidate_id"], r["seed"]): r for r in observations}
    rows = []
    for seed in seeds:
        a, b = by.get((winner_id, seed)), by.get((other_id, seed))
        known = bool(a and b and a["eligible"] and b["eligible"])
        rows.append(dict(seed=seed, delta_selected_minus_control=a["U"] - b["U"] if known else None,
            contribution_deltas={t: a["contributions"][t] - b["contributions"][t] if known else None for t in TERMS}))
    values = [r["delta_selected_minus_control"] for r in rows if finite(r["delta_selected_minus_control"])]
    return dict(per_seed=rows, complete=len(values) == len(seeds), n_whole_pairs=len(values),
        mean=statistics.mean(values) if len(values) == len(seeds) else None,
        positive=sum(v > 0 for v in values), negative=sum(v < 0 for v in values), zero=sum(v == 0 for v in values),
        minimum=min(values) if values else None, maximum=max(values) if values else None,
        leave_one_out_means=[statistics.mean(values[:i] + values[i + 1:]) for i in range(len(values))]
                            if len(values) > 1 and len(values) == len(seeds) else [],
        interpretation=("尚未收齐全部预定整轮配对，不能用部分均值宣称优势。" if len(values) != len(seeds)
                        else "存在收益反转或持平，不能宣称稳定优势。" if any(v <= 0 for v in values)
                        else "全部已收齐整轮配对方向为正；样本量小且硬件漂移独立性未建立，不据此宣称统计显著优势。"))


def load_study(workflow, protocol_path, search_path, static_audit=None, quality_audit=None):
    reader = Reader()
    workflow, search_path = Path(workflow).resolve(), Path(search_path).resolve()
    protocol = reader.read(protocol_path)
    progress = reader.read(workflow / "progress.json")
    if progress.get("bindings"):
        bound_protocol = next((r for r in progress["bindings"]
                               if Path(r["path"]).resolve() == Path(protocol_path).resolve()), None)
        if bound_protocol is None or bound_protocol["sha256"] != reader.sources[str(Path(protocol_path).resolve())]["sha256"]:
            raise ValueError("Protocol is not the immutable declaration bound to this workflow")
    result = reader.read(workflow / "result.json", optional=True)
    packet_path = protocol.get("development", {}).get("packet", {}).get("path")
    if packet_path is None:
        packet_path = progress["bindings"][0]["path"]
    packet = reader.read(packet_path)
    expected_packet = protocol.get("development", {}).get("packet", {}).get("sha256")
    if expected_packet and reader.sources[str(Path(packet_path).resolve())]["sha256"] != expected_packet:
        raise ValueError("Development packet differs from declared receipt")
    selection_packet = reader.read(workflow / "frozen-validation-packet.json", optional=True)
    declared = {digest(candidate(j["policy_contract"])): candidate(j["policy_contract"]) for j in packet["jobs"]}
    selected = (selection_packet or {}).get("selection", {}).get("candidates", [])
    selected_id = selected[0]["candidate_id"] if selected else None
    control_id = selected[1]["candidate_id"] if len(selected) > 1 else None
    stages = {}
    all_observations = []
    for label, title, seeds in PHASES:
        phase_state = next((s for s in progress.get("stages", []) if s["label"] == label), {})
        observations = []
        for row in phase_state.get("jobs", []):
            path = Path(row["raw"]["path"])
            raw = reader.read(path)
            expected = row["raw"].get("sha256")
            if expected and reader.sources[str(path.resolve())]["sha256"] != expected:
                raise ValueError("Registered terminal raw changed: " + str(path))
            observations.append(native_observation(raw, path))
        phase_declared = declared if label == "development" else {r["candidate_id"]: r["candidate"] for r in selected}
        report = reader.read(workflow / label / "analysis/report.json", optional=True)
        dfs = reader.read(workflow / label / "analysis/measured-dfs-comparison.json", optional=True)
        stages[label] = dict(title=title, seeds=list(seeds), status=phase_state.get("status", "NOT_STARTED"),
            observations=observations, ranking=rank_candidates(phase_declared, observations, seeds),
            producer_report_present=report is not None, producer_dfs=dfs,
            pair=paired(observations, selected_id, control_id, seeds) if selected_id and control_id else None)
        all_observations.extend(observations)
    combined_report = reader.read(workflow / "all-independent-pairs-analysis/report.json", optional=True)
    diagnostic = {name: reader.read(search_path / file, optional=True) for name, file in
                  (("plain_dfs", "plain-dfs/report.json"), ("bounded", "bounded/report.json"),
                   ("audit", "full-domain-audit.json"), ("domain", "plain-dfs/domain.json"),
                   ("controls", "structure-controls-inclusive.json"), ("top", "top-controls.json"))}
    historical = dict(status="NOT_REQUESTED", path=None)
    if static_audit is not None:
        static_audit = Path(static_audit).resolve()
        history_progress = reader.read(static_audit.parent.parent / "progress.json", optional=True)
        historical = dict(status="WAITING_TERMINAL_LINK_ONLY", path=str(static_audit))
        # Never inspect a half-run static ranking. The terminal supervisor must
        # have declared collection and auditing complete before loading it.
        if (history_progress and history_progress.get("status", "").startswith("COMPLETE")
                and static_audit.exists()):
            historical.update(status="TERMINAL_HISTORICAL_DEVELOPMENT", report=reader.read(static_audit))
    quality_history = reader.read(quality_audit) if quality_audit is not None else None
    if quality_history is not None:
        if (quality_history.get("schema") != "gsdeploy.static_quality_equality_posthoc_audit.v1"
                or quality_history.get("historical_results_modified") is not False
                or quality_history.get("GPU_used") is not False):
            raise ValueError("Unexpected posthoc quality audit contract")
    combined = paired([r for r in all_observations if r["seed"] in (81, 82, 71, 72, 73)],
                      selected_id, control_id, (81, 82, 71, 72, 73)) if selected_id and control_id else None
    complete = (progress.get("status") == "COMPLETE" and result is not None
                and all(stages[label]["producer_report_present"] for label, _, _ in PHASES)
                and all(stages[label]["status"].startswith("COMPLETE") for label, _, _ in PHASES))
    prelaunch = not progress.get("stages") and not progress.get("current")
    status = ("FINAL" if complete else "IN_PROGRESS" if progress.get("status") == "RUNNING"
              else "PRELAUNCH_STOPPED" if prelaunch else "INCOMPLETE_STOPPED")
    return dict(schema="gsdeploy.queue_aware_chinese_report.v1", generated_utc=datetime.now(timezone.utc).isoformat(),
        status=status, registered_terminal_native_runs=len(all_observations),
        workflow_status=progress.get("status"), workflow_phase=progress.get("phase"), error=progress.get("error"),
        workflow=str(workflow), protocol=protocol, declared_candidates=declared, stages=stages,
        selected=selected, combined_pairs=combined, combined_producer_report_present=combined_report is not None,
        diagnostic=diagnostic, excluded_controls=packet.get("excluded_controls"),
        static_history=historical,
        static_quality_audit=quality_history,
        selection_frozen=selection_packet is not None, physical_full_pool_optimality=False,
        inputs=list(reader.sources.values()))


def render(study):
    p = study["protocol"]
    lines = ["# 静态负载下通用/专用联合部署与路由研究", "",
             f"报告状态：**{study['status']}**；执行状态：{study['workflow_status']}；阶段：{study['workflow_phase']}。",
             f"生成时间：{study['generated_utc']}。这是读取已登记终端文件的快照；未完成、失败和不可评价项均保留。", ""]
    lines += [f"本执行已登记原生终端轮数：{study['registered_terminal_native_runs']}；预定开发14轮、独立验证4轮、最终确认6轮，共24轮。", ""]
    if study.get("error"):
        lines += ["停止原因：" + str(study["error"]), ""]
    if study["status"] == "PRELAUNCH_STOPPED":
        lines += ["本执行尚无已登记GPU原生终态，属于启动前停止/诊断；这不是已运行GPU实验的失败结果。其他执行根中的历史批次不归入本轮。", ""]
    c = p["contracts"]
    lines += ["## 负载、目标与结论范围", "",
        "固定到达率：" + "，".join(f"{t}={v} requests/s" for t, v in c["workload_rps"].items()) + "。",
        f"每轮测量{c['horizon_seconds']}秒，预热{c['warmup_seconds']}秒；rho上限{c['rho_max']}，GPU预算{c['gpu_budget_mib']} MiB。",
        "总池包含generalist_1、generalist_2、generalist_3及text/image/ocr/asr四类专用模型。局部诊断域只覆盖G2及四专用模型；G1/G3为UNKNOWN，不能按不可行剔除。",
        f"本轮原生实测声明{len(study['declared_candidates'])}个部署/策略组合；开发种子113/114。只有全部预定种子均可评价的组合才参与正式均值排名。",
        "U = −(wQ·Q_norm + wT·T_norm + wL·L_norm + wP·P_norm + wM·M_norm + wE·E_norm)，越大越好。",
        "Q为质量损失，质量通过总效用参与权衡；取消绝对质量准入，证据身份及稳定性/资源约束仍需通过。",
        "权重：" + ", ".join(f"{t}={number(c['utility_weights'][t])}" for t in TERMS) + "。",
        "不将diagnostic_U替代原生U；有限实测表冠军与冻结诊断模型最优分别报告，均不等于七池全部部署的物理最优。", "",
        "历史text-n2保留画像来自物理GPU UUID GPU-533a287b-0e15-a7c8-8e58-2b21bdcbb13a；当前实验预定固定GPU UUID "
        + p.get("execution", {}).get("expected_gpu_uuid", "未提供") + "。两者均为RTX4090、49140 MiB、driver 595.71.05，但不是同一物理GPU。",
        "历史standalone容量proxy采用跨卡迁移口径，不构成当前卡的物理容量证书；本轮有效性依赖本轮native资源证据与原稳定性判定。迁移限制不因同型号而消失。", "",
        "text→G2先验分流保护：`" + json.dumps(p.get("guard"), ensure_ascii=False, sort_keys=True) + "`。保护限制潜在暴露，实际分流由在线可见排队与局部质量/时延代价决定。", "",
        "## 七方案开发排名及次优对照", ""]
    development = study["stages"]["development"]
    winner = next((r for r in development["ranking"] if r["complete"]), None)
    rows = []
    for i, row in enumerate(development["ranking"]):
        gap = winner["mean_U"] - row["mean_U"] if winner and row["complete"] else None
        rows.append([str(i + 1) if row["complete"] else "待定", row["candidate_id"][:12],
            deployment_text(row["candidate"]["deployment"]), row["candidate"]["strategy"],
            f"{row['completed_runs']}/{len(development['seeds'])}", number(row["mean_U"]), number(gap),
            *[number(row["mean_contributions"][t]) for t in TERMS]])
    lines += [table(["排名", "候选ID", "部署副本", "路由策略", "终端轮数", "原生mean U", "距冠军差距",
                     *[f"−w{t}·{t}_norm" for t in TERMS]], rows),
        "每项贡献为负成本；六项相加为U。缺测不会被填成0；未收齐方案不使用部分均值排名。"]
    if winner:
        if all(r["completed_runs"] == len(development["seeds"]) for r in development["ranking"]):
            lines += [f"声明实测表的当前可评价冠军：{winner['candidate_id'][:12]}；不可评价候选保留，不外推其物理最优性。"]
        else:
            lines += [f"已收齐候选中的暂时最佳：{winner['candidate_id'][:12]}；其他候选尚未收齐，不是完整七方案结论。"]
    if study["selected"]:
        lines += ["开发后冻结的验证候选：" + "；".join(
            f"{r['candidate_id'][:12]}（{deployment_text(r['candidate']['deployment'])} / {r['candidate']['strategy']}）"
            for r in study["selected"]) + "。验证与确认后不重新选择。"]
    else:
        lines += ["开发选择尚未冻结，独立验证与最终确认不得据此提前消费或选优。"]
    lines += ["", "### 强纯专与其他混合部署", ""]
    for category, test in (("强纯专", lambda d: bool(d) and all(not x.startswith("generalist") for x in d)),
                           ("其他混合部署", lambda d: any(x.startswith("generalist") for x in d)
                            and any(not x.startswith("generalist") for x in d)
                            and (winner is None or d != winner["candidate"]["deployment"]))):
        controls = [r for r in development["ranking"] if test(r["candidate"]["deployment"])]
        for control in controls:
            lines += [f"{category}：{control['candidate_id'][:12]}，{deployment_text(control['candidate']['deployment'])}，"
                      f"mean U={number(control['mean_U'])}。"]
            if winner and control["candidate_id"] != winner["candidate_id"]:
                comparison = paired(development["observations"], winner["candidate_id"], control["candidate_id"], development["seeds"])
                lines += ["冠军减该对照的整轮ΔU：" + ", ".join(f"seed{r['seed']}={number(r['delta_selected_minus_control'])}" for r in comparison["per_seed"]) + "。" + comparison["interpretation"]]
        if not controls:
            lines += [category + "尚无已声明实测候选。"]
    lines += ["", "## 原生逐轮路由、效用与排队", ""]
    for label, title, _ in PHASES:
        stage = study["stages"][label]
        lines += [f"### {title}（{stage['status']}；seeds={stage['seeds']}）", ""]
        if not stage["observations"]:
            lines += ["尚无已登记终端原生观测。", ""]
        for row in stage["observations"]:
            lines += [f"候选{row['candidate_id'][:12]}，seed{row['seed']}：{row['status']}，eligible={row['eligible']}，U={number(row['U'])}。",
                "实际task→family请求计数：`" + json.dumps(row["routing_counts"], ensure_ascii=False, sort_keys=True) + "`。",
                "实际副本请求计数：`" + json.dumps(row["alias_counts"], ensure_ascii=False, sort_keys=True) + "`。",
                f"本轮text输入覆盖{row['text_unique_inputs']}种；分流给通用的text覆盖{row['text_spill_unique_inputs']}种；实际分流pool计数：`"
                + json.dumps(row["text_spill_input_counts"], ensure_ascii=False, sort_keys=True) + "`。",
                table(["项", "原始归一化成本", "效用贡献"], [[t, number(row["terms"][t]), number(row["contributions"][t])] for t in TERMS]),
                table(["副本", "请求数", "服务均值ms", "worker排队均值ms", "worker排队最大ms"],
                    [[alias, v["requests"], number(v["mean_service_ms"]), number(v["mean_worker_queue_wait_ms"]),
                      number(v["max_worker_queue_wait_ms"])] for alias, v in sorted(row["alias_summaries"].items())]),
                "稳定性：`" + json.dumps(row["stability"], ensure_ascii=False, sort_keys=True) + "`。"]
            quality = row["quality_evidence"].get("per_task", {})
            if quality:
                lines += [table(["任务", "质量损失", "text EM / image accuracy", "评分请求数"],
                    [[task, number(q.get("loss")), number(1-q["loss"]) if task in ("text", "image") and finite(q.get("loss")) else "见CER/WER损失",
                      q.get("requests")] for task, q in sorted(quality.items())]),
                    "text/image损失分别为1−EM与1−accuracy；OCR为CER，ASR为WER，不将字符/词错误率强行标为准确率。Q按声明任务到达率加权。"]
            lines += ["本轮原始GPU证据：`" + json.dumps(row["hardware"], ensure_ascii=False, sort_keys=True) + "`。"]
            for example in row["spill_examples"]:
                lines += ["text分流审计示例：`" + json.dumps(example, ensure_ascii=False, sort_keys=True) + "`。"]
            if row.get("error"):
                lines += ["原始错误：`" + json.dumps(row["error"], ensure_ascii=False) + "`。"]
            lines += ["原始证据：[run.json](" + row["raw"] + ")。", ""]
        if stage["producer_dfs"]:
            evidence = stage["producer_dfs"]
            lines += ["原生有限表DFS一致性：" + str(evidence.get("same_winner_and_value")) + "。"]
            lines += [table(["算法", "winner", "U", "oracle调用", "查表时间s"],
                [[name, str(evidence[name].get("winner_id")), number(evidence[name].get("objective")),
                  evidence[name].get("oracle_calls"), number(evidence[name].get("elapsed_seconds"))]
                 for name in ("plain_dfs", "bounded") if name in evidence]),
                "这些oracle调用读取已保存实测表；查表时间不代表GPU推理、补测成本或测量效率优势。", ""]
    lines += ["## 冻结选择的独立验证与最终确认", "",
              "以下ΔU统一为开发冻结冠军减冻结对照；正值才表示冠军较好。独立验证81/82与最终确认71–73分别报告；五对合并只作额外描述，不是第三次独立验证。"]
    for title, value in (("独立验证81/82", study["stages"]["independent-validation"]["pair"]),
                         ("最终确认71/72/73", study["stages"]["final-confirmation"]["pair"]),
                         ("全部五对合并", study["combined_pairs"])):
        lines += ["", "### " + title, ""]
        if value is None:
            lines += ["尚未冻结对照或未开始。"]
            continue
        lines += [table(["seed", "ΔU selected−control", *[f"Δ贡献{t}" for t in TERMS]],
            [[r["seed"], number(r["delta_selected_minus_control"]),
              *[number(r["contribution_deltas"][t]) for t in TERMS]] for r in value["per_seed"]]),
            f"已观察整轮配对={value['n_whole_pairs']}；正式mean ΔU={number(value['mean'])}；"
            f"正/负/零={value['positive']}/{value['negative']}/{value['zero']}。",
            "逐次留一均值：" + (", ".join(number(v) for v in value["leave_one_out_means"]) or "不可评价") + "。",
            value["interpretation"]]
    lines += ["", "请求数不是独立重复次数；不同seed仍不能证明硬件/服务漂移独立。这里没有建立显著性检验的独立性前提，不宣称统计显著。", "",
              "## 搜索最优性与效率证据", ""]
    d = study["diagnostic"]
    dfs, bounded = d["plain_dfs"], d["bounded"]
    if dfs and bounded:
        agrees = (dfs.get("winner_id") == bounded.get("winner_id") and dfs.get("objective") == bounded.get("objective"))
        lines += [f"冻结开发诊断模型：DFS与剪枝算法winner和目标一致={agrees}；diagnostic objective={number(bounded.get('objective'))}。formal native U仍为未建立。",
            table(["算法", "候选数", "访问叶", "oracle调用", "剪枝候选", "搜索s", "冷启动端到端s"],
                [[name, r.get("candidates"), r.get("visited_leaves"), r.get("oracle_calls"), r.get("bound_pruned"),
                  number(r.get("elapsed_seconds")), number(r.get("cold_end_to_end_seconds"))]
                 for name, r in (("plain DFS", dfs), ("bounded", bounded))])]
        if dfs.get("oracle_calls") and finite(bounded.get("oracle_calls")):
            lines += [f"oracle调用减少比例={number(1-bounded['oracle_calls']/dfs['oracle_calls'])}。这是冻结模型的评估次数优势。"]
        if d["top"] and d["top"].get("top10"):
            item = d["top"]["top10"][0]
            lines += ["诊断冠军部署与路由：`" + json.dumps(item["candidate"], ensure_ascii=False, sort_keys=True) + "`。"]
    else:
        lines += ["诊断DFS/剪枝完整报告尚未提供。"]
    if d["audit"]:
        audit = d["audit"]
        lines += ["全域边界审计：`" + json.dumps({k: audit.get(k) for k in (
            "total_unique_candidate_ids", "all_candidate_seed_evaluations", "bound_leaf_checks", "violations",
            "winner_agrees", "cpu_timing_exclusive")}, ensure_ascii=False) + "`。"]
    if d["controls"]:
        lines += ["诊断模型的结构对照（不作原生实测结论）：", "",
            table(["结构", "部署副本", "策略", "诊断状态", "诊断目标（非native U）"],
                  [[name, deployment_text(r["candidate"]["deployment"]), r["candidate"].get("strategy", "static"),
                    r.get("status"), number(r.get("objective"))]
                   for name, r in d["controls"].get("controls", {}).items()])]
    lines += ["诊断域为40个局部部署、7508个静态路由加120个因果策略候选，共7628个。G1/G3未覆盖；静态路由离散化与副本上限限制了结论范围。",
        "CPU搜索计时不是独占硬件实验，受同时运行的主机工作影响，倍速仅作描述；不能外推为GPU测量加速。",
        "原生GPU测量成本为实际已执行整轮数及每轮时长，未对全部候选重复GPU推理。所有失败及不可评价终态保留。",
        "纯通对照与未覆盖候选的排除原因（若有）保留：`" + json.dumps(study["excluded_controls"], ensure_ascii=False) + "`。", "",
        "## 独立历史静态分流证据", "",
        "历史固定1/32分流与不分流批次作为独立开发证据；不并入新113/114筛选，不并入81/82或71–73验证。"]
    history = study["static_history"]
    if history["status"] == "TERMINAL_HISTORICAL_DEVELOPMENT":
        lines += ["历史批次终端监督器已完成；以下保留原audit状态与效用，不重新选择新流程的方案。",
            table(["历史label", "seed", "audit状态", "U", "eligible"],
                  [[r.get("label"), r.get("seed"), r.get("audit_status"), number(r.get("U")),
                    r.get("utility", {}).get("eligible")] for r in history["report"].get("observations", [])])]
    elif history.get("path"):
        lines += ["历史批次未确认完整终态；不读取半程排名。终态audit路径：[report.json](" + history["path"] + ")。"]
    else:
        lines += ["未指定历史终态audit；不推断其结果。"]
    lines += ["", "### 固定比例路由与输入循环相锁的质量限制", "",
        "固定1/32的周期SWRR与重复输入池可能相锁，实际分流输入子集不一定代表全池。新工作流的fixed_1of32_swrr对照也需保留这一限制。",
        "队列感知路由依据可见工作选择执行路径，不采用固定1/32周期，但仍可能产生输入选择偏差或相同Q；其质量效应必须按实际输出评分，不能从历史聚合精度直接代入。"]
    quality_audit = study["static_quality_audit"]
    if quality_audit:
        lines += ["已完成历史静态seed109–111的只读事后复算：", "",
            table(["历史seed", "rank1实际Q", "rank2实际Q", "text分流数", "分流独立输入数", "分流EM改变数"],
                [[r["seed"], number(r["observations"][0]["saved_Q_norm"]), number(r["observations"][1]["saved_Q_norm"]),
                  r["spill_request_count"], r["spill_unique_inputs"], r["spill_score_changes"]]
                 for r in quality_audit["per_seed"]]),
            f"独立匹配并重评分{quality_audit['per_request_scored_count']}对请求，保存Q均可复现。Q相等来自实际输出在正式指标上的逐请求得分相等，未使用aggregate Q代替实测。",
            "历史text池32条；分流步长31/33交替形成64周期，仅覆盖每seed输入循环的第10/11位置，各重复分流2条输入。输出字符串变化不必导致text EM变化；正式指标不在观察结果后更换。"]
        pool = quality_audit["historical_fullpool"]
        lines += [f"同池历史画像重评分：专用EM={number(pool['specialist_EM'])}，G2 EM={number(pool['generalist_EM'])}；"
            f"{pool['same_correct']}条均正确、{pool['same_wrong']}条均错误、{pool['specialist_only_correct']}条只有专用正确、{pool['generalist_only_correct']}条只有G2正确。四条EM不同的输入均未进入上述实际分流子集。",
            "历史全池差异不是未执行路由的新原生反事实观测；实际各pair的Q贡献差为0仍原样保留，不能强加聚合画像差值。",
            "旧静态评价与新动态评价均通过实际worker_response预测对参考答案评分；动态策略仅使用冻结聚合质量先验，不读取逐输入真值。CPU内存输出扰动敏感性测试已验证新模块导出的评分函数随实际输出改变。该扰动不是GPU实验结果。"]
    else:
        lines += ["未提供独立质量复算artifact；不据此推断任何历史分流样本的实际质量差值。"]
    lines += ["",
        "## 可复查证据与当前结论", "",
        "report.json保存完整结构化结果，manifest.json保存本次读取文件的SHA256。运行中的progress属于生成时刻快照；报告不会修改原实验、重新挑选对照或触发GPU任务。",
        "当前证据支持范围：声明局部诊断域中的模型最优性、声明原生候选表中的效用比较，以及冻结选择后的整轮验证。七池物理全局最优尚未证明。"]
    return "\n\n".join(lines) + "\n"


def make_plots(study, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    complete = [r for r in study["stages"]["development"]["ranking"] if r["complete"]]
    if complete:
        fig, ax = plt.subplots(figsize=(9, max(3, len(complete) * .55)))
        ax.barh([r["candidate_id"][:8] for r in complete], [r["mean_U"] for r in complete])
        ax.invert_yaxis(); ax.set_xlabel("Native mean utility U (higher is better)")
        ax.set_title("Declared native candidates: complete development seeds only")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            name = "native-development-utility." + ext
            fig.savefig(output / name, dpi=180); files.append(name)
        plt.close(fig)
    combined = study["combined_pairs"]
    if combined:
        values = [r for r in combined["per_seed"] if finite(r["delta_selected_minus_control"])]
        if values:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.bar([str(r["seed"]) for r in values], [r["delta_selected_minus_control"] for r in values],
                   color=["#237a57" if r["delta_selected_minus_control"] > 0 else "#bd4545" for r in values])
            ax.axhline(0, color="black", linewidth=.8); ax.set_xlabel("Whole-run seed (81/82 validation; 71-73 confirmation)")
            ax.set_ylabel("U selected - U frozen control"); fig.tight_layout()
            for ext in ("png", "pdf"):
                name = "frozen-selection-whole-pairs." + ext
                fig.savefig(output / name, dpi=180); files.append(name)
            plt.close(fig)
    return files


def generate(workflow, protocol, search, output, *, plots=False, static_audit=None, quality_audit=None):
    output = Path(output).resolve()
    roots = (Path(workflow).resolve(), Path(search).resolve(), Path(protocol).resolve().parent)
    if any(output == root or root in output.parents for root in roots):
        raise ValueError("Report output must be outside input artifact directories")
    study = load_study(workflow, protocol, search, static_audit=static_audit, quality_audit=quality_audit)
    output.mkdir(parents=True, exist_ok=False)
    def write(path, value):
        with path.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    write(output / "report.json", study)
    (output / "研究报告.md").write_text(render(study))
    plots_written = make_plots(study, output) if plots else []
    write(output / "manifest.json", dict(schema="gsdeploy.queue_aware_report_manifest.v1",
        status=study["status"], generated_utc=study["generated_utc"], inputs=study["inputs"],
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), plots=plots_written,
        outputs=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.iterdir()) if p.is_file()]))
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workflow", "protocol", "search", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="Export standard matplotlib PNG and PDF figures")
    parser.add_argument("--static-audit", type=Path, help="Optional historical terminal audit, gated by completed parent progress")
    parser.add_argument("--quality-audit", type=Path, help="Optional independent posthoc static quality-equality audit")
    args = parser.parse_args()
    result = generate(args.workflow, args.protocol, args.search, args.output, plots=args.plots,
                      static_audit=args.static_audit, quality_audit=args.quality_audit)
    print(json.dumps(dict(status=result["status"], output=str(args.output.resolve())), ensure_ascii=False))


if __name__ == "__main__":
    main()
