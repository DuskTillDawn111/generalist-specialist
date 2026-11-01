"""Assemble the completed, independently audited coupled-study delivery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


TERMS = "QTLPME"
NAMES = {
    "coupled_spill": "新耦合效用分流",
    "matched_no_spill": "同算法、同驻留禁止分流",
    "legacy_latency": "同驻留旧延迟分流",
    "other_mixed": "增加图像专用的旧 refined 混合方案",
}


def read(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def link(path, label):
    return f"[{label}]({Path(path).resolve()})"


def build(root):
    root = Path(root).resolve()
    report = root / "final-report-v1"
    audit_dir = root / "native-final-independent-audit-v1"
    audit_path = audit_dir / "audit.json"
    delivered_audit_path = audit_dir / "delivery-audit.json"
    endpoint_audit_path = audit_dir / "endpoint-delivery-audit.json"
    endpoint_path = root / "native-endpoint-diagnostics-v1/diagnostics.json"
    packet_path = root / "native-development-packet-v1/experiment-packet.json"
    analysis_path = root / "native-development-execution-v1/analysis.json"
    terminal_path = analysis_path.with_name("terminal.json")
    cpu_path = root / "cpu-search-v2/summary.json"
    weight_bound_path = root / "native-weight-necessary-condition-v1.json"
    weight_bound = read(weight_bound_path)
    endpoint_audit = read(endpoint_audit_path)
    audit, delivered, endpoint, packet, analysis, terminal, cpu = map(
        read, (audit_path, delivered_audit_path, endpoint_path, packet_path,
               analysis_path, terminal_path, cpu_path))
    if not audit["verified"] or not delivered["verified"] or not endpoint_audit["verified"]:
        raise ValueError("Final native and published-results independent audits must pass first")
    if terminal["status"] != "COMPLETED" or len(terminal["completed"]) != 8:
        raise ValueError("The prospective eight-run development execution must be complete")
    if endpoint["measured_runs"] != 8 or len(endpoint["ranking"]) != 4:
        raise ValueError("Endpoint diagnostics are incomplete")
    if analysis["confirmation_gate_passed"] != audit["confirmation_gate_passed"]:
        raise ValueError("Confirmation gate disagreement")
    for label, run in endpoint["runs"].items():
        if not math.isclose(run["utility"]["U"], audit["measured"][label]["U"],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Endpoint utility disagrees with independent audit: {label}")
    base = read(packet["base_packet"]["path"])
    models = base["config"]["catalog"]["models"]
    lines = ["# 共享 GPU 耦合效用：本阶段综合结论", "",
             "八次预声明 GPU 开发运行全部完成，原始结果、六项效用、路由、稳定性和来源保护已独立复核。"
             "本报告区分已完成的实验结果、有限域搜索验证与尚未证实的最终研究目标。", "",
             "## 任务负载与模型池", "",
             "固定强度 Poisson 到达：text=1.85、image/asr/ocr 各0.01请求/秒。"
             "每轮30秒预热、600秒测量、最多300秒排空；seed303正序、seed304反序。"
             "率保持静态，具体到达和队列随时间变化。",
             "原 Q/T/L/P/M/E 六项等权，U 为负加权成本，越大越好。Q 使用实测输出误差，未恢复绝对质量准入。"
             "ρ≤0.8及原稳定性准入保持不变。E继续按原测量窗加启动摊销计分；整批排空能耗另列诊断，不替换E。", "",
             "|池内槽位|冻结模型|任务能力|", "|---|---|---|"]
    for slot, model in models.items():
        lines.append(f"|{slot}|{model['name']}|{', '.join(model['supported_tasks'])}|")
    lines += ["", "七模型源池保持冻结；本阶段原生比较的是以下四个明确的部署/策略候选，未穷举七模型全部部署组合。", "",
              "|候选|实际驻留|路由定义与比较意义|", "|---|---|---|",
              "|新耦合效用分流|text×2 + G2×1 + OCR×1|text按共同已观测前缀的条件效用选择两专用或G2；背景image/asr→G2，ocr→OCR|",
              "|同算法禁止分流|text×2 + G2×1 + OCR×1|保持新算法与驻留，仅禁止text→G2；用于分流消融|",
              "|旧延迟分流|text×2 + G2×1 + OCR×1|旧queue_latency路由；是完整策略对照|",
              "|另一混合方案|text×2 + G2×1 + OCR×1 + image×1|旧refinement/global_incremental，image→图像专用；同时改变驻留、背景路由和策略|", "",
              "G2即MiniCPM-o-4_5-GPTQ，text即Llama3-ChatQA-1.5-8B。"
              "完整混合负载CPU尚未识别最优，因此新原生策略采用事先声明的默认值，不能写成CPU已选出的物理最优。"
              "默认新策略使用mean+p90共同情景、margin=0、30秒漂移冷却、text/8分流令牌速率和32个pending请求的预测预算；"
              "超预算跳过预测并回退专用，未截断完整队列。", "",
              "## 八次实际测量", "",
              "下表准确率仅为文本任务；总效用仍包含全部四种任务。两专用分别为S0/S1。", "",
              "|候选|seed|U|text数|S0 / S1 / G2|text准确率|均值ms|P95 ms|",
              "|---|---:|---:|---:|---|---:|---:|---:|"]
    for role in NAMES:
        for seed in packet["seeds"]:
            run = endpoint["runs"][f"{role}-seed{seed}"]
            u = run["utility"]
            task = run["per_task_metrics"]["text"]
            counts = task["actual_routed_request_count"]
            accuracy = 1-u["quality_evidence"]["per_task"]["text"]["loss"]
            lines.append(f"|{NAMES[role]}|{seed}|{u['U']:.9f}|{task['offered_requests']}|"
                         f"{counts.get('text#0',0)} / {counts.get('text#1',0)} / {counts.get('generalist_2#0',0)}|"
                         f"{accuracy:.3%}|{task['mean_e2e_latency_ms']:.3f}|{task['p95_e2e_latency_ms']:.3f}|")
    states = sorted({r["stability"]["stability_state_v2"] for r in endpoint["runs"].values()})
    lines += ["", f"八次eligible均为true；实际稳定性状态集合为{states}。"
              "每个seed的四候选使用相同offered到达/输入cohort，且预热与测量均经因果回放。"
              "相同输入不表示两轮具有相同运行历史，不能把同ID请求的时延差当同状态物理反事实。", "",
              "|两seed均值排名|候选|平均U|", "|---:|---|---:|"]
    for rank, row in enumerate(endpoint["ranking"], 1):
        lines.append(f"|{rank}|{NAMES[row['role']]}|{row['mean_U']:.9f}|")
    winner = endpoint["ranking"][0]
    lines += ["", f"已测四候选按两seed平均U的第一名为「{NAMES[winner['role']]}」。"
              "此结论是开发数据上的有限候选排序，不是新独立确认，也不是七模型物理全局最优证书。", "",
              "## 新策略的分流消融", "",
              "收益定义为新分流U减同算法、同驻留禁分流U。下表为已乘等权的收益贡献，正数改善U，六项之和等于ΔU。", "",
              "|seed|ΔU|Q|T|L|P|M|E|", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pair in endpoint["paired"]:
        lines.append("|"+"|".join([str(pair["seed"]),f"{pair['delta_U']:+.9f}",
                                  *[f"{pair['gain_by_term'][t]:+.9f}" for t in TERMS]])+"|")
    mean_gain = statistics.fmean(p["delta_U"] for p in endpoint["paired"])
    positives = sum(p["delta_U"]>0 for p in endpoint["paired"])
    lines += ["", f"平均配对收益{mean_gain:+.9f}，严格正收益{positives}/2。"
              f"预声明确认门禁通过={analysis['confirmation_gate_passed']}。"
              "门禁要求两个开发配对各自完整有效、各自实际分流且各自ΔU>0；首配对已不满足。"
              "因此未执行seed305–310确认，六个种子保持未暴露，未换权重、换赢家或强制分流后冒充原协议确认。", "",
              "## 自主分流机制与预测缺口", ""]
    for pair in audit["paired"]:
        lines.append(f"- seed{pair['seed']}：自主分流{pair['actual_spills']}次，G2可见空闲{pair['G2_visible_empty_count']}次；"
                     f"双专用非空且至少一路已有waiting队列{pair['both_specialists_nonempty_with_existing_waiting']}次；"
                     f"实际G2服务超过初始mean预测两倍{pair['actual_G2_service_above_two_times_mean_forecast']}次。")
    lines += ["", "新策略已能在多副本拥塞时自主识别分流机会，但不是所有分流都由拥塞触发。"
              "mean/p90条件代理均判定改善，只能说明在线机制按设计运行，不能保证实测整窗六项U改善。",
              "seed303深挖已复现18次分流预测与输入/token协议。用事后真实active-set积分可缩小一部分服务偏差，仍有7次残余超过2倍；"
              "measurement:1015预测G2服务约0.860秒，实际4.382秒，且几乎全程两专用并发。"
              "目前校准覆盖隔离同输入批次，尚未证明可迁移到异步混合输入、持续忙碌环境。"
              "16/18次分流在mean/p90两情景中的前缀P95增量均为0，未来整窗尾部风险覆盖不足。"
              "这些是证据指向的模型缺口，尚不能认定为唯一物理因果机制。",
              "另一混合方案seed303/304的13/16次分流全部发生在双专用非空、已有waiting且G2空闲时。"
              "这验证了目标队列机制存在；它没有自身同驻留禁分流对照，因此不能将其与其他候选的U差异单独归因于分流，或只归因于增加图像模型。", "",
              "## DFS与算法效率", "",
              "|证据域|候选|DFS调用|优化调用|最优是否识别|范围|", "|---|---:|---:|---:|---|---|"]
    for key, name in (("full_workload","完整混合负载条件模拟"),
                      ("text_only_mechanism_diagnostic","text-only条件模拟")):
        row = cpu["studies"][key]
        lines.append(f"|{name}|24|{row['dfs_oracle_calls']}|{row['optimized_oracle_calls']}|"
                     f"{row['optimum_identified']}|{'原负载全部UNKNOWN，无赢家' if key=='full_workload' else '全部24个候选可算，全部ties一致'}|")
    native = read(report / "measured-dfs.json")
    lines.append(f"|原生实测表检索|4|{native['dfs']['oracle_calls']}|{native['bounded']['oracle_calls']}|"
                 "已测表有限赢家|已有8次测量的查表，无新GPU推理|")
    text = cpu["studies"]["text_only_mechanism_diagnostic"]
    reduction = 1-text["optimized_oracle_calls"]/text["dfs_oracle_calls"]
    lines += ["", f"真实CPU模拟调用减少{reduction:.2%}，完全来自10个无分流策略的预先证明的符号等价合并；"
              "strict bound剪枝为0，未缓存已经计算的目标值。"
              f"text-only观察耗时{ text['dfs_seconds']:.3f}/{text['optimized_seconds']:.3f}秒，"
              f"约{text['process_local_speedup']:.4f}倍，仅为一次顺序、非隔离观察。",
              "text-only唯一数值赢家为同一主驻留、allow_spill=true、mean+p90、margin=0、cooldown=30秒，"
              f"平均条件代理U={text['exact_max_objective']:.9f}。这不是原生实测U。",
              "完整混合负载首先因背景活动功率缺证据而UNKNOWN，背景联合服务也未完全覆盖。"
              "两个空winner和空tie集合相同不构成最优验证成功。text-only唯一赢家和DFS一致仅验证有限条件数值域；"
              "M真实峰值、原稳定性、原混合任务和物理全池最优不能由此推得。",
              "原生表DFS/优化的赢家、目标值、全部ties均经独立复核；其耗时是已有数据查表，不可包装为GPU搜索提速。", "",
              "## 对最终研究目标的判断", "",
              "本阶段完成了新共享GPU模型、受控校准、有限域搜索、八次GPU开发对照与独立终审。"
              "“算法能自主分流”已有真实证据；“分流在原六项效用下稳定增益”和“七模型全池物理最优”仍未证明。",
              "此前完整六维权重探索已证实：旧205–208四对不存在同一非负权重使四对全正，不能靠降低L/P权重得到该稳定性结论。"
              "本轮不事后改目标。重新选择权重会同时改变部署与路由问题，必须作为新实验冻结并独立验证。",
              "作为事后数学诊断，新seed303只有E项收益为正，因此在非负且和为1的权重域中，"
              f"该固定轨迹若要重评分转正，P95权重必须严格小于{weight_bound['necessary_strict_P_weight_upper_bound']['decimal']:.6%}。"
              "这只是必要条件，其他损失只会使要求更严。"
              f"例如P95保留至少5%权重时，任意其他权重的ΔU上界为{weight_bound['universal_delta_U_upper_bound_at_or_above_floor']['decimal']:+.9f}，仍为负。"
              "5%仅作明示的敏感性例子，并非用户给定的业务要求；未据此选新权重、改路由或宣称独立收益。",
              "若进入下一版本，应先验证异步混合输入的持续服务/功率支持域和预测残差，再评估带不确定性保护的排队分流与未来尾部情景；"
              "另一混合驻留需补其自身禁分流消融。完整混合代理可识别后才能恢复该域自主候选搜索与非空DFS最优验证。"
              "这些是尚待执行的后续工作，没有在本阶段被宣布完成，也没有绕过本轮确认门禁。", "",
              "## 可复核材料", ""]
    references = [
        (report / "实验报告.md","原始阶段报告"),
        (report / "native-results.csv","八次六项效用CSV"),
        (endpoint_path.parent / "deployment-routing-metrics.csv","实际部署、路由、准确率、延迟与资源CSV"),
        (endpoint_path,"全部终点及六项收益分解"),
        (endpoint_path.parent / "measured-results.png","开发结果图"),
        (report / "measured-dfs.json","原生有限表DFS与全部ties"),
        (audit_path,"最终独立全表审计"),
        (delivered_audit_path,"发布CSV与DFS独立核对"),
        (endpoint_audit_path,"路由终点、收益分解与调权必要条件独立核对"),
        (cpu_path,"CPU两域搜索完整结果"),
        (weight_bound_path,"本轮固定轨迹的调权必要条件（事后诊断）"),
        (root / "seed303-resource-mismatch-diagnosis-v1/README.zh.md","服务预测失配诊断"),
        (root / "native-other-mixed-independent-audit-v1/README.zh.md","另一混合方案逐次分流队列证据"),
        (root.parent.parent / "docs/queue_refined_full_weight_exploration_20261023.md","此前完整权重探索结论"),
        (packet_path,"冻结实验包"),
    ]
    for path, label in references:
        if not path.is_file():
            raise ValueError(f"Missing delivery reference: {path}")
        lines.append(f"- {link(path,label)}")
    target = report / "综合结论.md"
    with target.open("x") as stream:
        stream.write("\n".join(lines)+"\n")
    listed = [target, Path(__file__).resolve(), terminal_path, analysis_path,
              report/"index.json", *[p for p,_ in references]]
    unique = list(dict.fromkeys(listed))
    manifest = dict(schema="gsdeploy.coupled_final_delivery.v1",
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                    status="COMPLETED_EIGHT_DEVELOPMENT_RUNS_AND_INDEPENDENT_AUDITS",
                    primary_report=receipt(target),
                    candidate_domain="Four explicitly measured deployment/policy candidates; not the seven-model full pool",
                    measured_candidate_winner=winner,
                    physical_seven_pool_optimality=False,
                    confirmation_gate_passed=analysis["confirmation_gate_passed"],
                    stable_positive_gain_certified=False,
                    development_runs=8, confirmation_runs=0,
                    artifacts=[receipt(p) for p in unique])
    manifest_path = report / "delivery-manifest.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    return dict(report=receipt(target),manifest=receipt(manifest_path),
                development_runs=8,winner=winner,paired=endpoint["paired"],
                independent_checks=len(audit["checks"]),delivery_checks=len(delivered["checks"]),
                endpoint_checks=len(endpoint_audit["checks"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.root),ensure_ascii=False,allow_nan=False))
