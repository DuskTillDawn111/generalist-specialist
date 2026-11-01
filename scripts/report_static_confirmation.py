"""Verify native confirmation runs and summarize paired seed-level differences."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsdeploy.confirmation import verify_confirmation_run, validate_confirmation_packet
from gsdeploy.evidence import digest
from gsdeploy.io import save
from gsdeploy.static_study import read
from gsdeploy.study_reporting import measured_summary, paired_comparisons


def report_confirmation(packet_path, runs_root, output, *, explicit_runs=None):
    packet_path = Path(packet_path).resolve()
    packet = read(packet_path)
    check = validate_confirmation_packet(packet)
    if not check["valid"]:
        raise ValueError("Changed confirmation packet: " + repr(check["changed"]))
    jobs = {job["id"]: job for job in packet["jobs"]}
    sources = list(map(Path, explicit_runs)) if explicit_runs else sorted(Path(runs_root).rglob("run.json"))
    rows, ignored, seen = [], [], set()
    for path in sources:
        value = read(path)
        job_id = value.get("experiment_job_id")
        if job_id not in jobs:
            ignored.append(str(path.resolve()))
            continue
        if job_id in seen:
            raise ValueError("Multiple runs for one job; explicitly select one attempt: " + job_id)
        raw, utility = verify_confirmation_run(packet, jobs[job_id], path)
        seen.add(job_id)
        rows.append({"job_id": job_id, "plan_key": jobs[job_id]["candidate_key"],
                     "labels": jobs[job_id]["labels"], "seed": jobs[job_id]["seed"],
                     "path": str(path.resolve()), "sha256": digest(path), "utility": utility,
                     "offered_requests": raw["measurement_window"]["offered_requests"],
                     "errors": raw.get("errors", [])})
    selected = next(role["plan_key"] for role in packet["candidate_roles"] if role["label"] == packet["winner_label"])
    paired = paired_comparisons(rows, selected, packet["confirmation"]["seeds"],
                                planned_plan_keys=[c["plan_key"] for c in packet["candidates"]])
    result = {"schema": "gsdeploy.independent_confirmation_report.v1",
              "packet": {"path": str(packet_path), "sha256": digest(packet_path)},
              "status": "ALL_FROZEN_JOBS_VERIFIED" if len(seen) == len(jobs) else "INCOMPLETE_CONFIRMATION",
              "completed_jobs": len(seen), "planned_jobs": len(jobs),
              "missing_job_ids": sorted(set(jobs) - seen), "ignored_other_jobs": ignored,
              "runs": measured_summary(rows, packet["config"]["weights"]), "paired": paired,
              "candidate_roles": packet["candidate_roles"], "gpu_started": False,
              "physical_global_optimality_certified": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "report.json", result)
    labels = {c["plan_key"]: ", ".join(c["labels"]) for c in packet["candidates"]}
    lines = ["# 独立 GPU 配对确认", "", f"状态：{result['status']}；有效作业 {len(seen)}/{len(jobs)}。", "",
             "总效用越大越好；下表差值为预先选定方案减去竞争方案。每个种子是一组重复，不能把请求数作为独立重复数。", "",
             "| 竞争方案 | 完成种子 | 平均效用差 | 95% Student 区间 | 精确双侧符号翻转 p |",
             "|---|---|---:|---|---:|"]
    for pair in paired["comparisons"]:
        estimate = pair["mean_U_difference"]
        interval = pair.get("student_interval")
        lines.append("| " + " | ".join((labels[pair["rival_key"]], str(pair["paired_seeds"]),
            f"{estimate:.6f}" if estimate is not None else "未完成",
            f"[{interval[0]:.6f}, {interval[1]:.6f}]" if interval else "未完成",
            str(pair.get("exact_two_sided_sign_flip_p", "未完成")))) + " |")
    lines.extend(["", "Student 区间依赖独立且近似正态的种子差值假设。三个种子的精确双侧符号翻转检验最小 p=0.25，不能据此声称达到 0.05 显著性。多竞争方案的 Holm 校正值保存在 JSON。", "",
                  "该报告只评价已冻结并实际测量的方案、输入和种子；有限域代理模型证书与物理全组合最优分别报告。"])
    (output / "report.zh.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report_confirmation(args.packet, args.runs_root, args.output, explicit_runs=args.runs)
    print(json.dumps({key: result[key] for key in ("status", "completed_jobs", "planned_jobs", "gpu_started")}))


if __name__ == "__main__":
    main()
