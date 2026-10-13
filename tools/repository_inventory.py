"""Build repository navigation without importing research code or launching workers.

Inventory Git-tracked and nonignored source files, including executable code in
results/. Read recorded gates as evidence, without interpreting them as approval
to run experiments. Only docs/experiment_index.md and docs/repository_inventory.json
are written by --write.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("docs/experiment_index.md", "docs/repository_inventory.json")
SNAPSHOT_PARTS = {"source_snapshot", "method_v1_source"}
FOUNDATIONS = {
    "benchmark": "数据准备、质量评测与性能采样",
    "deployment_core": "部署数据结构、MILP/LP 与路由求解",
    "deployment_search": "部署/路由搜索、反馈与运行时适配",
    "performance_model": "性能模型接口、特征、采样与校准实验",
    "joint_deploy": "模型 worker、进程通信与测量底座",
    "placement": "早期放置实现与历史原型",
    "tools": "仓库维护工具",
}
PURPOSES = {
    "method_v1": "冻结原整数搜索的兼容入口",
    "method_v2": "针对性替换与类别多样化选择",
    "candidate_pool_audit": "候选模型登记、下载、准入与冻结",
    "candidate_pool_service_demand": "候选服务路径的 standalone 测量",
    "candidate_pool_v2_1_adapters": "候选池视觉问答扩展适配",
    "replica_scaling": "独立模型副本的扩展测量",
    "replica_capacity_calibration": "经验副本容量及显存规划输入",
    "integer_replica_pilot": "整数副本真机搜索试验",
    "integer_replica_multiworkload": "整数副本多负载试验",
    "quality_workload_audit": "质量、工作负载与模型来源审计",
    "data_quality_audit": "数据质量审计与重新分析",
    "module20": "CPU 效用候选生成与 FIFO 排序",
    "module21": "Poisson 排队效用与冻结真机验证",
    "module24": "效用回放、偏差与校准审计",
    "module25": "路由条件下的服务分布与效用预测",
    "module30": "分流可行性及非阻塞遥测验证",
    "module31a": "Qwen/Whisper 同版本合法对照",
    "module28_dynamic_split": "独立在线动态路由",
    "module32b_vqa_aware_routing": "保护 VQA 的 ASR 动态接纳策略",
    "module33_mixed_oracle": "Mixed Oracle 有限域与算法比较",
    "module34_candidate_pool": "嵌套候选池扩展与误差归因",
    "module35_utility_routing": "U 对齐 bundle/Pareto 动态规划与路由搜索",
}


def candidates() -> list[str]:
    run = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    )
    return sorted({p for p in run.stdout.decode().split("\0") if p})


def is_source(path: str) -> bool:
    return path.endswith((".py", ".sh", ".py.txt"))


def own_source(path: str) -> bool:
    return not SNAPSHOT_PARTS.intersection(Path(path).parts)


def syntax(path: str):
    if not path.endswith(".py"):
        return None
    try:
        return ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    except (SyntaxError, UnicodeError) as error:
        return {"error": type(error).__name__, "line": getattr(error, "lineno", None)}


def has_cli(tree) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        names = [n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)]
        values = [n.value for n in ast.walk(node.test) if isinstance(n, ast.Constant)]
        if "__name__" in names and "__main__" in values:
            return True
    return False


def imports(tree, local_roots: set[str]) -> list[str]:
    found = set()
    if not isinstance(tree, ast.Module):
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            found.add(node.module.split(".")[0])
    return sorted(found & local_roots)


def gate_record(path: str) -> dict | None:
    try:
        value = json.loads((ROOT / path).read_text())
    except (ValueError, UnicodeError):
        return {"path": path, "recorded_value": "无法解析"}
    # Lists of protected files and gate schemas are provenance, not verdicts.
    if not isinstance(value, dict):
        return None
    if any(word in Path(path).name.lower() for word in ("schema", "definition", "policy", "history", "historical")):
        return None
    # Preserve recorded field names/booleans, never synthesize a PASS/FAIL.
    if isinstance(value, dict):
        detail = value.get("gate", value.get("gates", value.get("flags")))
        if detail is None:
            detail = {k: v for k, v in value.items() if isinstance(v, bool)}
        if not detail:
            return None
    return {"path": path, "recorded_value": detail}


def purpose(directory: str, paths: list[str], trees: dict) -> str:
    name = Path(directory).name
    if name in FOUNDATIONS:
        return FOUNDATIONS[name]
    if name in PURPOSES:
        return PURPOSES[name]
    # Result directory names preserve the original stage's own terminology.
    if directory.startswith("results/"):
        return name.replace("_", " ")
    for path in paths:
        tree = trees[path]
        if isinstance(tree, ast.Module):
            doc = ast.get_docstring(tree)
            if doc:
                return doc.splitlines()[0][:120]
    return name.replace("_", " ") + "；阶段实验，主线身份未指定"


def build() -> tuple[str, str]:
    files = candidates()
    source_paths = [p for p in files if is_source(p) and (ROOT / p).is_file()]
    own = [p for p in source_paths if own_source(p)]
    trees = {p: syntax(p) for p in own}
    local_roots = {Path(p).parts[0] for p in own if len(Path(p).parts) > 1}
    local_roots.update(Path(p).stem for p in own if len(Path(p).parts) == 1)
    groups = {}
    for p in own:
        parts = Path(p).parts
        if len(parts) == 1:
            key = "."
        elif parts[0] == "results":
            key = "/".join(parts[:2])
        else:
            key = parts[0]
        groups.setdefault(key, []).append(p)
    gate_paths = [p for p in files if p.startswith("results/") and
                  p.endswith(".json") and "gate" in Path(p).name.lower() and
                  len(Path(p).parts) == 3]
    gates = {p: record for p in gate_paths if (record := gate_record(p)) is not None}
    result_dirs = sorted({"/".join(Path(p).parts[:2]) for p in files
                          if p.startswith("results/") and len(Path(p).parts) > 2})
    result_names = {Path(d).name: d for d in result_dirs}
    result_set = set(result_dirs)
    components = []
    for directory, paths in sorted(groups.items()):
        references = set()
        dependencies = set()
        for p in paths:
            tree = trees[p]
            dependencies.update(imports(tree, local_roots))
            if not isinstance(tree, ast.Module):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value in result_names:
                        references.add(result_names[node.value])
                    if node.value.startswith("results/"):
                        result_dir = "/".join(node.value.split("/")[:2])
                        if result_dir in result_set:
                            references.add(result_dir)
        if directory.startswith("results/"):
            references.add(directory)
        else:
            references.update(d for d in result_dirs if Path(d).name.startswith(directory + "_"))
        own_gates = [v for p, v in gates.items() if str(Path(p).parent) in references]
        components.append({
            "directory": directory, "purpose": purpose(directory, paths, trees),
            "role": "基础框架" if directory in FOUNDATIONS else
                    "冻结兼容入口" if directory in {"method_v1", "method_v2"} else
                    "结果目录内实现" if directory.startswith("results/") else "研究阶段实现",
            "source_count": len(paths),
            "entrypoints": sorted([p for p in paths if has_cli(trees[p])], key=lambda p: (
                "test" in Path(p).name or "tests" in Path(p).parts,
                Path(p).name not in {"campaign.py", "run.py", "seed.py", "main.py", "pipeline.py"},
                len(Path(p).parts), p)),
            "local_import_roots": sorted(dependencies - {directory}),
            "referenced_result_directories": sorted(references),
            "gate_records": own_gates,
        })
    sources = []
    for p in source_paths:
        raw = (ROOT / p).read_bytes()
        sources.append({"path": p, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    parse_notes = [{"path": p, **tree} for p, tree in trees.items() if isinstance(tree, dict)]
    inventory = {
        "schema_version": 1,
        "scope": "Git tracked and nonignored source; historical snapshots included in source hashes",
        "source_files": len(sources), "source_bytes": sum(s["bytes"] for s in sources),
        "historical_snapshot_sources": len(source_paths) - len(own),
        "source_type_counts": dict(sorted(Counter(Path(p).suffix for p in source_paths).items())),
        "components": components, "sources": sources,
        "syntax_notes": parse_notes, "recorded_gates": list(gates.values()),
        "result_directories_with_versioned_material": result_dirs,
        "local_only_assets": ["models/", "datasets/", "envs/", "hf_cache/", "pip_cache/",
                              "conda_pkgs/", "large raw runs, trajectories, journals, telemetry and tables"],
        "full_gpu_reproduction_available_from_git_alone": False,
    }
    lines = ["# 实验与源码索引", "",
             "由 `python tools/repository_inventory.py --write` 从 Git 可跟踪源码静态生成。", "",
             "入口列是发现的命令行候选，未导入模块或执行实验。依赖列是静态 import，动态加载及冻结文件依赖仍需阅读源码/manifest。", "",
             "Gate 原样来自已跟踪元数据；没有 Gate 的目录记为未汇总，不推断其通过、失败、归档或正式身份。编号较新不代表晋升为正式算法。", "",
             f"共 {len(sources)} 个源码/脚本文件，其中 {len(source_paths)-len(own)} 个位于历史源码副本目录。", ""]
    for heading, predicate in [
        ("基础框架与根目录脚本", lambda c: c["directory"] in FOUNDATIONS or c["directory"] == "."),
        ("根目录研究阶段与冻结入口", lambda c: c["directory"] not in FOUNDATIONS and c["directory"] != "." and not c["directory"].startswith("results/")),
        ("结果目录中的独立实现", lambda c: c["directory"].startswith("results/")),
    ]:
        lines += ["## " + heading, "", "| 源码位置 | 用途/身份 | 候选入口 | 本地依赖 |", "| --- | --- | --- | --- |"]
        for c in components:
            if not predicate(c):
                continue
            d = c["directory"]
            label = "根目录脚本" if d == "." else d
            target = ".." if d == "." else "../" + d
            entries = ", ".join(f"[{Path(p).name}](../{p})" for p in c["entrypoints"][:4]) or "API/worker；参见源码"
            dependencies = ", ".join("`" + p + "`" for p in c["local_import_roots"][:6]) or "—"
            if len(c["local_import_roots"]) > 6:
                dependencies += " 等（完整列表见 JSON）"
            description = (c["purpose"] + " / " + c["role"]).replace("|", "\\|")
            lines.append(f"| [{label}]({target}) | {description} | {entries} | {dependencies} |")
        lines.append("")
    lines += ["## 已记录的 Gate", "", "以下内容只转录结论字段；细节与支持范围以原文件和阶段摘要为准。", "",
              "| 原文件 | 记录值 |", "| --- | --- |"]
    for p, g in sorted(gates.items()):
        value = json.dumps(g["recorded_value"], ensure_ascii=False, sort_keys=True).replace("|", "\\|")
        if len(value) > 700:
            value = value[:700] + "…（完整字段见原文件）"
        lines.append(f"| [{p}](../{p}) | `{value}` |")
    lines += ["", "## 其他结果目录与数据依赖", "",
              "机器清单的 `result_directories_with_versioned_material` 覆盖无独立源码但有版本化材料的目录。", "",
              "`sources` 保存每个源码文件的 SHA256；历史源码副本只计入哈希清单，不当作新增独立实验。", "",
              "Git 未上传完整原始数据。目录与 manifest 存在，不代表其引用的轨迹、权重或记录全部可用；见 [复现说明](reproduction.md)。", ""]
    return "\n".join(lines), json.dumps(inventory, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Update the two docs inventory files")
    mode.add_argument("--check", action="store_true", help="Check generated docs without writing")
    args = parser.parse_args()
    contents = build()
    stale = []
    for relative, content in zip(OUTPUTS, contents):
        path = ROOT / relative
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        elif not path.exists() or path.read_text(encoding="utf-8") != content:
            stale.append(relative)
    if args.write:
        print("Updated " + ", ".join(OUTPUTS))
    elif stale:
        print("Inventory out of date: " + ", ".join(stale))
        print("Run python tools/repository_inventory.py --write")
        return 1
    else:
        print("Repository inventory is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
