"""Regenerate the date-grouped index of repository documentation."""

from collections import defaultdict
from pathlib import Path
import re


def main():
    docs = Path(__file__).resolve().parents[1] / "docs"
    groups = defaultdict(list)
    for path in sorted(docs.glob("*")):
        if path.suffix not in {".md", ".json"} or path.name in {"README.md", "index.md"}:
            continue
        dates = re.findall(r"(?<!\d)(20\d{6})(?!\d)", path.stem)
        groups[dates[-1] if dates else "规范与未标日期文档"].append(path.name)
    lines = ["# 完整文档索引", "", "由 `python scripts/update_docs_index.py` 生成；主题入口见[文档导航](README.md)。", ""]
    keys = ["规范与未标日期文档"] + sorted(k for k in groups if k != "规范与未标日期文档")
    for key in keys:
        if key not in groups:
            continue
        heading = f"{key[:4]}-{key[4:6]}-{key[6:]}" if key.isdigit() else key
        lines.extend([f"## {heading}", ""])
        lines.extend(f"- [{name}]({name})" for name in groups[key])
        lines.append("")
    lines.extend(["## 历史源码快照", "", "- [早期优化器副本](snapshots/optimizer_20261023.py.txt)", ""])
    (docs / "index.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
