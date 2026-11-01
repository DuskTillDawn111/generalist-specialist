"""Evidence-linked final report for this prospective study only."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from tools.queue_coupled_utility_20261023 import protocol
from tools.queue_refined_20261023.search import measured_table_search, identity


def measured_dfs(packet, execution):
    candidates = {}
    records = []
    for job in packet['jobs']:
        candidate = dict(role=job['role'], deployment=job['nominal_plan']['deployment'],
                         policy_contract_sha256=protocol.digest(job['policy_contract']))
        cid = identity(candidate)
        candidates[cid] = candidate
        path = Path(execution) / job['label'] / 'run.json'
        if path.exists():
            raw = json.loads(path.read_text())
            records.append(dict(candidate_id=cid, seed=job['seed'], utility=raw['evaluation']['utility'], raw=protocol.receipt(path)))
    table = dict(candidates=list(candidates.values()), seeds=packet['seeds'], records=records,
                 weights=packet['jobs'][0]['policy_contract']['weights'])
    dfs = measured_table_search(table, 'plain_dfs')
    bounded = measured_table_search(table, 'bounded')
    same = dfs['winner_id'] == bounded['winner_id'] and dfs['objective'] == bounded['objective']
    if not same:
        raise ValueError('Measured table DFS mismatch')
    return dict(table=table, dfs=dfs, bounded=bounded, same_winner_and_value=True,
                scope='Existing native measurement table, not a physical full-pool optimum or GPU search speedup')


def build(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = root / 'native-development-execution-v1' / 'analysis.json'
    report = json.loads(report_path.read_text())
    packet = json.loads((root / 'native-development-packet-v1' / 'experiment-packet.json').read_text())
    measured = measured_dfs(packet, report_path.parent)
    protocol.write(output / 'measured-dfs.json', measured)
    out = io.StringIO()
    fields = ['label', 'seed', 'role', 'U', 'eligible', 'request_count', *list('QTLPME')]
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for row in report['observations']:
        writer.writerow({**{k: row[k] for k in fields if k not in 'QTLPME'}, **row['terms']})
    with (output / 'native-results.csv').open('x') as stream:
        stream.write(out.getvalue())
    lines = ['# 共享 GPU 耦合路由实验结果', '',
             '本报告保留原六项等权效用、稳定性与能耗窗，使用新的受控校准与新种子；旧结果未覆盖。', '',
             '## 任务与部署', '',
             '静态 Poisson 负载：text 1.85、image/asr/ocr 各0.01请求/秒；600秒测量、30秒预热。七模型源配置池保持冻结。',
             '主驻留为两个文本专用副本、一个 MiniCPM 通用副本和一个 OCR 专用副本。背景图像/ASR走通用，OCR走OCR。',
             '四个明确对照是新耦合分流、同算法关闭分流、旧延迟分流、增加图像专用副本的另一混合部署；不是只与纯通/纯专比较。', '',
             '## 原生结果', '',
             '|运行|总效用U|有效|text→通用次数|稳定性|', '|---|---:|---|---:|---|']
    for row in report['observations']:
        value = 'UNKNOWN' if row['U'] is None else f"{row['U']:.9f}"
        lines.append(f"|{row['label']}|{value}|{row['eligible']}|{row['routing_counts'].get('text->generalist_2', 0)}|{row['stability'].get('stability_state_v2', 'UNKNOWN')}|")
    lines += ['', '逐对分流收益定义为新分流U减同算法禁止分流U：', '']
    for pair in report['paired']:
        value = 'UNKNOWN' if pair['delta_U'] is None else f"{pair['delta_U']:+.9f}"
        lines.append(f"- seed{pair['seed']}：{value}；实际分流{pair['spilled_text']}次。")
    lines += ['', '## 确认门禁与范围', '',
              f"预先声明的独立确认门禁通过：{report['confirmation_gate_passed']}。门禁要求两个开发配对完整有效、均实际分流且总效用收益均为正。",
              '未通过门禁时不启动无支持的正收益确认，不强制分流、不修改权重以制造符号。',
              'DFS与优化搜索的结果必须区分：新CPU共享事件模拟属于有限条件代理域；原生表DFS只校验已测四个候选的排序，不能把表查找时间当GPU实验效率。',
              f"本次原生表DFS与优化表搜索最优结果一致：{measured['same_winner_and_value']}。",
              '完整七模型配置池的物理全局最优仍未得到穷举证书；缺证据候选不作为不可行排除。', '',
              '在线代价是共同已观测前缀的Q/T/L/P及剩余整卡E条件代理。物理峰值M、未来到达和真实反事实服务不被假装已知；最终结论使用原生实测六项U。', '',
              '## CPU搜索与新校准', '']
    cpu_path = root / 'cpu-search-v2' / 'summary.json'
    if cpu_path.exists():
        cpu = json.loads(cpu_path.read_text())
        for name, value in cpu['studies'].items():
            lines.append(f"- {name}：DFS {value['dfs_oracle_calls']}次、优化{value['optimized_oracle_calls']}次，时间{value['dfs_seconds']:.3f}/{value['optimized_seconds']:.3f}秒，全部并列最优一致={value['same_optimum_and_all_ties']}，有限域已完整识别={value['whole_declared_cpu_domain_certified']}。")
        lines.append('这些是真实独立共享事件模拟调用；节省来自预先证明的符号等价，不是复用已算效用值。混合负载未支持的背景状态仍为UNKNOWN；text-only结果不能充当原负载最优。')
    lines += ['受控校准完成320个隔离批次/576个文本请求；另测同驻留两个60秒持续空闲阶段。新功耗模型以持续空闲证据替代短gap的高功耗，但仍不声称瞬态功率与所有反事实已精确建模。', '',
              '## 独立确认', '']
    confirmation_path = root / 'native-confirmation-execution-v1' / 'analysis.json'
    if confirmation_path.exists():
        confirmation = json.loads(confirmation_path.read_text())
        pairs = confirmation['paired']
        complete = len(pairs) == 6 and all(p['complete_eligible'] for p in pairs)
        positive = complete and all(p['delta_U'] > 0 for p in pairs)
        lines.append(f'预先保留的六配对完整={complete}，六次均正={positive}。')
        for pair in pairs:
            delta = 'UNKNOWN' if pair['delta_U'] is None else f"{pair['delta_U']:+.9f}"
            lines.append(f"- seed{pair['seed']}：ΔU={delta}，实际分流{pair['spilled_text']}次。")
    else:
        lines.append('未执行独立确认；见开发门禁或执行终止原因。不能把开发配对当独立确认。')
    lines += ['',
              '## 数据', '',
              f'- 原生分析：`{report_path}`',
              f'- 新执行包：`{root / "native-development-packet-v1/experiment-packet.json"}`',
              f'- CSV：`{output / "native-results.csv"}`',
              f'- 已测表DFS：`{output / "measured-dfs.json"}`']
    with (output / '实验报告.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    protocol.write(output / 'index.json', dict(report=protocol.receipt(output / '实验报告.md'),
                                               csv=protocol.receipt(output / 'native-results.csv'),
                                               analysis=protocol.receipt(report_path),
                                               confirmation_gate_passed=report['confirmation_gate_passed']))
    return output / '实验报告.md'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(build(a.root, a.output))
