"""Summarize final evidence without changing frozen methods or measurements."""
from .common import *


def main():
    gate = read(OUT / 'module34_gate.json')
    contrasts = read(OUT / 'gpu_contrasts.json')
    expansion = read(OUT / 'gpu_pool_expansion_pair.json')
    decisions = read(OUT / 'optimizer_decisions.json')
    timings = list(csv.DictReader((OUT / 'objective_and_search_error.csv').open()))
    assert gate['completed'] and gate['GPU']['observed_runs'] == 24
    def worst(method):
        return max(r['regret'] for r in decisions if r['algorithm'] == method and r['regret'] is not None)
    def speed(method):
        v = [float(r['oracle_over_solver_speedup']) for r in timings
             if r['pool'] == 'C' and r['algorithm'] == method]
        return f'{min(v):.1f}–{max(v):.1f}'
    lines = []
    for c in contrasts:
        ss = c['delta_statistics']; lo, hi = ss['ci95']
        lines.append(f"| {c['group'].replace('TEXT_HALF_MID18', 'TEXT10')} | {ss['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {c['verdict']} | {c.get('holm_adjusted_p', 1):.6g} |")
    exp = expansion['statistics']
    text = f'''# Module 34 最终结论与论文证据边界

已完成三个嵌套池、三个负载的九组 CPU 实验，以及原冻结清单的全部 **24 个 GPU 窗口**（6 个开发窗口、18 个独立配对窗口）。GPU 有效窗口为 {gate['GPU']['valid_runs']}/24。本轮方法、配置和请求轨迹均未因 GPU 结果而调整。历史 SHA256 与修改时间核验通过，文件数 {gate['history']['files']}。

## 六个研究问题的回答

1. **扩张时能否维持或改变正确决策？** 两个域内方法的预测 U 在 A→B→C 中均未下降，结构随候选可用性从 A 的纯 G 变为 B/C 的 Mixed。20A 九组的驻留模型与副本均匹配 U Oracle，但仅 5/9 完整路由达到预测最优；四个路由搜索缺口保留。V3 网格控制 6/9 达到 U 最优；原生 V3 仅 6/9 属于本有限域，其余两组域不同、一组原生数值失败，不能替换成控制版成功结果。

2. **纯 G 与 Mixed 是否各自竞争获胜？** 尚未建立双向交叉胜出的证据。A 只允许 Generalist，其纯 G 最优由候选域限制决定；B/C 的 CPU 参考最优为 Mixed。同 TEXT10 独立请求下，B 所选 Mixed 相对 A 所选纯 G 的探索性平均差为 {exp['mean']:+.6f} U，95% t 区间为 {exp['ci95']}，两方案均合法于 B；该比较不是完整 B 池 GPU Oracle，也不是经过三项主比较多重校正的第四个成功检验。纯 S 因缺少合法 VQA 路径不可行。

3. **相对启发式的优势是否得到证明？** A TEXT10 中，V3/CPU Oracle 的队列敏感路由与 20A/最快路径启发式可通过下表独立配对判断。20A 在该例与最快路径方案相同，不能将其优势归给 20A。C 的最小显存启发式与 20A/Oracle 选出相同方案，所以没有证明算法对所有合理启发式普遍占优。B/C 的结果按区间和 Holm 校正判断，负结果保留。

4. **有限 U Oracle regret 多大？** V3 网格控制最大预测 regret={worst('V3_J_GRID_CONTROL'):.6f}；20A 网格最大={worst('20A_U_GRID'):.6f}，C 三负载均为 0。C 池 8,154 个经过显存/覆盖筛选的配置中，20A 相对枚举快 {speed('20A_U_GRID')} 倍，V3 网格控制快 {speed('V3_J_GRID_CONTROL')} 倍。它们是 CPU 后端计时，不代表在线 GPU 搜索端到端加速。实测差距仅覆盖冻结的关键方案集合，不是全池 GPU regret。

5. **是否支撑论文主实验？** 已有受限空间正确性、扩张选择、求解效率、关键方案独立实测，以及目标/预测/路由搜索失效证据，可用于明确限定范围的论文主实验。尚不支持“原生 V3/20A 普遍达到真实全局最优”“三池同负载完整实测扩张曲线”或“纯 G/Mixed 双向竞争交叉胜出”。GPU_ORACLE_INCOMPLETE=true；不得由完整测完六个预选方案改称全池 Oracle。

6. **10 月 16 日前最关键的补测是什么？** 若论文主图要求 A→B→C 同负载扩张的实测证据，优先补 C 池 TEXT10 已冻结方法所选方案的独立轨迹，连接现有 A/B TEXT10 证据。只有论文保留纯 G/Mixed 双向交叉主张时，才再冻结两结构同时合法的竞争场景。若以 U 方法作为主算法，20A 的队列敏感候选生成缺口必须以独立命名版本及新 holdout 验证；本轮不修改算法重验收。跨零小差距可报告无法区分，无需为追求显著而无限追加。10 月 16 日后保留论文写作时间。

## 独立 GPU 配对结果

ΔU=CPU Oracle 所选方案−冻结对手。每项 3 个独立种子；开发窗口不混入区间。区间为 95% Student t 区间，n=3 的统计功效与近似正态假设限制保留；明确优势还需考虑三项预注册比较的 Holm 校正。

| 冻结比较 | 平均 ΔU | 95% CI | 判定 | Holm p |
| --- | --- | --- | --- | --- |
{chr(10).join(lines)}

原始 V3 与同网格控制结果独立保留；J/U 的 Oracle 差异同时包含目标定义与性能代理差异。20A 四个预测缺口来自候选路由生成；真实观测偏差另来自服务画像、并存干扰、Poisson 到达与周期 FIFO 预测等适用性差异，不能全部归为搜索错误。

详见 `module34_summary.md`、`module34_gate.json`、`continuation_scientific_audit.json` 和 `final_audit.json`。项目产物保存在 `/root/autodl-tmp/generalist-specialist/results/module34_candidate_pool/`。Module 34 完成后停止，未自动进入 Module 35。
'''
    (OUT / 'module34_final_answers.md').write_text(text)
    print('FINAL_ANSWERS_WRITTEN', flush=True)


if __name__ == '__main__':
    main()
