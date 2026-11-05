"""Render a reproduced positive case and an explicitly unvalidated design."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    args=p.parse_args();out=args.input.resolve()
    report=json.loads((out/'report.json').read_text())
    design=json.loads((out/'capacity-scenario-design.json').read_text())
    lower,upper=design['necessary_share_interval']
    first=next((dict(denominator=d,numerator=u,share=u/d)
                for d in range(1,101) for u in range(d+1) if lower<=u/d<=upper),None)
    # This is a scalar sizing calculation. The restricted predictor has not
    # been called for this changed load, including for the grid alternatives.
    grid=dict(first_rational_in_sizing_interval=first,
        base_K_le_16_has_feasible_share=False,
        existing_pair_denominator_extension_can_cover=True,
        caveat='Sizing only; targeted pair grid is a required comparator, not excluded to favour continuous search')
    assert first['denominator']==22
    (out/'grid-sizing.json').write_text(json.dumps(grid,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for seed in (42,43,45):
        rows=sorted([r for r in report['rows'] if r['seed']==seed],key=lambda r:r['specialist_share'])
        axes[0].scatter([r['specialist_share'] for r in rows],[r['U'] for r in rows],label=f'trace {seed}')
    axes[0].set(xlabel='Text share to Specialist',ylabel='U (higher is better)',
                title='Six declared ratios per trace\nLegacy predictions; isolated points')
    axes[0].legend(fontsize=8)
    q=design['proposed_OCR_qps'];bg=design['G_background_rho']
    sg=design['legacy_G_ocr_service_seconds'];ss=design['legacy_S_ocr_service_seconds']
    rs=[i/1000 for i in range(1001)]
    axes[1].plot(rs,[bg+q*(1-r)*sg for r in rs],label='G rho: assumed legacy law')
    axes[1].plot(rs,[q*r*ss for r in rs],label='S rho: assumed legacy law')
    axes[1].axhline(.8,color='black',ls='--',lw=1)
    axes[1].axvspan(lower,upper,color='green',alpha=.2,label='Sizing feasible interval')
    axes[1].set(xlim=(.9,1),ylim=(0,1.8),xlabel='OCR share to Specialist',ylabel='rho',
                title='Proposed high-load design\nNOT measured or validated')
    axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'scenario-evidence.png',dpi=160);plt.close(fig)
    rows42=[r for r in report['rows'] if r['seed']==42]
    table=['| Text给S比例 | G/S实际请求数 | U |','|---|---:|---:|']
    for r in sorted(rows42,key=lambda r:r['specialist_share']):
        table.append(f"| {r['specialist_share']:.6f} | {r['text_G_requests']}/{r['text_S_requests']} | {r['U']:.9f} |")
    text=f"""# 连续分流价值：已有正例与下一场景设计

2026-11-02。本轮只做18次完整预测bundle复核，用时{report['elapsed_seconds']:.3f}秒；没有GPU采集、画像拟合或新负载完整预测。所有实际效用结果均来自原 `calibrated_replay_v2`，不代表新阶段预测器已具备真实画像。

## 1. 已有支持域里确有非1:1、非端点的正例

固定 D={{G2:1,image:1,OCR:1,Text:1}}，四任务各0.074836 QPS，1800秒原轨迹seed42。只有Text路由改变，其他任务和部署/驻留、六项效用、约束完全相同。

{chr(10).join(table)}

Text给S 78.1180%的候选，比同部署最好的端点U=-0.343243168提高0.017870770，比1:1提高0.017648736，比该一维完整K8切片的最佳5/7提高0.003695894。131条Text请求真实映射为29条G、102条S；不是只改比例而不改分配。本轮逐请求重建FIFO，核对正式predict的任务Mean/P95、完整分母、窗口完成数和排空数。

这已证明：**在这个冻结预测模型和固定部署问题中，端点及1:1不能覆盖最好的已知路由，低K也有损失。** 它没有证明连续全域最优、真实GPU收益或整体联合部署胜出。原Joint少部署Text的U=-0.309302874仍更高，78.1180%方案尚差0.016069524，额外驻留成本没有收回。

## 2. 稳定性不能只展示最有利轨迹

78.1180%相对最佳端点在seed42/43/45的增益约为+0.017871/+0.000030/−0.001245；相对1:1在43/45均落后。另一历史比例43.6454%虽然在三条轨迹都超过端点，也在43/45落后1:1。两条检查轨迹以前已查看过，本轮是复现，不能重新称为盲测。

因此合理目标是证明“连续搜索可根据负载和轨迹选择不同非对称比例”，不是宣布某个固定78:22永久最好。未来应在训练轨迹选择路由，固定后检查未见轨迹，并同时报告逐轨迹寻优结果作为不可部署的oracle参照，避免把看过未来请求的收益当成在线策略收益。

## 3. 更有区分度的场景：双边容量约束的非对称热点

令r为任务给S的比例，q为任务到达率，b为G的其他任务压力，sG/sS为该执行配置下的服务需求，n=1、原rho上限0.8。在给定服务律下必须满足：

`b + q(1-r)sG <= 0.8`，`qr sS <= 0.8`。

因此 `max(0, 1-(0.8-b)/(q*sG)) <= r <= min(1,0.8/(q*sS))`。

若q大于任一方单独容量、但低于双方剩余容量之和，两个端点都失败；两边能力不对称时，1:1也可能失败，内部非对称比例才有可行机会。这是模型结构结论，不依赖把效用权重改成偏好分流。

用现有G2/OCR单副本服务表只做选点计算：保留Text/ASR/Image原负载，固定G2+image+OCR，G承担Text/ASR，Image给S。历史G/OCR服务约8124.51ms/435.15ms，G背景压力{bg:.6f}，剩余OCR容量约{design['G_residual_capacity_rps']:.6f}/{design['S_capacity_rps']:.6f} QPS。候选OCR负载为{q:.6f} QPS，比例区间约 **{lower:.4%}–{upper:.4%}给S**；中点约 **{design['proposed_interior_share']:.4%}给S、{1-design['proposed_interior_share']:.4%}给G**。

| OCR给S | 选点模型G rho | 选点模型S rho | rho条件 |
|---|---:|---:|---|
"""
    for key,row in design['sizing_rho_at_controls'].items():
        ok=max(row.values())<=.8
        text+=f"| {float(key):.4%} | {row['G']:.6f} | {row['S']:.6f} | {'满足' if ok else '超出'} |\n"
    text+=f"""
**这组高负载数字仅是待测选点，不是已运行的新负载预测。** 旧预测器明确拒绝 `{design['replay_gate']}`；未绕过门禁，完整FIFO、质量、M/E及最终U均未计算。高负载会改变干扰、资源和排空，历史均值外推不能当作安全容量或实际可行性证书。

按这个选点模型，当前基础K1..16都没有比例落入该区间；第一个有理候选是21/22。仓库已有按任务扩展pair denominator的能力，因此公平比较必须加上“低K+OCR定向分母22”基线。不能只与过粗的K比较后宣称连续方法唯一有效。价值应同时体现效用、求解成本和跨负载适应性。

## 4. 最小后续验证顺序

1. 先从匹配的新阶段画像确认这个部署的真实容量；必要时调整预先声明的热点负载，不能坚持使用未经支持的1.881452 QPS。
2. 限定1个部署、1个热点任务、1档负载。训练数据拟合进度和生命周期；验证用一条冻结公开轨迹分别评价G端点、1:1、S端点、旧K网格、定向细网格和连续搜索。连续候选来自训练数据，不能用验证时序回填规律。
3. 保留全部offered请求和超载/排空结果，禁止丢弃失败请求制造更低P95。记录实际分配、服务、等待、Q/T/L/P/M/E及调用数。
4. 固定部署证据成立后，比较合法替代部署和副本数：增加OCR副本可能让纯S重新占优，必须纳入同域Joint对照。仅胜同部署端点不能宣称联合方案胜出。
5. 若真实测量未支持预期，则报告失败或调整机制解释，不通过修改预测耗时/效用/约束强行制造赢家。

![已有正例与明确未验证的热点选点]({out/'scenario-evidence.png'})

原始每点预测及完整请求账本保存在本目录trace*.json，report.json保存18点，capacity-scenario-design.json和grid-sizing.json保存待测选点与网格对照；sources.json记录来源。已有真实画像补齐事项仍按用户要求暂缓，本轮未启动补采或大范围搜索。
"""
    (out/'REPORT.md').write_text(text)
    print(out/'REPORT.md')


if __name__=='__main__':main()
