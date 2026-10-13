"""Save plot data and report the bounded negative admission experiment."""
from .isolation import install
install('supervisor')
from .common import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def main():
    proto=read(OUT/'experiment_protocol.json');cpu=read(OUT/'cpu_results.json');completion=read(OUT/'CPU_stage_completed.json')
    smoke=read(OUT/'admission_smoke_result.json');assert not completion['GPU_trigger_passed'], 'This report implements the predeclared early-stop branch only'
    valid=[v for v in cpu['oracles'] if v['eligible']];models=proto['members']['P2'];figdir=OUT/'paper_figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
    def export(fig,name):
        for suffix in ['png','pdf','svg']:fig.savefig(figdir/(name+'.'+suffix),bbox_inches='tight')
        plt.close(fig)
    selection=[];columns=[];matrix=[]
    for scene in proto['development_scenarios']:
        for pool in proto['pools']:
            columns.append(scene.replace('LOW18','LOW')+'\n'+pool)
            chosen=next(v for v in cpu['oracles'] if v['scenario']==scene and v['pool']==pool)
            col=[]
            for m in models:
                available=m in proto['members'][pool]
                selected=m in chosen['deployment'] if chosen['eligible'] else None
                positive=sum(r.get(m,0.) for r in chosen['routing'].values())>1e-9 if chosen['eligible'] else None
                selection.append(dict(scenario=scene,pool=pool,model=m,candidate_available=available,selected=selected,nonzero_route=positive,CPU_evaluable=chosen['eligible']))
                col.append(3 if not chosen['eligible'] else 2 if selected else 1 if available else 0)
            matrix.append(col)
    csvsave('paper_figures/deployment_selection_data.csv',selection)
    fig,ax=plt.subplots(figsize=(11,4));ax.imshow(np.array(matrix).T,cmap=ListedColormap(['#f1f3f5','#cfd9e3','#2a9d8f','#f1d7ab']),vmin=0,vmax=3,aspect='auto')
    ax.set_xticks(range(len(columns)),columns,fontsize=9);ax.set_yticks(range(len(models)),['Qwen (G)','MiniCPM (G)','Phi4MM (G)','RoBERTa (S)','Whisper-small (S)'])
    for x,col in enumerate(matrix):
        for y,v in enumerate(col):ax.text(x,y,['—','eligible','selected','N/A'][v],ha='center',va='center',fontsize=8,color='white' if v==2 else '#263238')
    ax.set_title('Frozen Module35 agrees with finite CPU U Oracle: low loads admit specialists')
    ax.legend(handles=[Patch(color='#cfd9e3',label='Available candidate'),Patch(color='#2a9d8f',label='Resident with positive routing'),Patch(color='#f1d7ab',label='Frozen predictor lacks a LID sample')],loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,frameon=False)
    export(fig,'01_model_admission_selection')

    fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True);urows=[]
    for ax,scene in zip(axes,['LOW18','LOW18_HALF']):
        vals=[v for v in valid if v['scenario']==scene]
        xs=np.arange(len(vals));ax.plot(xs,[v['predicted_U'] for v in vals],'-o',color='#176b87',label='Finite Oracle = Module35 = 20A = fastest-path')
        ax.plot(xs,[v['P0_best_U'] for v in vals],'--',color='#9d5353',label='Best G-only allowed deployment')
        ax.set_xticks(xs,[v['pool'] for v in vals]);ax.set_title(scene+' (total '+str(round(sum(vals[0]['rates'].values()),3))+' QPS)');ax.set_xlabel('Nested candidate pool');ax.grid(axis='y',alpha=.2)
        for v in vals:urows.append(dict(scenario=scene,pool=v['pool'],Oracle_U=v['predicted_U'],best_G_only_U=v['P0_best_U'],delta_U_admission=v['delta_U_admission'],source='CPU_PREDICTION_NOT_GPU'))
    axes[0].set_ylabel('Predicted system U (higher is better)');axes[1].legend(loc='upper left',bbox_to_anchor=(0,-.25),fontsize=8,frameon=False)
    export(fig,'02_predicted_admission_utility');csvsave('paper_figures/predicted_admission_utility_data.csv',urows)

    delta=[v for v in cpu['breakdown'] if v['arm']=='BEST_MIXED_MINUS_G_ONLY'];fig,ax=plt.subplots(figsize=(9,4));xs=np.arange(len(delta));width=.18
    items=[('T_contribution','Throughput','#c4a000'),('L_contribution','Mean latency','#2a9d8f'),('P_contribution','P95 latency','#176b87'),('M_contribution','Memory','#c66b5b')]
    for i,(key,label,color) in enumerate(items):ax.bar(xs+(i-1.5)*width,[v[key] for v in delta],width,label=label,color=color)
    ax.axhline(0,color='#555',lw=.8);ax.set_xticks(xs,[v['scenario'].replace('LOW18','LOW')+' / '+v['pool'] for v in delta]);ax.set_ylabel('Contribution to predicted U(Mixed) − U(G-only)');ax.set_title('Predicted service-time savings exceed the added memory penalty');ax.set_ylim(-.025,.32);ax.legend(ncol=4,loc='upper left',frameon=False)
    export(fig,'03_admission_utility_components');csvsave('paper_figures/utility_component_data.csv',delta)

    fig,ax=plt.subplots(figsize=(6,3));ax.axis('off');ax.text(.5,.73,'Paired GPU utility comparison: NOT EXECUTED',ha='center',fontsize=13,weight='bold');ax.text(.5,.44,'No prespecified CPU scene retained G-only after expansion.\nFrozen stop rule prevents a paired GPU campaign.\nThe successful co-residency/path smoke supplies feasibility only.\nU, paired differences and 95% confidence intervals are unavailable.',ha='center',va='center',linespacing=1.8,fontsize=10)
    export(fig,'04_gpu_comparison_not_executed')
    gpu=dict(status='NOT_EXECUTED_CPU_RETENTION_GATE_NOT_MET',valid_independent_pairs=0,planned_maximum_pairs=3,actual_formal_GPU_sessions=0,actual_feasibility_smoke_sessions=1,
        U_G_only=None,U_best_Mixed=None,throughput_loss_G_only=None,throughput_loss_Mixed=None,mean_E2E_G_only_ms=None,mean_E2E_Mixed_ms=None,P95_E2E_G_only_ms=None,P95_E2E_Mixed_ms=None,
        memory_G_only_mib=None,memory_Mixed_mib=None,stability_G_only=None,stability_Mixed=None,paired_delta_U=None,CI95_lower=None,CI95_upper=None,
        development_probe_seed=351101,reserved_validation_seeds=[351201,351202,351203],validation_trajectories_executed=False,
        reason='No G-only retention case in the bounded CPU screening; feasibility smoke is not a utility measurement.')
    csvsave('gpu_paired_admission_results.csv',[gpu]);csvsave('paper_figures/GPU_comparison_status_data.csv',[gpu]);save('GPU_execution_accounting.json',gpu)
    runtime=[v for v in cpu['decisions'] if v['eligible'] and v['algorithm'] in ['CPU_U_ORACLE','MODULE35_FROZEN']]
    fig,ax=plt.subplots(figsize=(6,3.5))
    for label,color in [('CPU_U_ORACLE','#c66b5b'),('MODULE35_FROZEN','#176b87')]:
        for scene,marker in [('LOW18','o'),('LOW18_HALF','s')]:
            rs=[v for v in runtime if v['algorithm']==label and v['scenario']==scene]
            xx=[next(z['evaluated_routes'] for z in valid if z['scenario']==scene and z['pool']==v['pool']) for v in rs]
            ax.plot(xx,[v['seconds']*1000 for v in rs],'-'+marker,color=color,label=label+' / '+scene)
    ax.set_xlabel('Finite deployment-routing candidates evaluated by Oracle');ax.set_ylabel('Solver time (ms; single descriptive run)');ax.legend(fontsize=7,frameon=False);ax.grid(alpha=.2)
    export(fig,'05_solver_runtime');csvsave('paper_figures/solver_runtime_data.csv',runtime)
    figure_manifest=dict(figures=[dict(file=str(p.relative_to(OUT)),source='CPU_PREDICTION' if p.name[:2] in ['01','02','03','05'] else 'GPU_NOT_EXECUTED_STATUS',publication_claim_allowed=p.name[:2]!='04') for p in figdir.glob('*.png')],
        raw_plot_data_saved=True,real_GPU_utility_figure_available=False,load_switch_curve_available=False,not_measured_values_never_plotted_as_zero=True)
    save('paper_figures/figure_manifest.json',figure_manifest)
    table='\n'.join(f"| {v['scenario']} | {v['pool']} | {v['structure']} | {v['predicted_U']:.6f} | {v['delta_U_admission']:+.6f} | {', '.join(v['active_specialists']) or '—'} |" for v in valid)
    sourceSHA=ref(ROOT/'results/module35_utility_routing/u_aligned_optimizer.py')['sha256']
    minms=min(v['seconds']*1000 for v in runtime if v['algorithm']=='MODULE35_FROZEN');maxms=max(v['seconds']*1000 for v in runtime if v['algorithm']=='MODULE35_FROZEN')
    ratios=[next(z['seconds'] for z in runtime if z['scenario']==v['scenario'] and z['pool']==v['pool'] and z['algorithm']=='CPU_U_ORACLE')/v['seconds'] for v in runtime if v['algorithm']=='MODULE35_FROZEN']
    report=f'''# G-only Retention under Lightweight Specialist Admission：独立论文补充实验

本轮得到**有限预筛选中的负结果**：两个可评价低负载下，扩池后 Oracle 和 Module 35 都主动接纳合法轻量 Specialist，没有找到 G-only 保持案例。按照结果揭晓前冻结的规则，完成 CPU、可运行性审计和报告后停止，不启动正式 GPU 配对，不追加负载，不修改算法、画像、权重或质量门槛。

## 协议、候选与合法性

- 输出：`results/paper_extra_g_only_retention/`；Module 35 保持只读。本任务是验证，不是算法开发。
- P0 = Qwen、MiniCPM、Phi4MM；P1 = P0 + RoBERTa；P2 = P1 + Whisper-small；严格嵌套，原部署仍保留。
- 固定预算 49,140 MiB，安全系数 {proto['safe_factor']:.12f}；冻结20A `U = −0.25T −0.30L −0.30P −0.15M`，质量为硬约束。归一化基准、服务时间、容量和显存表完全沿用历史版本。
- 预先固定 LOW18（前五任务各 0.1 QPS、VQA 0.2）、其 1/2 和 1/8 倍；六任务比例固定，全部 QPS 为正。没有针对希望得到的结构调整目标任务占比。
- 副本域为每个模型 0/1；路由域为完整合法任务端点，即各 Specialist 使用比例 0% 或 100%。结论限于该离散域，不声称连续或更细分流空间的最优性。
- RoBERTa：990 MiB，Text 服务 5.5064 ms，历史准确率 0.955 ≥ 0.925；Whisper-small：1616 MiB，ASR 190.1659 ms、LID 27.0841 ms；ASR WER 0.042180839129459274 ≤ 0.042181，LID 按历史语义有效性准入（没有伪造数字阈值）。权重张量规模、实际权重大小及 SHA256 均保存于候选清单。
- Stage A 单次共驻留原生检查：Qwen + MiniCPM + RoBERTa + Whisper-small 四 worker，**{len(smoke['path_probe_records'])} 次各合法路径请求全部完成**，安全 idle 显存 **{smoke['safe_idle_memory_mib']:.3f} MiB**，全部 worker 正常退出，GPU 释放检查通过，无错误或意外写入。该检查证明加载、路径与共驻留资格，不能代替完整窗口的持续稳定性或效用验证。
- P0 Oracle 的完整六任务 G-only 是 Qwen + MiniCPM，各 1 副本，预测安全显存 42,059.152 MiB。P1/P2 均存在有正路由、完整覆盖、资源合法的 Mixed，新增 S 没有因质量、预算或加载失败被动排除。

`experiment_protocol.json` 在本轮 Oracle 调用前冻结。加载检查另有调用前冻结的 `admission_smoke_protocol.json`，仅限定一次 Stage A 检查，并澄清正式 GPU 触发条件是 P1/P2 中至少一池保持 G-only。没有根据数值结果修订场景。补充驱动曾有一次 CSV 重复键错误；失败日志及源代码已保留，仅修复结果序列化，冻结优化器未变。

## CPU 结果

| 负载 | 候选池 | Oracle / Module 35 结构 | 预测 U | ΔU_admission（相对同负载 P0） | 非零使用 S |
|---|---|---|---:|---:|---|
{table}

有效六个单元中，Module 35 的模型、副本、完整路由与 Oracle **6/6 一致，最大 Regret = 0**。原20A同域网格对照和保留的原生20A均达到同样选择；**简单最快路径启发式也达到最佳 U**。不能据此声称本小实验中新优化器优于这些基线。最小显存覆盖启发式在 P1 保持 G-only，但其 Regret 为 0.33118，因此不能把该启发式保持现象归给 U 优化器。完整部署、分流比例、搜索指标和运行时间已保存。

新增 Specialist 的最佳非零使用比例已逐一检查：RoBERTa 的 Text 为 100%；Whisper-small 的 ASR 与 LID 均为 100%。这些不是加载后零路由的伪 Mixed。还保留各模型最优 idle 与最优不部署对照。

LOW18 的 P1 Mixed−G-only：平均延迟项 +0.167269，P95 项 +0.167269，新增显存项 −0.003357，吞吐项 0，总计 **+0.331180**；P2 则分别 +0.265286、+0.265286、−0.008837、0，总计 **+0.521736**。HALF 的 P2 另有有限窗口吞吐项 +0.005952，总增益 +0.527688。这是**预测服务时间收益超过常驻显存代价**，没有出现可报告的拒绝部署原因。

两有效低负载下 G-only 的预测最大 rho 远低于 0.8，因此覆盖和容量可行；但“低排队”不等于“六任务归一化服务成本都低”。文本由 Qwen 的 44.6387 ms 降至 RoBERTa 的 5.5064 ms。统一缩放负载保留任务权重，原U中的基础服务时间收益仍存在，不能期待低总QPS自动导致拒绝 S。

**1/8 负载不计为保持或切换成功，也不计 Regret。** 固定120秒、phase=.37 的原预测器中，LID 的首个样本时间为 138秒，六任务 nominal cohort 缺失，P0/P1/P2 均 U=null，原因 MISSING_TASK。Module 35 同样返回无合法完整预测方案。保留这个原始限制，没有延长预测窗口或替换负载来生成结果。它不是 Specialist 质量、显存或加载不可行。

每池完整候选路由数为 8、24、104。Module 35 在有效单元耗时 {minms:.2f}–{maxms:.2f} ms；Oracle {min(v['seconds']*1000 for v in runtime if v['algorithm']=='CPU_U_ORACLE'):.2f}–{max(v['seconds']*1000 for v in runtime if v['algorithm']=='CPU_U_ORACLE'):.2f} ms；对应加速约 {min(ratios):.2f}–{max(ratios):.2f} 倍。只有单次描述性测量，无重复计时置信区间，不作为独立性能泛化结论。

## GPU 与高负载对照

正式 G-only vs 最佳 Mixed GPU 效用对照**未执行**：预声明的 CPU 保持条件不成立。独立配对数 0；U、吞吐损失、均值/P95 E2E、配对差和95%CI均为 null。已有共驻留检查不能转换成真实 U 优势，也不能称为统计上“两者无差异”。开发路径检查 seed=351101；保留给独立配对的 351201/202/203 未执行，未重复使用开发轨迹作验证。

Module 35 TEXT10 的 A 与本 P0 相同，但历史 B 含 Whisper-medium/large，历史 C 含多个其他 Specialist，均与本 P1/P2 不同。因此仅单独保存 TEXT10 历史语境和逐项可比性审计，**不拼接低负载保持/高负载切换曲线**，不新增高负载扫描或 GPU 场景。

## 论文使用边界与五个最终回答

1. **是否找到扩池后仍自主保持 G-only？没有。** 两个有效开发负载均选择 Mixed；最低负载预测不可评价。范围限定于本次固定候选、画像、负载和端点域，不证明所有低负载都无法保持 G-only。
2. **S 是否具备部署资格？是。** 原质量准入、测量画像、合法非零路由和实测四模型共驻留均通过；持续稳定性与正式GPU U比较未验证。
3. **Oracle 与 Module 35 是否一致？是。** 有效6/6完整方案一致，Regret=0；另外3个缺LID样本单元一致返回无有效预测，不能算3次成功选择。
4. **真实GPU U是否有可靠差异？没有可回答的配对测量。** 不满足CPU阶段进入条件，GPU正式比较未执行；不能用可运行性检查替代。
5. **能否支持论文的负载感知拒绝准入能力？本补充实验不能提供该正向证据。** 可以写入有限域CPU正确性、合法S主动接纳、效用分项和未发现保持案例的负结果；不支持真实G-only优势、保持→切换负载曲线、完整GPU全局最优或基线独占优势。

图表1–3和5仅标注CPU预测/描述性计时；图4明确展示GPU未执行状态，无伪造真实U柱形图。每图均保存原始CSV数据。`final_gate.json` 各状态分开记录；实验到此停止，不开启后续模块。

冻结 Module35 optimizer SHA256：`{sourceSHA}`。历史完整性检查涵盖历史结果、原始代码和模型权重；核验结论在最终Gate与独立SHA256审计中记录。
'''
    (OUT/'final_summary.md').write_text(report)
    save('report_values.json',dict(valid_cells=len(valid),invalid_cells=len(cpu['oracles'])-len(valid),retention_cases=0,algorithm_SHA256=sourceSHA,
        paired_GPU_runs=0,smoke_sessions=1,model_admission_feasible=smoke['passed'],module35_ms=[minms,maxms],speedup_single_run=[min(ratios),max(ratios)],figures=figure_manifest))
    print('REPORT_COMPLETE',len(valid),'valid cells','formal_GPU_pairs',0,flush=True)


if __name__=='__main__':main()
