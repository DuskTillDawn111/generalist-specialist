from .common import *
import statistics

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(v) for v in r)+' |' for r in rows]])
def main():
 fm=read(OUT/'frozen_problem_manifest.json');sp=read(fm['space']['path']);ds=read(OUT/'optimizer_decisions.json');oo=read(OUT/'cpu_oracles.json');cs=read(OUT/'gpu_contrasts.json') if (OUT/'gpu_contrasts.json').exists() else [];status=read(OUT/'campaign_status.json');hi=read(OUT/'history_integrity_after.json') if (OUT/'history_integrity_after.json').exists() else {'passed':False,'pending':True};errs=list(csv.DictReader((OUT/'objective_and_search_error.csv').open()));obs=read(OUT/'gpu_results_table.json') if (OUT/'gpu_results_table.json').exists() else [];complete=status['status']=='COMPLETED';cpuverify=read(OUT/'CPU_oracle_correctness.json')['passed'];reuse=read(OUT/'module33_GPU_reuse.json');quality=read(OUT/'quality_and_profile_preservation.json')['passed']
 def matches(m):return sum(r['regret'] is not None and r['regret']<1e-8 for r in ds if r['algorithm']==m)
 gpuvalid=sum(r['eligible'] for r in obs);supported=[c['group'] for c in cs if c['verdict']=='PREDICTED_ORACLE_SUPPORTED'];point_reversed=[c['group'] for c in cs if c['delta_statistics']['mean'] is not None and c['delta_statistics']['mean']<0];familywise_supported=[c['group'] for c in cs if c.get('significant_familywise_05') and c['verdict']=='PREDICTED_ORACLE_SUPPORTED'];reversed_=[c['group'] for c in cs if c['verdict']=='GPU_RANKING_REVERSED'];unresolved=[c['group'] for c in cs if c['verdict']=='STATISTICALLY_UNRESOLVED']
 gate=dict(module=34,completed=complete and hi['passed'],research_closed=complete or status['status']=='STOPPED_ON_FAILURE',deadline=fm['deadline_local'],scope='Finite endpoint routes K1, replicas0/1, budget49140MiB; no full model-pool or continuous optimum claim.',
    flags=dict(ORACLE_SPACE_INVALID=not(cpuverify and quality),CPU_ORACLE_COMPLETED=cpuverify,OPTIMIZER_OBJECTIVE_MISMATCH=True,OPTIMIZER_SEARCH_GAP=matches('20A_U_GRID')<9,GPU_ORACLE_INCOMPLETE=True,GPU_RANKING_MISMATCH=bool(reversed_),GPU_RANKING_UNRESOLVED=bool(unresolved),FINITE_PREDICTED_U_OPTIMALITY_SUPPORTED=cpuverify,MIXED_EMPIRICAL_OPTIMALITY_SUPPORTED=False),
    CPU=dict(cells=9,route_counts_by_pool={'A':8,'B':32,'C':8154},V3_native_finite_legal=6,V3_native_U_optimal=matches('V3_NATIVE_CONTINUOUS_PRESERVED'),V3_grid_control_U_optimal=matches('V3_J_GRID_CONTROL'),existing20A_grid_U_optimal=matches('20A_U_GRID'),original_native_failure_retained=True,U_method_is_existing20A=True,new_U_algorithm_developed=False),
    GPU=dict(campaign=status,valid_runs=gpuvalid,observed_runs=len(obs),expected_runs=24,preselected_distinct_plans=6,full_pool_oracle=False,independent_contrasts=cs,supported=supported,familywise_supported=familywise_supported,empirical_mean_direction_reversed=point_reversed,strict_CI_direction_reversed=reversed_,reversed=reversed_,unresolved=unresolved,historical33_admitted_C=sum(r['admitted_C_endpoint_binary'] for r in reuse),historical_development_not_independent=True),
    history=hi,optimizer_search_gap_methods=['20A_U_GRID','20A_NATIVE_CONTINUOUS_PRESERVED'],quality_unchanged=quality,no_algorithm_changes_after_GPU=True,formal_V3_not_replaced=True,no_automatic_Module35=True,
    paper_claim='Bounded CPU correctness, measured key-competitor outcomes, and negative search/prediction results. Does not establish universal V3 or20A optimality or pureG-vsMixed competitive crossover.')
 gate['CPU']['decision_component_agreement']=read(OUT/'decision_component_agreement.json')
 gate['CPU']['controlled_V3_no_original_acceptance_improver_cells']=sum(r['no_strict_acceptance_improver'] for r in read(OUT/'V3_original_acceptance_audit.json'))
 gate['GPU']['historical_independent_comparison']=read(OUT/'historical_large_vs_new_small_analysis.json')
 save('module34_gate.json',gate)
 rows=[]
 for o in oo:
  scen=o['scenario'];pool=o['pool'];ru=[]
  for m in ['V3_J_GRID_CONTROL','20A_U_GRID','FASTEST_PATH_STATIC','MIN_MEMORY_COVER_FASTEST']:
   r=next(r for r in ds if r['pool']==pool and r['scenario']==scen and r['algorithm']==m);ru.append(f'{r["regret"]:.6f}' if r['regret'] is not None else r['status'])
  rows.append([scen,pool,o['U_oracle']['structure'],f'{o["U_oracle"]["predicted_U"]:.6f}',o['evaluated_routes'],*ru])
 cpu_table=table(['负载','池','U参考结构','参考U','已评价有限配置','V3网格控制regret','20A U regret','最快路径regret','最小显存regret'],rows)
 timing=[]
 for scenario in sp['scenarios']:
  for pool in ['A','B','C']:
   rr=[r for r in errs if r['pool']==pool and r['scenario']==scenario and r['algorithm'] in ['V3_J_GRID_CONTROL','20A_U_GRID']];o=next(o for o in oo if o['pool']==pool and o['scenario']==scenario)
   timing.append([scenario,pool,f'{o["seconds"]:.4f}',*[f'{float(r["seconds_median"]):.4f} / {float(r["oracle_over_solver_speedup"]):.2f}×' for r in rr]])
 timing_table=table(['负载','池','穷举秒','V3控制秒 / 加速','20A U秒 / 加速'],timing)
 gpu_table=table(['比较','开发优胜','独立n','观测ΔU（Oracle所选−对手）','95%CI','判断','Holm p'],[[c['group'],c['development_best'],c['delta_statistics']['n'],c['delta_statistics']['mean'],c['delta_statistics']['ci95'],c['verdict'],c.get('holm_adjusted_p')] for c in cs])
 fail=[r for r in ds if r['status']!='FEASIBLE'];failuretable=table(['负载','池','方法','结果'],[[r['scenario'],r['pool'],r['algorithm'],r['status']] for r in fail])
 expansion=read(OUT/'gpu_pool_expansion_pair.json') if (OUT/'gpu_pool_expansion_pair.json').exists() else None
 text=f'''# Module 34：候选池扩张与部署选择评估

本轮已完成九组有限 CPU 空间的穷举及独立校验。原生 V3 保留；V3 的网格控制适配与已有20A U方法独立命名。GPU campaign 状态为 **{status['status']}**，有效测量 **{gpuvalid}/{len(obs)}**（预定24窗口）。历史保护核验：{hi}。

## 冻结问题与评价口径

三个嵌套池 A3⊂B5⊂C10，共用六任务正QPS、原20A效用和归一化参考、49140MiB安全显存预算、原质量准入、rho≤0.8。只改变候选集合。冻结U=−0.25T_norm−0.30L_norm−0.30P_norm−0.15M_norm（Q权重0，质量为硬准入）；L/P按配置任务QPS加权并除原冻结任务延迟参考，M=safe_memory/49140，T=1−min(完成比例,窗口内完成QPS/配置总QPS,1)。容量rho≤.8按原冻结画像与经验容量系数检查，不按本轮测试重写画像；实测另按原完整cohort、drain与稳定性规则准入。有限任务路由网格K=1：每任务选择一个合法模型；每模型副本0或1，驻留的空闲模型也计显存。结论覆盖这一受限空间，不覆盖原始多副本池或连续分流空间。原动态策略未改。

A：Qwen、MiniCPM、Phi；B新增Whisper Large/Medium；C再新增Roberta、ConvNext、PARSeq、Whisper Small和质量合法但名义性能受支配的Bart。Whisper Small ASR恢复原8B完整精度质量准入，WER0.042180839129459274≤0.042181；33对其ASR路径的排除是当轮人为受限域，不是原质量失败。来源逐项见quality_admission_provenance.csv。三种负载LOW18、ASR32B直接来自历史18/32B；TEXT_HALF_MID18仅将历史18 text_mid_rep1文本20QPS乘1/2为10QPS，其他任务不变，先冻结再运行CPU。

正式V3的J=Σ_m[D_m/(1−D_m/c_m)]/Σ_tλ_t，D_m=Σ_t r_tm·s_tm（秒），c_m为原经验容量系数，是无额外参数的拥塞代理。以J分配路由和候选排名，但驻留部署的接受判据仍为原NoiseAwareRuntimeObjective（均值/P95/显存与5%改进门槛），不能把它描述为全局单一J优化器。V3_NATIVE_CONTINUOUS_PRESERVED执行原逻辑，两个最终结果不在K1域，一次原生J数值失败；这些结果不得计算本有限域regret。V3_J_GRID_CONTROL只将路由/排名注入为固定部署下的精确K1 J路由，原候选产生、top2及接受/停止不变，是独立域控制实验，不冒充未改原生V3。两者的运行时后端均是原20A CPU预测，计时不包含在线GPU测量。

20A_U_GRID使用原20A 42次MILP候选产生计划和原U预测排序，只做路由整数单位适配；20A_NATIVE_CONTINUOUS_PRESERVED另行保留。没有开发新U算法，没有按本轮GPU结果调参。原20A的效用函数不等于经过充分验证的性能预测器，其历史UTILITY_MODEL_INSUFFICIENT结论保留。

## CPU主结果

{cpu_table}

8/32/8154是经过覆盖与显存筛选后、逐项容量和预测稳定性校验的路由配置数；所有配置均用同一冻结评价接口。全驻留向量数分别8/32/1024，筛除向量见feasible_deployments.csv。独立验证采用“先任务路径、后驻留超集”的逆向枚举，全部九组与原枚举集合完全相等。A/B所有路径及C五个预定位置另用独立标量FIFO模拟核对原预测U，误差≤1e-8。

原生V3有限合法6/9，其中U最优{matches('V3_NATIVE_CONTINUOUS_PRESERVED')}/9；其余不合法域/失败不算成功。V3网格控制合法9/9、U最优6/9；20A U合法9/9、U最优5/9，其中C三负载均最优。A→B→C的参考U及两个域内方法U均未降低。已有20A在九组中部署结构、驻留模型和副本均与U Oracle一致，四处错误全部发生于路由；不能只用“结构选对”掩盖路由regret。V3网格控制结构一致9/9、具体驻留模型/副本一致8/9、U等价6/9，见decision_component_agreement.json。Bart移除后有限Oracle U完全不变，见unhelpful_candidate_control.json；该记录证明Oracle层面的无益新增控制，不宣称已对所有算法额外做独立消融。

A只能用纯G保证覆盖，故其纯G最优不能当作允许Mixed时的纯G竞争胜利。B/C参考最优均Mixed。纯S缺少合法VQA路径，全部不可行；不得填入伪合法U。最小显存启发式在C三场景匹配Oracle，算法对所有合理启发式的普遍优势未成立。

## 误差归因

20A在A/B的ASR32B及TEXT10四格中，生成集内U排名正确，但漏掉更好的部署/路由；这是候选生成搜索误差。另核对原生20A连续方法与网格适配的九组最终方案，二者驻留/U相同、路由按冻结1e-10归一化指纹等价（C有约3e-15的MILP浮点残差）（existing20A_native_vs_grid_decisions.json），因此本轮四个终点路由遗漏不是仅由整数网格适配新产生。二者中间候选域仍不同，原生结果单独保留。最大预测regret10.844309（A TEXT10）。同一G2部署中，最快服务路由把图像/ASR/VQA留给忙于文本的Qwen；Oracle将这三类任务转给MiniCPM，降低队列竞争。B TEXT10同理，V3控制已找到相同U参考最优。这证明队列敏感候选生成是已有20A的关键缺口，不能仅改名为U优化就声称解决。

V3控制的三处U差距LOW18/C .004391、ASR32B/A .355421、ASR32B/B .041308，对应其已找到J参考最优，但该J方案并非U最优。其余C ASR/TEXT10虽没有选择全局J方案，却在原多指标接受规则下选到U最优；这再次说明全局J差距不是该引擎唯一“自身目标搜索误差”。J拥塞代理与U的周期FIFO性能接口不同，J/U两种精确Oracle差异同时反映目标定义及预测代理差异；不能将其全量归为纯目标权重误差。详见objective_and_search_error.csv和oracle_margins.json。另对每组所有合法驻留状态（C302状态/组）先选择精确J网格路由，再使用原NoiseAwareRuntimeObjective逐项与控制版最终方案比较，九组均无原接受规则判为严格更好的候选，见V3_original_acceptance_audit.json。这是完整有限域的多指标“无可接受改进”审计；比较器含容忍门槛且可能依赖路径，不能说成唯一标量全局最优，也不能外推到原生连续域。正式V3原生数值失败完整保留，不在GPU后修复重验收。

原20A预测器固定使用120秒周期到达、phase0/.37及1.25×/2×服务压力场景；名义U来自phase.37，压力上下界不是统计置信区间，也不是Poisson请求模拟。真实GPU为预冻结Poisson协议（开发120秒、独立240秒），故到达突发性、画像设备、并存干扰及时间窗都属于预测适用性差异，不能全部归因于搜索。per_task_prediction_diagnostics.csv分列模型服务、worker队列及主机传输/调度残差，不通过改预测器消除本轮负结果。

真实预测误差依据每个有效GPU窗口的观测U与同一预算下冻结预测U计算，见gpu_results_table.json；排名结论必须依据独立观测及其不确定性，不以预测自证真实最优。prediction_search_decomposition.csv按恒等式“观测有符号差=冻结预测差+差分预测误差”分列；两类误差可以抵消，不强行拆成正百分比。method_empirical_regret.csv将每方法映射到实测相同方案，报告A/C两方案及B四个同TEXT10已测方案集合的观测均值regret，仍不是完整CPU池真实Oracle；未测方法保留缺失，不伪造实测值。

## 搜索效率

{timing_table}

20A计时为三次CPU独立求解的中位数；V3和Oracle各一次。20A每次42次MILP，评价次数与合法候选访问比例在objective_and_search_error.csv。A很小，20A常比直接枚举慢；C复杂度8154个路由时20A快118–252倍，V3控制快239–905倍。访问次数不等于不同域内状态数，不能把重复运行时调用当成唯一空间访问。精确合法域内访问比例和预测分量调用数见optimizer_evaluation_count_audit.csv。原保存字段J_route_solves对共享网格ranker缓存重复求和；计数审计从全部日志重构缓存部署集合，确认实际状态数为原字段一半，再计算各状态所有有限路由J评价次数。原字段/原求解结果保留，不把缓存数当函数调用数；原生连续J内部迭代调用未插桩，明确缺失，未伪造。未执行真实GPU后端在线搜索计时，不能据此宣称端到端部署延迟同倍率降低。

## GPU独立证据及边界

原GPU清单在观察任何GPU测量前冻结六方案、开发seed340101/120秒、独立seed340201/2/3/240秒和AB/BA/AB次序。CPU选择记录在gpu_selection_record.json：增加A最大搜索缺口病例，保留原CPU freeze；B最大20A/最快路径预测regret；C预定ASR32B Phi与Qwen比较。开发测量建立这三个两方案集合的排序；独立轨迹验证各自预选方案，未按开发结果临时换竞争结构。每窗口全六任务Poisson工作负载、共同输入轨迹与路由均匀数、30秒暖机、60秒有界drain。稳定准入沿用原分类器：unstable不合法；recoverable_congested必须完整完成并确认恢复才可获得U，因此“有效”不等于全部窗口全程steady（逐项gpu_stability_audit.csv）。严格检查所有请求、原worker时间戳、系统/每worker稳定性、安全显存、外部GPU进程、温度/时钟、持久终端日志及两套原U算术。原module26.arithmetic在未调用其旧_path观察器时，components.U/null和components.eligible/false保持默认占位；它的raw_arithmetic_U是纯算术结果，本轮顶层gpu_observation.eligible与U才是经全部冻结检查后的准入值（两套算术误差≤1e-12），不以占位字段替代本轮准入。不稳定或硬约束/技术无效方案U为NULL。

{gpu_table}

ΔU=CPU U Oracle所选方案减对手。误差线是三个独立种子配对差值的95%t区间（n=3，依赖配对差值近似正态，统计功效有限）；同时报告三比较Holm校正p，不将跨零微小差距当唯一赢家。对照温度均值差≤5℃、时钟相对差≤15%时才作可比性判断。开发与独立记录全部保留，包括负结果、故障和不稳定窗口。

A Oracle纯G与B Oracle Mixed复用相同TEXT10独立输入，可以检查同一B域内两种结构的观测对比：{expansion}。这仍只是两个预选方案的比较，不能代替全部B配置GPU测量。

Module33八组有效开发GPU中，只有三组符合本轮C成员、副本1与端点路由域；已按49140预算重算U，原U/raw/Gate均只读。OCR2及0.5分流方案明确排除，详见module33_GPU_reuse.csv。重算历史开发数据不是独立新轨迹，也不把33旧预算下的全八配置Oracle自动移植为34全C真实Oracle。另外复用33三组原独立240秒LargeASR结果，按新预算重算，并对本轮SmallASR独立结果作次要未配对Welch比较（historical_large_vs_new_small_analysis.json）；seed不同，绝不称配对验收。该分析在本轮C独立结果产生前锁定，不改变原三项主配对或方法，环境不匹配时不作明确胜负结论。software_environment_versions.json记录本轮实际Python与包版本；14F未提供完整可比包版本，所以不根据相同env路径推断历史软件完全相同。本轮未安装/修改原环境，原worker代码和权重保护。历史14F性能画像来自另一GPU UUID；33/34测量使用当前同一4090（49140MiB）UUID，设备来源差异保留。

共六个预选方案全部测量也只建立各两方案集合内观测最佳，不覆盖CPU枚举全集；故GPU_ORACLE_INCOMPLETE保持true，MIXED_EMPIRICAL_OPTIMALITY_SUPPORTED不自动通过。GPU排名反转时记录诊断，把性能预测改进留到后续研究，原方法结果不替换。

## 全部失败与不可行结果

{failuretable}

所有准备阶段软件事件见infrastructure_preflight_events.json；它们未产生科学测量，全部科学raw无重试/覆盖。启动预检发现监控JSON序列化别名缺失，发生于任何模型启动/GPU测量之前；原清单和错误日志保留。frozen_gpu_manifest_v2只补序列化别名、接受版本清单参数、正确raw provenance，配置/轨迹/算法不变。Phi CPU缓存预检首次误把示例训练文件也纳入依赖检查，缺datasets/sacrebleu；真实推理的三个原动态模块及MiniCPM原模块随后通过，未装包或改推理。技术campaign停止原因（若有）见campaign_status.json及gpu_campaign_failure.json；不删除负结果或覆盖raw。

## 论文适用性与10月16日前优先项

已具备嵌套池、严格有限CPU Oracle、方法自主部署与路由、预测regret、真实关键竞争对照/失败和效率图的材料。原生V3不是所有同域病例均成功，20A存在四个显著搜索缺口；不能写成“优化器普遍接近真实全局最优”或“纯G/Mixed交叉获胜已证明”。论文可采用限定范围的正确性、失效分析和可复现系统比较，具体GPU支持程度以本节结果为准。

10月16日前最关键的是补一组C池TEXT10同负载的独立GPU测量，使A→B→C扩张链具有完整实测证据；当前C的GPU验证在ASR32B，不能拼成三池同负载实测曲线。若论文必须主张纯G与Mixed竞争交叉获胜，再冻结一个两者均合法的开发场景及独立轨迹；本轮A强制G-only不能充当该证据。跨零的小差距可以诚实报告无法区分，只有论文必须主张严格优势时才增加少量预定配对。已有20A的候选生成缺口若成为论文主算法问题，需要独立命名版本和新holdout；不得在本轮结果上改算法重验收。多副本/连续全局最优不属于现有证据，优先收窄论文范围，停止无限追加模块，保留10月16后写作时间。

## 复现与产物

所有新源、缓存、临时、GPU记录在results/module34_candidate_pool。冻结问题/空间为frozen_problem_manifest.json、candidate_space_definition.json；九组候选在cpu/*_candidates.csv；独立校验CPU_oracle_correctness.json；决策/时效在optimizer_decisions.csv、optimizer_runtime_benchmark.csv；GPU清单frozen_gpu_manifest_v2.json；图在figures/*.png和*.pdf；Gate为module34_gate.json。

CPU复现应在新的隔离输出目录复用这些冻结源/空间，禁止覆盖本轮时间和GPU原始记录；独立验证可执行PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=results/module34_candidate_pool:. /root/miniconda3/envs/gsdeploy/bin/python -m m34.verify_cpu。campaign具有禁止覆盖raw保护，不应重跑同名窗口。历史SHA256及mtime前后全检含权重；执行器Landlock/seccomp与Python审计限制所有历史写入。Module34结束后停止，不自动进入35。
'''
 (OUT/'module34_summary.md').write_text(text)
 (OUT/'ranking_error_analysis.md').write_text('# 排名与决策误差\n\n'+text.split('## 误差归因')[1].split('## 搜索效率')[0]+'\n\n'+text.split('## GPU独立证据及边界')[1].split('## 全部失败')[0])
 print('REPORT_WRITTEN',status['status'],flush=True)
if __name__=='__main__':main()
