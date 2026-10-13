"""Compose Module30 reports from independently reproduced, offline observations."""
import sys
sys.dont_write_bytecode=True
import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_text())
def write(n,s):(P/n).write_text(s)
def main():
    st=read('stability_reproduction.json')['runs'];guard=read('dynamic_guard_breakdown.json');a=read('audit_findings.json')
    busy=read('run_analysis.json');pressure=read('tail_native_pressure.json');assoc=read('overlap_associations.json');q=read('G_quality_cluster_sensitivity.json')
    params=json.loads(Path('/root/autodl-tmp/generalist-specialist/results/module28_dynamic_split/router_parameters.json').read_text());S=params['S_worker'];G=params['G_worker']
    dyn=st['Dynamic'];old=st['S_only'];sf=busy['Dynamic']['worker_busy'][S];gf=busy['Dynamic']['worker_busy'][G]
    tail=list(csv.DictReader((P/'tail_growth_decomposition.csv').open()))
    lid_slope=float(next(r for r in tail if r['run']=='Dynamic' and r['scope']=='WORKER_TASK' and r['key']==S+'|lid')['slope_contribution_rps'])
    fraction=lid_slope/dyn['formal_tail_only']['slope']
    overlaps=assoc['lid|True']['service'];non=assoc['lid|False']['service']
    summary=f'''# Module30 / Stage A：稳定性失败的离线归因

**结论：原有限窗口硬失败精确复现，主要定位到S上的LID在末60秒的局部拥塞；没有动态收益或共置干扰因果证据。新Gate为GO，范围仅是准备有预算的稳定性边界探索预注册；当前负载的合法U三臂比较为NO_GO。未启动GPU。**

|原600秒口径|S-only|Dynamic|
|---|---:|---:|
|发送 / 窗口内成功 / 末未完成|7607 / 7575 / 32|7607 / 7588 / 19|
|排空秒数|2.444690107|1.429819885|
|尾150秒OLS斜率，请求/s|{old['formal_tail_only']['slope']:.9f}|{dyn['formal_tail_only']['slope']:.9f}|
|尾5个bin首末均值增长|{old['formal_tail_only']['growth']:.9f}|{dyn['formal_tail_only']['growth']:.9f}|
|相邻bin递增比例|75%|75%|
|原分类 / 合法U|unstable / NULL|unstable / NULL|

**观测事实。** `sustained_positive_tail_growth_with_nontrivial_end_backlog`要求斜率≥0.02、均值增长≥5、递增比例≥0.6、末pending≥5同时成立；两次均满足。Dynamic末19个为S上的18 LID+1 ASR，S-LID贡献全系统斜率{fraction:.2%}。尾bin均值为2.361、2.957、2.333、9.576、11.271，bin末pending为1、3、3、0、19：570秒曾完全回到0，故不是全程单调增长，末段局部事件主导有限窗口判定。

**可复核计算。** 67次G分流按60秒为{a['G_diversion_by_60s']}，集中在90–180、240–300和540–600秒，首末为94.011498399/598.582042411秒；尾150秒20次。7005个实际目标决策中814个满足当时滞回与代价条件：67分流，747留S；后者全部同时令牌未满，互斥首要理由为令牌361、冷却346、G在途34、G背景6。各守卫失败互相重叠，不能相加或推断放宽后会多分流747次。

**当前最支持的描述性解释。** 最后30秒S收到402、响应383，平均13.4/12.7667rps，handler忙碌97.29%；全600秒S仅忙{sf['native_busy_formal_seconds']/600:.2%}，G忙{gf['native_busy_formal_seconds']/600:.2%}。记录支持“局部到达/服务失衡仍留在S、冻结节流在拥塞时限制转移”的组合描述；不证明永久容量不足。G空闲门槛在814个代价合格点只阻断40个，不是当前主要记录限制。

**假设与无法判断。** ASR共占S {sf['busy_by_task']['asr']:.3f}秒；403对执行重叠中S-LID服务mean={overlaps['mean_ms']:.2f}/{non['mean_ms']:.2f}ms（重叠/未重叠）。ASR、共享资源、温度、输入长度和插桩可能有影响，但无受控反事实。真实worker receipt/等待队列UNKNOWN，S-only collector提交状态UNKNOWN。G-LID正确61/67，名义Wilson95%为81.81%–95.83%，仅27个不同输入；重复样本及选择偏差阻止将其视为独立质量认证，预测错误类型/闭集字段UNKNOWN。

**决策与保留。** 先预注册稳定性边界扫描，固定负载网格、种子、顺序和停止规则；满足共同稳定与原全部准入后才能另行批准三臂合法效用比较。过载救援只报告稳定性，不能冒充ΔU。Module28 Gate DYNAMIC_EVIDENCE_GAP、Module29 EVIDENCE_INCOMPLETE及功能验证Gate均原样保留；U/ΔU=NULL。309个输入集合、大小、SHA-256前后核验不变。所有新文件仅在本目录。
'''
    write('summary.md',summary)
    competing=f'''# 竞争性解释：支持、反证与未知

本报告中的“支持”只指与记录一致。没有匹配新对照或政策干预，不进行因果归因。主队列是host send-start减reader成功响应的B(t)，不是worker等待队列。

|解释|可复核支持|反证/限制|未知与检验需求|
|---|---|---|---|
|触发较晚、次数有限|首次94.0115s，67/7005=0.9565%，仅20次在450–600s；当前814个cost-eligible点只转移67|首个分流早于尾段350多秒，不能用“首次太晚”单独解释末尾失败；实际总体转移率0.1117rps低于0.575令牌上限|放宽规则后的状态/收益UNKNOWN，不能用独立快照计算更多真实分流；需另协议，不在此调参|
|令牌/冷却限制有效动作|747个cost-eligible但留S点全部令牌不足；互斥首要理由361 token、346 cooldown、34 inflight、6 background；token桶容量1，闲时信用不能无限积累|这些是同一实际轨迹中的相关失败条件；冷却失败多数也token失败，不能独立叠加为两个因果贡献；保护G可能必要|加速转移是否拥塞G、膨胀S、损害背景任务UNKNOWN；保留冻结策略探索其当前稳定边界|
|G空闲门槛是主要瓶颈|7005决策中G pending>0有854次；其中355次G目标在途，499次优先理由为G其他工作|G_empty通过6151/7005；cost-eligible内仅40次G非空（34+6），远少于节流限制；native G全窗闲{1-gf['native_busy_formal_seconds']/600:.2%}|G handler闲不是物理GPU有免费余量；没有receipt ACK，在线idle为host pending条件|
|S上非目标ASR阻塞|124个ASR占S正式窗{sf['busy_by_task']['asr']:.6f}s，尾窗{sf['busy_tail_by_task']['asr']:.6f}s；最后30秒占2.759401s；串行handler期间LID无法同时执行|末未完成仅1个ASR、18个LID，且S-LID贡献约94.77%斜率；不能将全部增长归ASR|无同条件去除ASR/分离副本对照；质量/输入长度和执行成本混杂|
|到达突发与服务速率局部不平衡|尾S发送数332/323/308/362/402；最后30s入13.4rps、出12.7667rps，S忙97.29%；540–600 bin均值明显增大；593.067–600高队列episode被截断|600s S忙87.20%、尾150s忙85.12%；570s队列归0；不支持全时域连续单调积压或已证实永久过载|600秒+一个种子不能估计长期边界。记录为stationary Poisson轨迹，未获得规则周期性证据；5s microburst表仅描述，不能倒选有利种子|
|共置执行膨胀导致失败|403对重叠；S LID重叠n382 mean79.1824ms、未重叠n6556 mean68.7753ms；ASR重叠21/未重叠103，mean466.39/326.97ms|重叠组在高队列/时段/输入上有选择偏差，较长服务本身更易被分入重叠组；另一S-only run无G目标也unstable|GPU/CPU/热态/框架各贡献与无分流服务反事实UNKNOWN；不能从组均值差宣称因果膨胀|
|插桩、热态、固定窗口切点影响观测|Dynamic原decide mean29.94µs，但含日志arrival mean977.00µs；S-only状态检查不是相同插桩；分别运行、热轨迹不同；5个尾均值点有非零残差|都保留干净退出与全部排空；这些因素不是原硬失败条件本身；不能移600s边界使历史合法|缺少统一harness的交错匹配对照；不能量化插桩或温度的因果影响|

## 失败的确定性位置

Dynamic全系统斜率0.0814665635分解：S-LID 0.0772063225、S-ASR 0.0019515757、G-LID 0.0013068069、G-VQA 0.0010215902，其他近0。任务与worker是不同分组，不把两套分组再次相加。末150s共发送1859、响应1840，450s B=0、600s B=19。最后30s之前570s B再次为0，故19个末未完成全由570–600s净流入产生。

原`growth=8.9104267153`是最后和第一个tail-bin的**时间均值差**，不是端点增长19。`increasing_fraction=.75`是4次相邻均值差中3次上升，不代表事件级75%时间增长；OLS残差为{dyn['formal_tail_only']['residuals']}。字段和数值均可由stability_bins.csv/tail_growth_decomposition.csv复核。

## 排空与判定口径

原类名中的“sustained”是冻结有限窗口规则。20个30s正式bin及尾150s的OLS完全不含排空。原实现还读取全cohort恢复、排空超时、完整观测peak/警告；本审计照原实现复现并单列它们，未将排空纳入尾段拟合，也未以最终恢复撤销unstable。不得改为“窗口末未完成即失败”、不得改变bin数量、移动截点或追溯延长窗口。
'''
    write('competing_explanations.md',competing)
    quality=list(csv.DictReader((P/'quality_by_task_worker.csv').open()))
    lines=['# 质量、时间与干扰证据边界\n\n',
        '原LID策略SEMANTIC_VALIDITY_ONLY，准确率是report-only、没有数值阈值。本报告不发明新阈值，不追溯剔除错误，不恢复合法U。推理成功率与correct、WER、闭集语义有效性分别报告。\n\n',
        '|运行/worker/task|请求n|correct观测n/正确|accuracy|WER观测n/mean|闭集flag观测n|\n|---|---:|---:|---:|---:|---:|\n']
    for r in quality:
        short='S' if r['worker']==S else 'G' if r['worker']==G else r['worker'].split('#')[0]
        lines.append(f'|{r["run"]}/{short}/{r["task"]}|{r["n"]}|{r["correct_available"]}/{r["correct"]}|{r["accuracy"] or "UNKNOWN"}|{r["WER_n"]}/{r["WER_mean"] or "UNKNOWN"}|{r["closed_set_flag_n"]}|\n')
    lines.extend([f'\nG LID 61/67=91.0448%，名义95%Wilson[{q["nominal_Wilson95"]["lower"]:.6f},{q["nominal_Wilson95"]["upper"]:.6f}]。这67次只有27个不同sample_id；按输入cluster的探索bootstrap区间[{q["exploratory_sample_cluster_bootstrap_percentile95"][0]:.6f},1.000000]（10000次、分析RNG=300030，不改变负载种子）。两种区间都受路由选择、重复样本与未验证可交换性限制，不是独立GPU复现或总体质量保证。\n\n',
        'Frozen轨迹与两次raw的task/pool_id/sample_id/offset完整对上，可获得LID语言标签，聚合见quality_LID_by_language.csv。Dynamic原预测文本被脱敏；6个incorrect的实际预测类、解析失败/错语言等错误子类型UNKNOWN，不能构造混淆矩阵。S/G路由子集不相同，也不能用另一run的同ID回答恢复本次未执行的S反事实。\n\n',
        '**缺失字段澄清：G LID的67条worker_response均未保留valid_closed_set_output。** 历史统计explicit_invalid=0只表示没有记录到False，不证明67条全都有效；本Module记closed_set_validity=UNKNOWN并保持历史文件原样。VQA有单独有效性flag，不能借用到LID。\n\n',
        '时间分解使用同一formal origin：scheduled→enqueued为调度/观测/日志等host时间；enqueued→send为host sender排队；send→native handler start为worker_queue_wait_ms（包括pipe/transport，非纯worker等待）；handler start→end为服务墙钟（包含CPU/GPU，不是kernel时间）；end→reader response为传播。collector commit晚于reader收到，只有已commit且按冻结harvest逻辑可见报告才能进EWMA。上述分量逐请求和e2e相加核对，router开销属于host分量，不能再加一次。\n\n',
        '动态原decide mean/P95=29.94/46.54µs；instrumented observation+decision=916.15/1690.08µs；完整arrival=977.00/1795.35µs。S-only约99µs状态检查不等于原decide或同样的完整插桩开销。warmup/prime与正式窗分开，cohort延迟可含排空完成，但正式窗口吞吐不含排空。\n\n',
        'worker pending/accepted waiting队列缺receipt ACK，均UNKNOWN。host pending含发送队列、pipe、执行和报告传播；native未开始handler数量也只能作transport-inclusive代理。native执行时间是收到响应后离线映射，不能用它或之后的完成时间回填过去路由。\n\n',
        '403对S/G执行重叠复核，包括382个S LID、21个S ASR；G串行handler意味着VQA不能与同worker G-LID同时执行。4个VQA在G-LID执行时提交，等待mean287.24ms，其余101个mean19.11ms；这是串行时序关联，不能将差值当附加因果延迟。无新增匹配S-only/fixed控制，输入、时段、温度、CPU共享、策略选择和服务时长选择偏差未控制。\n'])
    write('quality_interference_caveats.md',''.join(lines))
    draft='''# 下一次GPU实验预注册草案（不执行、未获执行批准）

目的首先是识别**当前冻结路由器的600秒有限窗口稳定性边界**，而非直接比较合法U或优化器创新。以下是候选协议，必须在任何新GPU测量前审阅、冻结代码/数据/顺序/预算；本Module没有生成新轨迹、启动worker或批准执行。

## 不变项与待冻结项

沿用原模型池、驻留各1副本、冻结DynamicRouter逐字节代码和参数、质量政策/任务语义、Module20A效用参考、原安全显存预算/因子及stability_v2全阈值。不要为通过测试改G令牌率、冷却、滞回、P10/P95、背景惩罚或unstable标准。600秒formal、30秒warmup、原prime、60秒排空上限、原温度条件与最大5°C组内热差；所有arm使用统一collector/时点/日志harness。未参与开发的轨迹按预注册seed一次生成；相同level/seed两臂用相同输入、计划到达和背景任务。

## 候选分段扫描与规模

基准向量为原冻结LID11.5、其他五任务各0.2rps。共同乘数网格固定为 **[0.60,0.80,1.00]**，按升序扫描，保持任务比例且六任务正QPS。这是公开的基准缩放规则，不以某seed的动态结果择点。

初扫仅S-only和冻结Dynamic两臂，每level两组新seed **300101、300102**，共3×2×2=12个600秒formal会话。每组交错顺序按(level_index+seed_index)奇偶预定S/D或D/S；不能因上一臂科学unstable而遗漏安全可执行的已注册同组另一臂，因为过载救援正需要该对照。任何基础设施/显存/外部进程/任务错误则按安全规则停止。

若出现相邻网格level、至少一臂从两seed均非unstable变为至少一次unstable，可**仅一次**在最低这对相邻level的算术中点细扫，seed固定 **300201、300202**，两臂共4会话。无明确相邻变化则不细扫，不向上/下无限外扩，不用效用或动态优势选细扫位置。双臂在同一level两seed都unstable时保留结果、完成当前安全组后跳过更高level。

复核候选边界：按固定规则选择“最高两臂均通过原稳定性/合法性检查的已扫描level”，seed **300211、300212、300213**，两臂6会话。没有共同通过点则停止，结论EVIDENCE_INCOMPLETE/无共同稳定区域；不得补有利seed。初扫+唯一细扫+复核最多22会话、13200秒formal（另计prime/warmup/冷却/排空）；实际执行前须批准这一预算，可缩小规模但必须在结果前冻结。

## 输出、验收与停止

每会话保留全部计划/发送/窗口内成功/末pending/排空，统一原20×30s bin、尾5bin OLS及全分类条件、六任务mean/P95、正确率/语义诊断、显存、温度、overhead、decision state/harvest与G/S实际路由。保留质量评分所需的规范化语言类别/闭集有效性标识，禁止答案进入策略。receipt ACK/真实worker等待若仍没有则明确UNKNOWN；这些不是原B(t)稳定性规则所必需，但限制机制解释。

资源超限、OOM、worker异常、外部GPU进程、任务/语义路径故障、排空超时立即停止，完整保留失败，无重试。科学unstable不修改阈值，按上述预注册边界规则停止更高level，不按效用再选择seed。动态在共同稳定区若G路径始终0，忠实报告策略未启用；不能提高负载或松守卫强迫双路径。

**过载救援**：S-only unstable、Dynamic非unstable，只能作为有限窗口稳定性分类改善候选；baseline合法U=NULL，不能计算合法ΔU。至少三组新seed重复，并保留所有阴性组和有限窗口限制。

**稳定区域三臂合法效用比较**：必须另行审阅批准。候选level由最高共同稳定规则选定，不用已有效用选点；在该level冻结S-only、开发预选固定10%、原Dynamic，统一模型/QPS/输入/热态/插桩。新seed **300301、300302、300303** 三组，三臂顺序事先拉丁方交错，共9会话/5400秒formal；不自动属于上面的22次预算。每组全部对照须满足原质量、显存、容量和稳定性条件；Dynamic两条目标路径都非零。任何一组不合法，U/ΔU保持NULL，不当作正收益。

正式三臂另须满足原manifest每任务至少72个成功观测的覆盖要求；容量pilot不能以样本不足的steady判定替代正式效用准入。所有候选新seed在执行批准前核对未参与历史策略开发；任何冲突只能按结果之前冻结的生成规则处理，不能根据效用替换seed。

若要沿用Module28独立收益门槛，须在新实验前冻结“相对于该组最强合法稳定对照三组ΔU均>+0.01”及CI报告规则；n=3区间非常脆弱，不能证实跨模型泛化。负收益不换seed、不重调参数、不挑best-fixed比例；不进入正式优化器集成。

## 批准条件与证据边界

当前数据足以提出有预算、按原定义的边界测量问题，故StageA GO仅针对**预注册准备**。启动前须批准具体预算和protocol，证明统一新harness仅扩日志而不改策略/模型语义、输入源/原实现哈希一致、GPU隔离与预算可用、所有seed与arm顺序冻结、保留G语义质量证据。现有记录不具备在原11.5 LID负载直接比较合法U的条件。最终边界是有限600秒、特定硬件/模型池/输入分布的操作性边界，不是无限时域容量保证。
'''
    write('next_experiment_preregistration_draft.md',draft)
    gate=dict(module='Module30 StageA',recommendation_gate='GO',scope='PREREGISTRATION_PREPARATION_FOR_BOUNDED_FINITE_WINDOW_STABILITY_BOUNDARY_ONLY',
        GPU_execution_authorized_by_this_gate=False,GPU_runs_started=0,formal_utility_comparison_at_original_load='NO_GO',
        causal_dynamic_gain_or_interference_attribution='EVIDENCE_INCOMPLETE',reason='Exact finite-window failure localized; actual router and service logging usable. Need independent boundary measurements before legal utility controls.',
        original_gates_preserved=dict(Module28='DYNAMIC_EVIDENCE_GAP',Module29_A='EVIDENCE_INCOMPLETE',Module29_B='CONTROLLED_SPLIT_LOGGING_VERIFIED'),
        original_stability_preserved='unstable',legal_U=None,delta_U=None,optimizer_modified=False,
        unobservable_states=['S-only collector commits','worker receipt/accepted pending/waiting queue','Dynamic G-LID prediction class and closed-set validity','controlled no-diversion service counterfactual'],
        required_before_GPU=['Review and freeze exact budget/grid/new seeds/ordering/stopping rules','Unified harness with original code/quality/memory/classifier hashes unchanged','GPU isolation and thermal protocol','Quality scoring evidence and complete causal-cutoff decision logs'],
        next_experiment_automatic=False)
    (P/'module30_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(gate,ensure_ascii=False))
if __name__=='__main__':main()
