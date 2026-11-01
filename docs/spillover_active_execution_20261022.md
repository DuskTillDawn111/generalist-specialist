# 当前推进目标与执行链（2026-10-22）

最新推进（北京时间23:56）：自主开发搜索与同域DFS核对已经完成，详见
`docs/spillover_autonomous_search_closure_20261022.md`。每seed完整域7508标识，
两个seed及均值目标均由有界DFS与完整枚举同优，当前赢家为text×2＋G2×1＋OCR×1、
文本不分流；有界DFS单seed本轮约3.62×／2.43×，MILP三次仍限时未闭合，未声称其优势。
当前结果仅是未验证开发评价函数的有限域结果，不是七池物理最优。

为区分完整搜索前两名，新的`search-top-two-prequential-development-v1`八条GPU
校准／配对实验已启动（109–112，AB／BA／BA／AB）。rank1为不分流，rank2为1/32分流，
未将rank2冒称算法赢家。运行状态在`execution-v1/progress.json`，终态后自动审计并汇总。
这是新批次，下面22:53记录的八条107／108实验仍为已完成的历史批次。

最新状态（北京时间22:53）：当前八条GPU开发实验全部完成并独立审计通过，汇总位于
`text185-eight-terminal-audit-v1`。原single supervisor退出且旧progress未更新，root已在新目录
恢复CPU汇总／图表，原证据未覆盖、GPU未重跑。107四方案最好为text2混合不分流，
108最好为text2混合1/32分流；两次平均后者最高，但尚非稳定收益或算法／DFS最优性证明。
single107/108 U=-2.681685343/-1.700739204，两个种子均低于text2细分流。
以下调度描述为阶段历史，以最终八条审计为准。

最终目标是在冻结七模型池、固定静态任务负载、原六项总效用和原稳定性规则下，
由算法自由选择部署、副本数和任务路由；希望找到有真实收益的“专用多副本向通用分流”，
并用同域MILP／有界DFS／plain DFS核对最优及搜索效率。混合、纯专、纯通都保留资格，
结论由原总效用比较决定。

## 正在执行的有限批次

1. `text185-fine-development-v1`：text1.85 rps，image/asr/ocr各.01，H600，warmup30。
   部署text×2＋G2×1＋OCR×1，文本比较不分流与1/32分流；107正序、108反序。
   四条均完成且独立核验合格；107分流ΔU=-.033757532，108分流ΔU=+.241645449。
   108文本P95从5461.72降至4124.62ms，但两seed方向不同，未证明稳定收益。
2. 四条终态后先独立核验原trace、输入、SWRR、native质量、原评价器、资源与worker回收。
3. 通过后串行运行已冻结`text185-strong-controls-v1`两条：同负载、同107/108到达与输入，
   text×2＋image×1＋OCR×1＋ASR×1，全部任务去各自专用。
4. 汇总三个具体方案、两个开发种子的六项实测损失、总效用和稳定性；排序只针对已观测方案。
   续跑只由完成与证据状态控制，不根据收益正负跳过条件，不自动重试。

这条链已由`staged-continuation-v1`启动，执行状态写入其
`execution/progress.json`。fine和strong六条已全部通过审计，supervisor状态COMPLETE。
纯专107/108 U=-1.150695406/-2.812254623；107三者最好为混合不分流，108为混合分流。
汇总在其`execution/all-six-comparison`，不自动消耗保留验证或确认种子。

## 与GPU收集并行的子对话工作

- 路由与搜索：细分流域已统一到三种求解器；完成107反转诊断，并准备上述串行续跑工具。
- 模型数据：精确核验共享worker与原画像的请求／协议等价性，再用可辨识的低维形式准备
  局部条件服务模型。按run留一仅是开发诊断，不替代独立验证，不用真实未来overlap作特征。
  `context-identity-equivalence-v2`已闭合G2/OCR/image的部分精确身份；
  `context-design-inventory-v4`严格可用15113请求、86个run-path块，设计rank10/12，
  缺G1/G3方向。未匹配与失败记录仍保留；不再为已闭合身份重复GPU测量。
  `local-context-development-v1`整run留一log-RMSE从.692降至.422，但封存的108
  几何尺度FIFO预测方向失败；不据此选路或正式算rho。预测已含逐输入b与精确aliasFIFO，
  不能将失败简单解释为完全没有考虑输入分配；待补的是条件服务随alias组合的变化。
  `context-resource-candidates-dev-v1`的19-run能耗候选最坏留一误差超原阈值，保持开发状态。
  `alias-context-development-v2`检查了计划alias/partner工作量，未同时改善整run与整deployment
  留组误差，不采用新增复杂度。`strong107-resource-prequential-validation-v1`记录未见部署
  数值外推探针：M_full低估1.47%、E_full组合低估21.14%，不升级模型资格。
  `strong-fine-tail-review-v1`定位strong108长尾集中于60–90秒服务减速及后续队列，
  不能直接归因背景活动。`swr-period-exposure-v1`确认偶数幂分母与text32池导致G2仅2输入暴露，
  邻近奇数分母改善覆盖；尚无新份额GPU数据，当前域与队列不变。
- 原画像缺口：5个上下文覆盖9个缺口的补测包已完成CPU准备，当前仍为UNKNOWN。
  独立复核补测执行合同，安排在当前有限批次之后，GPU不并发。

另已准备`text185-single-specialist-controls-v1`：text×1＋G2×1＋OCR×1，文本各.925 rps，
其余路由同fine。这是同107/108的强混合替代，正式proxy maxrho=.5281511663、
safe驻留proxy22265.96MiB，已通过冻结runner逐请求CPU核对。
它用于检验第二个文本副本是否必要；同时改变副本数与分流份额，不能孤立归因副本数量。
该两条现已通过root review及8项CPU门禁测试，`single-control-continuation-v1`已启动，
前序六条审计／资源／回收及supervisor COMPLETE门禁已通过，现正串行执行single107，
随后single108，最后汇总八条。
prepare阶段记录的“未排GPU”是历史状态；实际执行状态以续跑目录的`execution/progress.json`为准。

## 当前尚未完成的证据

当前GPU点是开发配对，尚非算法自主选出的七池最优。新耦合模式仍是安全占位，完整预测UNKNOWN；
旧失败transfer模型不激活。局部模型和七池覆盖须分别说明，未覆盖组合不能被当成不可行删除。

后续先冻结新预测器及声明域，再用81/82独立检查真实误差和排序；随后进行相同有限域、
相同完整评价器下的自由搜索和DFS核对，同时记录完整评价次数及冷进程实时时间。
比较强替代方案后再使用71–73最终确认。真实GPU有限比较与预测函数有限域最优分别报告。

若实测始终不支持分流收益，记录当前条件不支持；不能通过改质量、权重、稳定性或挑种子制造结论。
实验详情见`docs/text185_fine_development_results_20261022.md`，反转诊断见
`docs/text185_seed107_component_diagnosis_20261022.md`。
