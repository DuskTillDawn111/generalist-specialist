# 可见前缀与剩余全过程滚动预测 v1

本次在独立模块 `rolling_visible_queue_v1.py` 实现并实际运行通用 `FullHorizonModel` 后端的
两次可见切点续算，不改任何冻结模型或主 v54。主 v54 尚不支持滚动续算，该事实由接口显式返回
`MAIN_LIFECYCLE_CONTINUATION_NOT_IMPLEMENTED`。没有 GPU 实验或独立物理精度结论。

## 主路径审计与设计边界

| 已有组件 | 可用事实 | 不能自动补出的状态 |
|---|---|---|
| observed_state v2 | 连续反馈序号、parent reader 可见时间、START/PHASE/END/READY、完整公开到达 | 未观测kernel游标；没有RESULT不等于handler仍未结束 |
| v54底层 LifecycleDagPhaseModel v21 | 仅从原phase start运行的单次预测 | 明确拒绝checkpoint/pause，scheduler_checkpoint_supported=False |
| v10 phase-DAG checkpoint | 自己的模拟进度、旧scheduler状态、完整预测前缀重放一致性 | 不是物理观测；不能迁移到v54的oracle、mark clock、累计SAVE budget |
| 通用 FullHorizonModel | 支持当前明确segment cursor、剩余全arrival schedule、FIFO、service repricing | 不能由host age估GPU进度；额外batch/gate/handoff身份不能省略 |

后续主引擎真续算须完整保留 running/active/node残差、所有事件heap、FIFO、receipt/READY、
oracle持有者/restore/history/handoff、原累计SAVE budget和mark clock，并将预测隐状态与观测事实分开。
主路径还缺这些状态导出/恢复及可见前缀条件化；本次不通过复制旧检查点规避。

## 当前可执行接口

1. `new_session` 绑定原五字段完整公开请求、原route、全部future arrivals、alias、执行身份和原window/drain。
2. `advance` 只接受序号连续、reader receipt不晚于cutoff的事件；`advance_runtime_records` 先查可见时间，
   才访问worker response body，并验证历史前缀未改变。已知公开schedule变为当前arrived的部分按原顺序入账。
   这是declared schedule，不是未来实测到达率。旧session不修改；frozen模式不能越固定cutoff读取反馈。
3. `remaining_state` 将请求分为completed-ready、busy、arrived-waiting和future。completed需可见END与READY；
   busy需同执行身份、同冻结profile、当前时刻的直接segment progress receipt；仅凭START/已过时间返回UNKNOWN。
   START不可见时，还需当前“确实仍等待未开始”的证据，避免把延迟START误当排队。END后READY不可见的handoff仍UNKNOWN。
   未开始/未来请求使用完整冻结solo/resource work；busy只扣直接观测的segment剩余量，完整future arrival envelope保留。
4. `forecast_suffix` 实际调用原 `FullHorizonModel.forecast` 执行所有剩余请求和全部future arrivals，原路由不变，
   验证剩余cohort完整。该后端把续算起点记为START，因此适配器将busy请求的原可见START恢复到记录中，
   只用于原排队/服务边界计量，预测END不变。原completed prefix与suffix一起重新计算原Q/T/L/P/M/E。

当前通用后端仅支持同执行身份的shared、unbatched segment模型；不能把它当成主v54物理服务模型的替代。
当前无连续可见prefix energy/owned peak契约，E及full_U保持UNKNOWN，不用0填补；P95和队列可单独数值计算。
receipt校验保证输入身份一致，不是传感器真实性认证。当前无接入主运行时的直接segment progress生产器，
CPU fixture证据需显式opt-in且产物保留该来源，不能改称GPU观测。

## 实际CPU续算例

公开请求：a在0ms进入s，b在75ms进入s，c在125ms进入g；隔离solo work分别100/100/50ms，
CPU fixture规定两owner共跑时各用一半参考速率。50ms切点仅a的START可见，并有a剩余50%的直接fixture游标。
实际续算得到a/b/c完成100/250/225ms，原排队时间0/25/0ms，完整三请求P95=167.5ms。
同可见前缀换为a剩余25%的fixture游标，实际续算得到a完成75ms、完整P95=145ms，说明反馈进入了重新执行。

210ms切点再消费新的可见a/b END+READY及c START，a/b退出剩余工作，c的当前剩余60%游标驱动新续算：
c完成240ms，三个原请求仍保留，原排队时间0/45/0ms。未来completion body为可访问即报错的对象时，
未到reader receipt也不会被读取；主v54对象、无busy游标、无waiting证据、过期/异profile游标均不能数值续算。

`python -m unittest tests.test_rolling_visible_queue_v1 -v`：6项通过，覆盖上述真实模拟器执行、
二次prefix增量、future cohort、原START/队列/P95、等待证据、未来body隔离、序号/回退/冻结cutoff拒绝。
这些是解析CPU场景，不是物理预测准确度、主v54 rolling已完成或最终六项收益证据。
