# Module 24B：HIGH 独立真实效用验证

Gate：**HIGH_NO_UTILITY_GAIN**。At least one comparable independent pair failed the required positive-utility criterion。

本轮只验证 HIGH。LOW/MID 均为自主 G-only 不构成失败；Module24A 原 gate 和全部历史文件保持不变。未执行 Module24C。

## 冻结与 CPU 复核

全部 288 个副本部署枚举，其中 71 个合法部署参与自由路由竞争，54110 次效用评估。复核通过=True，最优部署/路由与24A一致，ASR G份额=13.1833204618%，原预测U=-0.7765777086，同驻留S-only预测U=-0.8078013113，预测ΔU=0.0312236028。不是数值并列、零请求残差或三阶段强制约束。自由连续路由搜索有多种子/端点/局部精化，不声称全局数学证明。

驻留：Qwen×1、Whisper-large-v3×1、RoBERTa×2、ConvNeXt×2、PARSeq×2；8个独立worker进程。QPS：Text=4、Image=2、OCR=5、ASR=1.52、LID=.03、VQA=.2；预算34567.5534 MiB。Text/Image的原优化小份额原样保留，两臂只有ASR路由不同。质量按既有硬准入及运行期输出有效性检查，不新增质量奖励。

U=-.25*T_loss-.30*L_norm-.30*P_norm-.15*M_norm；沿用20A各任务参考值和固定offered权重。独立排序样本原为0、双副本共置支持缺失；此次结果不回写预测器，也不证明跨组合泛化。

## 双副本共置预检

通过=True。显存：{"postwarm_idle_mib": 30002.0, "postmeasurement_idle_mib": 30556.0, "postwarm_safe_mib": 33327.84036555694, "postmeasurement_safe_mib": 33943.25345676814, "sampled_owned_peak_mib": 30556.0, "budget_mib": 34567.5534, "passed": true}。错误：[]。GPU释放=True。

| 模型 | 预测g(2) | 实测g(2) | 单活跃/双活跃完成容量QPS |
|---|---:|---:|---|
| roberta_sst2 | 1.9011185575645404 | 1.7818151663174815 | 242.89700150938546/432.7975611424632 |
| convnext_tiny | 1.850654107747522 | 1.8593156254190344 | 195.49828491309628/363.4930158815422 |
| baudm/parseq:parseq@v1.0.0:parseq-bb5792a6.pt | 1.807922326459751 | 1.794546646274835 | 98.79915103865494/177.2996851512191 |


扩展比使用固定8副本驻留、单家族单活跃/双活跃的各10秒closed-loop诊断，其他模型此时闲置。不能把它当完整HIGH并发下的饱和容量；后者用固定120秒Poisson负载检查完成、busy、队列增长和积压。预检不是独立效用样本。NVML峰值为0.5秒采样峰值，不是分配器瞬时峰值。


## 正式独立配对

正式运行=2/6，合法可比配对=1/3；固定新种子24201/24202/24203，臂顺序A-B/B-A/A-B，轨迹和输入在预检前冻结。warmup10秒、measurement120秒、drain最多60秒，未筛种子、延长窗口或重试结果。

| 配对 | Proposed U | Control U | ΔU | 状态 |
|---|---:|---:|---:|---|
| 1 | -1.625463080094763 | -1.518752523672681 | -0.10671055642208183 | COMPARABLE |
| 2 | None | None | None | NOT_EXECUTED_OR_INCOMPLETE_PAIR |
| 3 | None | None | None | NOT_EXECUTED_OR_INCOMPLETE_PAIR |


high_proposed_rep1：全部完成 1531/1531，窗口内完成 1527，实际ASR G/S完成=20/154，G份额=11.494253%；safe memory=33241.193757 MiB，采样实际峰值=29924.000000 MiB。


high_control_rep1：全部完成 1531/1531，窗口内完成 1526，实际ASR G/S完成=0/174，G份额=0.000000%；safe memory=33238.972049 MiB，采样实际峰值=29922.000000 MiB。


配对1的20秒匹配时间块条件敏感性95%范围=[-0.17894853354322127, -0.05204527084244898]，有效重采样=992/1000、缺少任务=8/1000。此范围不是独立重复的效应置信区间。


独立配对平均ΔU=-0.10671055642208183；paired t95 CI=None。单个配对仅有每臂一次GPU会话，不具备独立重复CI；CSV另有20秒匹配时间块敏感性区间，明确限定为条件描述，不增加独立样本数。n=3仍不能验证正态假设，即便三次方向相同单侧符号检验p=.125。稀疏LID预期每窗口3.6请求，P95的有限样本不确定性必须保留；缺少任务、持续拥塞或不完整运行不计算正常U。

每任务和pooled mean/P95、offered/accepted/completed QPS在system_utility_comparison.csv；busy/worker queue/host queue/backlog和family聚合队列在worker_stability.csv；效用分项和配对分项收益在utility_decomposition.csv。无共享家族调度队列，双副本为独立FIFO队列。


可比配对收益分解均值：{"T": 0.00016339869281045694, "L": -0.030519393755143676, "P": -0.07634492063734288, "M": -9.64072240577929e-06}（T吞吐、L任务归一化mean、P任务归一化P95、M显存）。收益排序不能只看pooled latency。


第一组负收益触发预先冻结的停止规则，后两组未执行。该真实负结果足以使本轮成功条件失败，但单个独立配对不能估计总体效应的独立重复置信区间，不能宣称跨会话显著负收益。


## 后续决策与完整性

是否值得进入独立Module24C：False。当前证据不支持推进该模型池正式后续收益验证。

全部16039个历史文件metadata不变，所有24A输出和读取的20A科学输入SHA256不变。复现源码module24b/，新原始GPU/CPU日志完整保留。已完成阶段拒绝静默重跑。
