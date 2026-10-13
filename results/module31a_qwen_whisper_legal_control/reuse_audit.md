# Module30 审计与复用

Qwen C1 s_only/split05/split10 实际 ASR G/S 为 0/595、28/567、53/542，实际比例 0%、4.705882%、8.907563%。前两窗初版遥测 check 共用采样长锁，合法 U=NULL；split10 使用 v2，U=-1.2243815965，跨版本 ΔU 禁止。旧数据不参与本次效用对照。

本次复用 C1 seed=301001 的六任务输入/到达/路由 uniform 轨迹。ASR=1 QPS，Text/Image/OCR/LID/VQA=0.2 QPS。Text→RoBERTa-base SST-2 FP32，Image→ConvNeXt-tiny FP32，OCR→PARSeq v1.0.0 严格加载 FP32，LID→Whisper-small FP32，VQA→Qwen2.5-Omni-7B BF16；ASR 未分流部分→Whisper-small，分流部分→Qwen。五副本单卡、34567.5534 MiB 预算不变。检查点全文件 SHA256、脚本快照、依赖、质量准入和精确轨迹见 manifest。

复用 SafeMonitorV2；依赖的 SafeMonitor.sample 仅作串行 I/O 和 runtime.lock 一致快照，SafeMonitorV2.check/set_active 覆盖旧控制方法。通过的 2118.546294 秒长压与故障注入证据均有哈希。GPU 型号、UUID、驱动、总显存与原冻结一致。

效用：module20.optimizer.components 与 module26.common.arithmetic 独立复算；Q=0、T=.25、L=.3、P=.3、M=.15；六任务按冻结到达率加权，参考延迟不变，完整 cohort/600s 和窗口内吞吐不混淆，预热/path probes 不计入。原稳定性、质量、资源、+0.01 门槛不变。SLO 使用历史稳定/完整/排空、资源与质量约束；未新增绝对时延阈值。单组三窗仅探索，原独立统计支持规则不能由一次配对轨迹满足。

Git 元数据不存在，commit=NULL，源码内容 SHA256 和副本代替身份冻结，明确报告此复现限制。Module14F–30 结果及 Gate、权重、正式优化器原文件不修改，Stage D/31B–31E 不执行。
