# 资源控制 CPU 实现与接续说明

日期：2026-10-24，Asia/Shanghai。用户本轮要求先写软件，不在 GPU 上实验。

2026-10-27 接续：[全过程到达与执行段预测器](full_horizon_predictor_20261027.md)已实现并成为资源 Dispatcher 默认路径。下文保留 10 月 24 日交付记录，其中“准入预测尚未实现”描述由本次接续替代；GPU、权重合并与硬件隔离边界仍然有效。

## 本轮交付

1. `gsdeploy/resources.py`：统一资源配置契约及逐 worker 绑定。支持默认共享、MPS 计算上限、静态 SM 集合声明、Green Contexts 集合声明和整请求准入。检查全部驻留 alias、单设备、模式字段、SM 重叠／越界、并发上限及后台服务覆盖。MPS percentage 只记为客户端上限，不作互斥分区证明；本实现仅接受整数百分比。
2. `gsdeploy/runtime/admission.py`：实际可执行的非抢占整请求准入。发送线程取得许可后才提交；RESULT／ERROR、发送失败、worker 退出归还许可；关闭唤醒等待线程。不读取未来请求、不抢占已执行的 generate、不声称是 kernel 调度。支持每 alias 的在途数上限与可重叠资源组的并发上限。
3. `gsdeploy/resource_study.py`：CPU 实验包准备、输入与源码哈希、候选／资源／执行上下文身份，以及从原设计完整重建的独立校验。入口仅有 `prepare` 和 `validate`。
4. `tools/resource_control_20261024/policy.py`：接回既有六项条件效用策略，保持背景 SWRR 与文本分流开关。服务／功率提供者必须绑定当前执行上下文，否则为 UNKNOWN。P95 使用 1/6 权重；全未来负载、物理显存差和严格净收益仍未得到保证。
5. 能耗记录加入可选资源配置身份；旧调用不带该参数时保持原键。显式新资源域不继承旧预测／最优证书，正式评估不会给尚未验证的新原生协议颁发有效 U。

## 软件控制与硬件控制的完成边界

|方式|本轮完成|仍待完成|
|---|---|---|
|默认共享|逐 worker 环境绑定；显式研究清除继承的 CUDA_MPS_* 配置|设备实际状态仍需记录；清除子进程变量不能改变外部服务器设置|
|整请求准入|实际 host 准入、释放、关闭与等待计时；CPU 子进程行为已测|准入感知预测器、GPU 兼容性和收益验证|
|MPS percentage|每 worker 上限及 session 环境准备、冲突校验|真实 MPS session、平台支持与限制生效核验；当前受控原生启动拒绝执行|
|静态 SM／Green Contexts|集合／容量／不重叠的声明与校验|原生 context adapter、实际分配粒度与真实 SM 绑定；没有把声明冒充实际分区|
|同权重共同 batch|保留独立后端研究项|尚未实现，物理副本定义与 KV／输出正确性需另测|

10 月 24 日原策略的 `CoupledEventModel` 假定各 alias 队首同时进入执行，当时整请求准入候选返回 `HOST_ADMISSION_FORECAST_NOT_IMPLEMENTED`。10 月 27 日已用 `FullHorizonModel` 替换该占位接口：有既存请求时仍要求真实准入状态和匹配的服务／显存／功率证据，不沿用错误的同时起跑假设。

## 当前待实验设计

配置：`configs/resource_spillover_study_20261024.json`。复用已有七模型目录、真实 worker 注册及 160 条输入文件，不改旧实验包。

- 固定驻留：text×2、generalist_2×1、ocr×1；完整七模型目录保留在有效配置中，本批不宣称完成全池自由搜索。
- 负载：text=1.85 rps，image／asr／ocr 各 0.01；Poisson，H600，warmup30，drain300。
- 四个候选执行配置：共享；所有 worker 合计最多两个完整请求、每 alias 最多一个；MPS 上限 30/30/30/10；MPS 上限 35/35/20/10。
- 顺序中的四个百分比依次表示 text#0、text#1、generalist_2#0、ocr#0；只是两个开发假设点，既非测得容量，也非已选最优分配。合计 100% 不代表 SM 物理互斥。
- 各配置都生成允许／禁止文本分流两臂；所有背景服务保持不变，双方拥有同样的候选配置集合。第二个 seed 反序执行设计，不省略失败或不利条件。
- seed501/502 是未预留的开发草稿，必须在新的可执行包中核查历史占用／保留集；当前不解锁 GPU。
- 共 4×2×2＝16 条 NOT_RUN 条件；全部 U、服务、显存、能耗预测保持 null。没有启动队列，没有重试或运行任何 GPU 命令。

这些条件使用相同种子的完整到达与输入序列哈希。分流开关不是固定分流比例；它是在线策略候选的开关。实际输出、分流数与资源使用只能在后续真正运行后记录。

## 运行 CPU 准备和校验

在项目根目录：

```bash
python -m gsdeploy resource-study prepare \
  --design configs/resource_spillover_study_20261024.json \
  --output artifacts/resource-control-cpu-20261024/experiment-packet.json

python -m gsdeploy resource-study validate \
  --packet artifacts/resource-control-cpu-20261024/experiment-packet.json
```

已有输出拒绝覆盖。设计、输入或相关源码改变后，原包校验失败；使用新的输出路径重新生成，不覆盖冻结产物。独立模块入口 `python -m gsdeploy.resource_study` 等价。

准备及校验不启动 worker／MPS、不查询 GPU、不加载模型。CPU 测试使用 `gsdeploy.backends.mock` 或不导入模型的 Python 子进程，不执行注册表里的原生模型命令。

## CPU 验证

新测试集中在 `tests/test_resource_control.py`、`tests/test_resource_study.py`、`tests/test_request_admission.py`，覆盖模式／份额／SM 合同、环境继承和篡改、同 cohort 配对、源码与输入改动、P95 权重的实际评分影响、UNKNOWN 回退、禁止原生启动及实际 host 准入。

准入行为测试使用两条真实 CPU mock worker 进程，验证合计上限为 1 时六条请求的 transport 提交区间确实串行，保留共同原到达时刻，等待记录落在提交之前；另检查关闭唤醒和各类非法配置。

本轮最终 CPU 回归共 127 项：126 项通过、1 项按原条件跳过；其中新增资源配置、实验包和准入测试 45 项全部通过。另通过 Python 编译与 `git diff --check`。该记录不代表 GPU 性能、硬分区或新域最优性验证。

## GPU 恢复后的接续

1. 先在单独的平台核验协议中检查可用硬件、工具链、MPS／SM 分区／Green Contexts 与实际后端兼容性。不要运行这份 CPU 草稿冒充已验证原生协议。
2. 按核实的资源能力完成原生 adapter 和观测记录，再做有限独立／共同活动画像。每个受控资源模式的新服务、M/E 和持续供给证据分别绑定；不能把全卡速度按份额线性缩放。
3. 为整请求准入实现与实际 FIFO／许可规则一致的事件预测，再检验全体请求端到端效用。继续保留原稳定性，不遗漏主机等待或背景义务。
4. 冻结真实可支持的有限资源域及预测器，重新核对有效界、同域 DFS 和强不分流对照；随后才独立确认收益。

目前没有新的 GPU 收益、隔离生效、容量通过或全域最优结论。
