# v6 同域独立训练 capture02/03 的 CPU 准备

新增工具 `profile_training_repeats_prepare_v1.py` 只做准备和审核，不提供执行操作，也不被冻结的 v6/v3 画像执行依赖导入。它要求原 natural/kernel 两套完整训练审查 PASS，并核验 26 jobs、50 measured、18 priming、原 raw/plan/context/attempt/执行 receipt；原训练 run 不会重启。

原 packet 下新增 `training-repeat02/{natural,kernel}` 和 `training-repeat03/{natural,kernel}`。每个 target 的 plan/context 与相应原 mode 逐字节相同，pair 仍绑定同一个原不可变 training-pair。该布局满足冻结 v6 `run.parents[1]/smoke` 合同，真实 standard execution_gate 检查原两份 PASS smoke 及 supervisor 完成证据，无旁路或新执行后端。

四个 target 各有新的标准 supervisor-v3 manifest、capture-metadata 和未来独立 attempt/raw/执行目录。完整执行 source closure 仍由原不可变 snapshot、每个 target 的 context/source receipts 及新 supervisor manifest 绑定，避免复制和混淆已经冻结的源码。准备工具自己的静态 import closure 另存为 preparation_sources，原画像 closure 不变。

计划顺序是 natural02 → kernel02 → natural03 → kernel03，对应全 campaign 执行序号 3–6。这个固定顺序没有随机化或热状态平衡，不能据此消除时间/热漂移。每个 capture 都会重新启动 collector 与四个 worker，并保持原 priming；fresh process 不代表冷 GPU。准备时实际执行序号、时间、温度均为 None，实际 GPU 冷态未认证。原 NVML raw 记录功率而未记录温度，温度不可补造。natural 仍带 CUDA event/query/stdout/phase 仪表，kernel 带 nsys；两者公开工作配对，不能称为完全无扰动的时间配对。

no-retry 作用于每个独立 target：失败/部分完成必须保留，不可在原目录重新执行。capture02/03 是模型设计要求的独立重复，不是已有成功或失败 run 的重试。准备工具拒绝已存在 target 或执行输出，审核遇到 attempt/raw/trace 会被冻结 v6 gate 拒绝。

标准预算不变：GPU 启动前 18GiB，运行中 13GiB floor、6GiB 增量、2400s timeout、同一 experiment lock 和 owned-process cleanup。CPU 准备不会因空闲不足启动线而伪称已执行，也不降低 GPU 门槛。最终启动仍由 root 审核后调用标准 supervisor；本次没有 GPU 操作。

6 项 CPU 合同测试通过：四个独立目录/相同 bytes/标准 gate、既存失败 target 拒绝、缺少 smoke PASS 拒绝、原 review/raw 绑定变化拒绝、已有 attempt 后审核拒绝、新工具不在冻结执行 closure。fixture 仅替代历史执行完成 receipt；真实 v6 source/pair/mode 和 supervisor manifest 审计均运行。正式新 packet 会再次使用真实原执行证据审核。
