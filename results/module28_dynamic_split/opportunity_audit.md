# Module28 Stage A：CPU-only 机会审计

读取 Module19–27 原始结果，逐请求重建见 historical_target_request_audit.csv；六任务及全系统 mean/P95 重算见 historical_system_audit.json，worker 服务/等待及质量可用性见 historical_worker_paths.csv。无独立 raw 的 Module20/23/25 保留原摘要与链接关系，不制造新观测。

Module27 11.5QPS S-only：6806个LID中1010个(14.84%)到达时S≥8个已入队未完成请求且G没有未完成请求；737个(10.83%)满足S≥10。真实native handler区间确认这些G空队列窗口也没有G执行。平均GPU利用率不作为模型空闲证据。S服务mean69.13ms，等待294.02ms；固定10%后S服务74.60ms、等待318.23ms，G服务383.52ms、等待190.90ms，VQA等待20.49→161.16ms。

Module26 ASR S等待约1s、G空队列短时窗口也存在，但三次18%固定分流负收益且G ASR约1.3s、S服务约19%膨胀；本轮不增加ASR候选。保留同一Module27 LID池的一个保守队列策略，不再调整固定比例网格。选择11.5因为同机实测尾部排队/G空队列窗口更充分，S-only和10%曾合法稳定，所有六任务正QPS。

可支持：真实S排队、G host-visible空队列和retrospective native闲态、固定分流服务膨胀和背景等待损失。机会满足前两项，第三项仅可形成开发估计，需要GPU补测；第四项仅有同池历史硬约束先验，新动态尚未认证。

UNSUPPORTED：未执行路由的逐请求反事实完成时间或ΔU；没有worker receipt ACK，host send不得冒充接收；native START/END只能回顾，不能作为当时已知信息；共置物理GPU/CPU/热态因果无法唯一分离；预测proxy不构成系统效用证明。新GPU轨迹是检验这些证据缺口的合理有界补测。

参数、模型、负载、效用、质量、预算、运行顺序、稳态/突发及独立验证门槛全部见 frozen_candidate_manifest.json，冻结后不得依据结果调参。G-only容量不合理，预先排除。正式部署优化器不修改。
