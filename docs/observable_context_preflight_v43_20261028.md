# v43 完整四候选 CPU preflight

`observable_context_preflight_v43.py` 复用冻结 v38 的 `ObservedForecast` 与 `save_forecast`，保留完整四候选循环、每组原 57 请求和五百万 event 上限。修改只涉及进度文件：每个完成候选使用独立、exclusive create 且 fsync 的 `progress-report-{candidate}.json`，最终 `report.json` 只创建一次；scheduler 诊断路径也按候选分开。没有 resume、跳过已执行候选、替换 UNKNOWN 或放宽 guard。原 v38 源码与旧失败报告保留。

低成本回归测试通过，覆盖第二候选以及四候选累计进度独立写入；重复写同候选仍拒绝。实际 CPU 执行采用 BLAS/OMP/MKL/NUMEXPR 单线程、原 v38 configuration 与 v11 prepared01，输出 `artifacts/predictor-observable-context-v43-20261028-preflight02`。运行 441.56 秒，四候选完整退出，source bytes unchanged。

balanced、skewed、spill_low、spill_high 全部为 `UNKNOWN / UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION`。每组仍包含完整 57 个公开请求，但 queue 时钟停在 0：已登记 1 个，未来 56 个；scheduler query 数分别为 1、2、1、1，transition 数均为 2，oracle decision 数均为 0。starts、finishes、六项 metrics 均为 null，不能由此计算 P95、ΔU 或候选收益排序。

四个 gzip 结果均完整解压校验 SHA 与长度；四份完成进度和唯一最终报告均保存。证据包在 `artifacts/predictor-observable-context-preflight-v43-20261028-cpu01`。此次只修复写进度导致的第二组流程中止，实际模型的同 context 再获得支持缺口仍保留为 UNKNOWN。未 seal，未启动 GPU。
