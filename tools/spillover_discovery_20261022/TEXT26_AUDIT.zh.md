# Text2.6 四计划审计

检查输入齐备情况：

```bash
python tools/spillover_discovery_20261022/audit_text26_development.py --check-only
```

采集后运行（输出目录必须尚不存在）：

```bash
python tools/spillover_discovery_20261022/audit_text26_development.py \
  --output artifacts/static-spillover-study-20261022/text26-development-audit-v1
```

默认输入 `text26-development-v1/experiment-packet.json` 及其 `runs/` 下四个预声明计划的所有 attempt 目录。程序核验四计划恰为text n2份额4/7、n2份额1/2、n1份额1/2、n3份额4/7；G2和OCR各1副本，text2.6、背景image/ASR/OCR各.01、seed51、H600。只审计该开发目录，不读取其他验证或确认数据，不启动GPU。

成功条件调用未修改的冻结execution-source-v5 `_import_source`，重新检查原始包绑定、执行身份、硬件/检查点、实际输出质量与完整效用。失败条件另做身份/transport/队列cohort核验并调用原 `evaluate_run` 保留评价；不绕过稳定性、不改变阈值、不激活模型。身份不可信的观测仍保留为失败，正式U为空。

仅当双方都严格导入成功、原评价可用且同一有序任务/输入/到达trace，才输出实测ΔU。所有失败、稳定性不通过或不完整条件都保留，`U=null`，不能用诊断U比较。原六项算术及稳定性证据可以供排错查看，但程序不据此排名。

输出audit.json、中文README、每尝试逐请求CSV及manifest。CSV包括实际模型输出、质量错误分子/参考分母、输入身份和服务/等待/端到端时间；完整JSON保存任务/路径/别名的服务与等待均值/p95、实际质量、六项成本及贡献。

退出码0表示所有明确存在的尝试已审计，不代表四计划都有效；2表示缺少输入，状态INCOMPLETE；3表示原包或审计合同失败。完整执行遇到部分输入缺失时，仍保留已存在尝试的审计，状态保持INCOMPLETE，不补造剩余运行。单个坏raw不会使其他已有尝试消失。`--check-only`只检查路径，不读取raw内容，不评价。

四个有限开发计划的比较不能称全池最优、独立验证或统计优势。稳定性设置和生产数学保持不变。

本次正式审计使用text26-development-audit-v2。v1保留原样：旧发现逻辑误将attempt1.log普通文件当作attempt目录，产生假缺失；修复后仅接受数字结尾的实际attempt目录，并在v2重新审计同一批原始文件。test_text26_audit_attempts覆盖此情况。原始运行与冻结源均不修改；v1的INCOMPLETE不代表四条数据缺测。最终汇总须显式传入`--text-audit .../text26-development-audit-v2/audit.json`。
