# 五开发点 CPU 审计调用

检查五个明确输入文件是否齐备，只读取路径存在性：

```bash
python tools/spillover_discovery_20261022/audit_five_development_points.py --check-only
```

五条测量结束后，用一个尚不存在的输出目录运行：

```bash
python tools/spillover_discovery_20261022/audit_five_development_points.py \
  --output artifacts/static-spillover-study-20261022/five-development-point-audit-v1
```

默认输入为 development-asr18-v1 中原始两条和 development-small-spillover-v2 中新三条，均为 attempt1。可通过 `--first-development`、`--small-development`、`--basis-config` 显式指定存储位置。脚本只使用预声明的五条路径，不扫描或等待其他运行，不读取61/62及最终确认目录，不启动GPU。

退出码：0表示五点审计完成；2表示输入不齐；3表示严格导入或合同失败。`--check-only` 文件齐全仅表示存在，不代表其内容通过严格审计。正常调用若缺文件，输出 `INCOMPLETE`、明确缺失列表，不读取部分raw、不拟合、不补造结果。输出目录禁止覆盖；重试请换新版本目录。

完整审计使用冻结 predictor-source-v1 的 `_import_source` 核验原始包/文件指纹、执行身份、硬件/检查点、实际队列/质量/能耗/完整效用，再检查固定负载、部署、路由份额、seed51、H600和开发用途。各实测指标重算，缓存的实际逐任务时延不作为审计依据。

9参数模型仅使用初始 ASR1.8 的0/25%分流两条，四条共同路径各两个倍率端点，G2 ASR仅一个倍率。新增三条不拟合参数：1.8/12.5%为插值检查，1.9配对为工作负载迁移检查。完整外部诊断保留冻结预测 Q/M/E；另保存使用实测 Q/M/E 的 FIFO 算术诊断以隔离服务误差，两者明确分开。所有诊断正式 `production_U=null`、`production_eligible=false`。

产物含 audit.json、README.zh.md、manifest.json。报告保存五条负载/路由、六成本与贡献、同负载四组配对差、逐任务服务/等待/均值/p95、路径服务与冻结/经验预测误差。只有一组开发seed，不宣称独立验证、统计优势或全域最优。
