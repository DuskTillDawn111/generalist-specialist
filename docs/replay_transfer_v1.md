# 零再拟合的新负载迁移验证

`gsdeploy.replay_transfer` 为独立模式 `replay_transfer_v1`，schema为 `gsdeploy.replay_transfer.v1`。它保留原 `calibrated_replay_v2` spec 的内容、版本和状态不变，另建新包络。原逐输入表、全局服务比例、NNLS功率、startup、memory和已见deployment内存映射逐项对照原配置receipt；新观测永不更新参数。

正式状态是 `TRANSFER_UNVALIDATED` → `VALIDATED_ON_NEW_WORKLOAD` 或 `TRANSFER_VALIDATION_FAILED`。未验/失败配置的公共预测 U 为空、无正式bounds、SearchSession拒绝。私有验证预测只有 `diagnostic_partial_U`，U与eligible继续为空/False。代码、源配置、source manifest（若提供）、新工作负载、quality、GPU/checkpoint、baseline/memory/capacity、评分合同、验证packet/run/report均绑定。

最小调用流程（在冻结本次新源码后生成新artifact，勿在旧目录写入）：

```python
from pathlib import Path
import json
from gsdeploy.config import SystemConfig
from gsdeploy.replay_transfer import (
    prepare_transfer, declare_validation, predict_validation,
    validate_transfer, activate_transfer,
)

raw = prepare_transfer(
    ".../surrogate-v2/system-config.json",
    {"asr": 1.8, "text": .05, "image": .05, "ocr": .05},
    source_manifest_path=".../surrogate-v2/manifest.json",
    development_seeds=(51,), validation_seeds=(61, 62),
    confirmation_seeds=(71, 72, 73),
)
config = SystemConfig(**raw)
# 保存raw；仅可用诊断预测筛查，不能标作已验证全池最优。
diagnostic = predict_validation(config, deployment, routing, horizon=600, seed=61)
```

探索结束后，预声明新packet：恰好两个job、seed按61/62排序、H600、phase=`calibration`、purpose=`replay_transfer_independent_validation`；workload和评分合同与transfer config一致。实际采集可沿用冻结v5的 `task_replica_proxy/standalone_proxy`，无需把新预测器带进worker。packet中的质量、画像、预算和运行身份仍严格绑定。

```python
protocol = declare_validation(config, validation_packet_path)
# 先保存protocol.json并封存，再运行GPU；验证前不得读取其实际结果。
# 两条完成后，sources每条含id/path/sha256/packet_path/packet_sha256。
report = validate_transfer(config, protocol, sources)
# 保存report.json，失败也保存，不能用同两条重拟合后重新叫holdout。
activated_raw = activate_transfer(config, protocol_path, report_path)
```

`validate_transfer` 使用严格原生 importer，每条来源必须是相应calibration身份；新phase/seed不能替换成探索51或正式71–73。它核对GPU、checkpoint、质量/输入/画像及评分合同，然后重放同seed逐输入FIFO并计算七项误差。实际worker_ms只用于service误差分母；不进入任何参数估计。两条都满足原有固定thresholds且排序一致才通过。报告声明two-heldout范围，不宣称所有负载或物理组合已验证。

`validated_transfer_spec(data,rates)` 是正式服务/搜索门禁；`check_scoring(config)` 检查完整评分配置。`SearchSession` 支持新模式，底层MILP/DFS继续使用同逐输入界。每次新文件/文件身份变化重新SHA校验；同一未变OS文件身份（device/inode/size/mtime/ctime）下只缓存读取与校验结果，避免每次候选重读大型run。代码绑定为相对路径+SHA，移动相同源码到新冻结根不改变数学版本。

测试 `python -m unittest tests.test_replay_transfer -q` 覆盖未验拒绝、原spec不变/参数零更新、独立验证激活、失败保留、输入/工作负载/参数/来源/report/code篡改拒绝、seed/purpose隔离、SearchSession以及partial界有效性。原v2数学文件保持冻结v5原字节。
