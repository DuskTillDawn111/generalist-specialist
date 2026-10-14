# 原生后端与执行

## Worker 注册

规划与 CPU mock 使用基础 Python/NumPy 环境。原生 worker 在专属环境运行；不把所有模型版本强行装进一个环境。复制 `configs/workers.example.json` 到本地配置，设置实际 Python、模型检查点和环境参数。

```json
{
  "qwen": {
    "command": ["/path/to/qwen/env/bin/python", "-m", "gsdeploy.backends.qwen"],
    "env": {
      "GS_MODEL_PATH": "/path/to/qwen-checkpoint",
      "GS_PROMPT_FILE": "/path/to/project/configs/prompts.json"
    }
  }
}
```

运行时为每个 worker 设置项目包路径，原生单 GPU 默认 `CUDA_VISIBLE_DEVICES=0`，可在注册表显式覆盖。一个模型的多个副本使用同一注册配置启动多个独立进程。

| 后端 | 依赖与配置 |
| --- | --- |
| `qwen` | Torch、匹配版本 Transformers、qwen-omni-utils；BF16 Qwen Omni 检查点 |
| `phi` | Torch、匹配模型 remote code 的 Transformers/Pillow/SoundFile；BF16 Phi 多模态检查点 |
| `minicpm` | Torch、匹配 MiniCPM remote code 的环境；BF16 检查点 |
| `whisper` | Torch、Transformers、SoundFile；原 FP32 Whisper 检查点 |
| `classifier` | Torch/Transformers 或 TorchVision；`GS_CLASSIFIER` 选择 roberta/bart/distilbert/convnext/resnet |
| `parseq` | 原版本 STRHub、Torch、Pillow；FP32 权重和 `GS_PARSEQ_EXPERIMENT` |

## 数据池

输入 JSONL 每行含 `pool_id`、`task_type`、`sample`。sample 保留任务原字段，例如文本 input/label、图像/音频路径、label_id、duration_s，以及视觉问答选项。`examples/demo_samples.jsonl` 是 mock 输入，不能用于真实模型测量。

## 执行与结果

```bash
gsdeploy run --config configs/low18.json --plan artifacts/low18.json \
  --samples /path/to/formal-pool.jsonl --backend native \
  --workers /path/to/local-workers.json --duration 120 --warmup 10 \
  --output artifacts/native/run.json
```

记录包括配置、计划、请求、时间戳、窗口、排空、关闭报告、指标和稳定性。GPU 进程内存采样不可用时使用显式标注的规划代理做诊断；不会把代理值当成测量证据。

原生运行在加载模型前检查 worker 解释器、检查点、提示文件、样本文件、统一 GPU 选择、显存预算和已有 GPU 进程。`gsdeploy preflight` 默认仅检查输入；加 `--require-gpu` 才检查 GPU。`run --backend native` 自动要求 GPU 检查通过。预检不导入模型库，也不验证环境里的依赖版本和实际推理。

## 检查点与质量证据

未提供准入清单时，原生运行只输出诊断效用。旧版 Python 参数 `quality_admission_confirmed=True` 已移除，布尔值不能授予正式效用资格。提供 `--admission` 时，系统校验全部已部署模型的检查点文件、质量报告 SHA256、报告所属检查点及每条正流量路由的质量阈值；缺失或不匹配会拒绝运行。

```bash
gsdeploy checkpoint --path /absolute/path/to/checkpoint \
  --output artifacts/checkpoint.json
```

目录检查点哈希覆盖全部文件，仅排除 `.cache`、`.git` 和 `__pycache__`。新增文件也会改变身份；大型权重需要完整读取。把输出放入 `configs/admission.example.json` 对应模型的 `checkpoint` 字段，在本地补充报告路径、报告哈希和实际准入规则。示例中的空值是待填项，不能作为已通过的证据。

质量报告格式：

```json
{
  "model": "roberta_sst2",
  "checkpoint_sha256": "<checkpoint 命令输出的 sha256>",
  "tasks": {
    "text": {
      "metric": "accuracy",
      "value": 0.95,
      "dataset": "<数据集版本、划分和样本清单哈希>",
      "protocol": "<评估代码版本、提示词、精度和预处理说明>"
    }
  }
}
```

以上数值仅说明格式。报告必须来自实际质量评估。准入规则包含 `metric`、`threshold` 和 `direction`：`min` 表示至少达到阈值，`max` 表示至多达到阈值。规则应使用研究中已确定的标准。核验会验证记录一致性，不会重新运行质量评估；报告来源及评估协议仍需实验负责人确认。

正式 `U` 还要求运行正常完成、GPU 内存为真实测量且资源有效、稳定性通过。离线评估重新检查记录中的路由覆盖和阈值；原始记录包含 worker 注册表、输入池哈希、GPU 信息和准入证据。它不是完整的环境锁文件，原生适配器仍需真机回归。

## 六任务验收入口

先复制 worker 示例、补充模型环境路径，准备每个任务 5 条真实样本。默认 `configs/low18.json` 选择 RoBERTa、ConvNeXt、PARSeq、Whisper Small 和 Phi，覆盖六个任务。

```bash
python experiments/native_acceptance.py \
  --workers /path/to/workers.local.json --samples /path/to/smoke30.jsonl \
  --output-dir artifacts/acceptance-inputs
```

默认只生成计划和输入预检报告，不查询 GPU、不启动 worker。之后在可用 GPU 上使用新的输出目录，加 `--execute --admission /path/to/admission.json` 执行 5 秒预热、30 秒测量。该规模用于接口和证据链验收，不能替代正式性能实验。输出目录必须不存在，以免覆盖实验。
