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

当前 CLI 未替代论文实验的检查点哈希、完整质量及环境冻结。正式效用资格中的质量确认由已验证调用方通过 Python API 提供；CLI 默认保守保留 `U=null` 和诊断算术。原生适配器尚需真机回归，CPU mock 测试不能替代它。
