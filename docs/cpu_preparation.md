# 四任务 CPU 前置准备

当前研究仅含上下文 QA、图像分类、整页 OCR、ASR，LID/VQA 不进入本轮。
候选、版本和约束以 `configs/study.json` 为准。模型下载、依赖编译、CPU 预处理不等于 GPU 验收。
最终机器可读状态保存在：

```text
artifacts/four-task-small-pool-20261015/final-preparation/cpu-readiness.json
```

该报告检查资源内容、环境锁与 CPU 导入、全部输入前处理、测试、实验清单及哈希。
只有全部通过才报告 `CPU_PREREQUISITES_COMPLETE_GPU_UNVALIDATED`；质量准入、运行 catalog 和 U 保持未知。

## 数据与评分

正式质量输入 1600 条，性能校准 360 条。当前清单由 study.json 的
`quality_preparation.manifest` / `calibration_preparation.manifest` 和 SHA-256 绑定。
前处理规范固定后使用 evaluation-v6/calibration-v6 版本；媒体和逐任务清单引用不可变 v5 文件。

| 任务 | 质量集 | 校准集 | 独立性及范围 |
|---|---:|---:|---|
| QA | 500 | 64 | 6 篇未用于校准的文章；质量集可回答/不可回答各 250；完整段落 |
| 图像 | 400 | 200 | 200 个来源类各 2 张新图；预测仍为 1000 标签；ID/内容哈希隔离 |
| OCR | 200 | 32 | 原生坐标/文本前缀核验、排除校准页和相同图片；完整参考修复 |
| ASR | 500 | 64 | 质量集 dev-other 33 位说话人与 test-other 校准隔离；4–30 秒完整音频 |

ASR 4–8/8–12/12–20/20–30 秒配额为 200/150/120/30；不是自然时长权重。
ImageNet-A 只覆盖 200 个来源类，不能称为完整 ImageNet1K；最少的来源类在校准后只剩 2 图。
QA 文章簇只有 6 个，置信区间范围必须报告。这些是相对本项目校准的隔离，不证明训练数据无重叠。

OCR 原生产脚本对每行 `split(',')` 后只取 `tokens[8]`，丢掉文本中的逗号及后文。
新解析只分割前 8 个坐标逗号，不利用预测补标注。
原生测试归档 SHA-256：`41b3c746a20226fddc80d86d4b2a903d43b5be4f521dd1bbe759dbf8844745e2`。
361 页中 342 页核验通过，331 页可恢复；17 页文本前缀不一致、1 页编码不符、1 页缺文件，排除。
32 页校准均通过核验，新版本修复其参考；旧画像/原始 CER 保持不变。

部分原图约 3500 万像素。所有候选共用 RGB/EXIF 规范化后的整页，最长边 2048、
LANCZOS、无损 PNG，不裁掉页面内容；保留原图路径、原哈希、原/新尺寸和标注哈希。
Qwen 图像预算固定为最多约 1280 视觉 token，音频原生补齐预算为 30 秒、禁止截断。
HTML/Markdown 评分规则、EOS、最低质量、bootstrap、五项参考值见 [固定协议](compact_protocol.md)。

## 本地资源与依赖

数据/权重位于仓库外 `/root/autodl-tmp/gsdeploy-assets`，按模型 ID 与不可变 revision 分目录。
Hub 元数据的小文件使用 Git blob SHA-1，大文件使用固定 LFS SHA-256 验证。
只取实际使用的一种权重格式；Phi/Whisper 的完全相同旧文件通过硬链接复用，未删除旧模型。
传输镜像的文件也必须匹配原始固定版本的内容哈希。

使用 5 个独立 Python 3.10 环境，旧实验环境不升级：

| 环境 | 用途 | Transformers |
|---|---|---|
| compact-qwen | Qwen Omni | 固定源 cb39f7d 构建的 4.52.0.dev0 wheel |
| compact-minicpm | MiniCPM GPTQ | 4.44.2 + 固定 AutoGPTQ fork |
| compact-common | Phi | 4.48.2 + FlashAttention 2.7.4.post1 |
| compact-deepseek | DeepSeek OCR | 4.46.3 + FlashAttention 2.7.3 |
| compact-standard | T5 / SigLIP / Whisper | 4.51.3 |

传递依赖的完整哈希锁在 `configs/environments/*.lock`，源/二进制层由
`configs/environments/native-lock.json` 记录。`configs/compact_environments.json` 绑定解释器、
锁文件和每个环境的 CPU 导入审计。
PyTorch 2.6/CUDA12.4、ABI=false；CPU 导入仅验证包/动态链接兼容，尚未执行 CUDA kernel。

MiniCPM 固定 fork a9c8109 构建 ExLlama v1 和通用 fallback 扩展；执行器显式禁用 v2/Marlin。
CPU 编译使用 CUDA12.4、固定 SM89、Clang14、单任务；没有 GPU 枚举或初始化。
当前容器仅 2GiB 内存、0.5 CPU 配额，编译使用明确记录的 精简头文件和等价 dtype 分派 API 补丁减少模板解析开销、适配 PyTorch2.6，
不改变 kernel 计算语句。补丁与构建入口在 `configs/environments/autogptq-memory.patch` 和
`scripts/build_compact_gptq.py`；构建日志和原/新文件哈希保留。

## 重跑 CPU 检查

在项目根目录执行，输出到新目录；被中断的完整前处理可在完全相同的合同下续跑：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/check_compact_inputs.py \
  --inventory artifacts/four-task-small-pool-20261015/final-preparation/resources/resource-inventory.json \
  --classmap /root/autodl-tmp/gsdeploy-assets/datasets/imagenet-class-index.json \
  --output NEW_CPU_AUDIT_DIRECTORY
```

串行检查 3×(1600+360)+(1600+360)=7840 次真实 tokenizer/媒体前处理。
不构造候选神经网络，不读权重；每条记录保留输入长度、张量形状和源哈希。
原生请求绑定与 EOS 观察另用 CPU 张量替身做回归测试。
实验准备/未来运行命令见 [实验顺序](next_experiments.md)。
