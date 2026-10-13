# 复现与材料边界

## Git 中有什么

首次上传包括研究源码、可读阶段说明、结果目录中的可执行代码与冻结源码副本，以及 `.gitignore` 明确放行的轻量画像、目录、冻结清单、Gate 和摘要。源码按原字节保存，`.gitattributes` 禁用自动换行转换。

模型权重、数据集、专属环境、缓存、原始请求 JSON/JSONL、大规模轨迹/遥测、Nsight/SQLite、日志、图片和大表格主要保留在原工作区。部分小 JSON 是汇总或配置，不代表其引用的所有原始文件已上传。

**克隆 Git 仓库即可查阅源码与精选证据，但不能仅凭克隆完整复现历史 GPU 实验。** 冻结 manifest 内的路径与 SHA256 是历史身份，不在整理时改写成新路径或新哈希。

## 查阅和 CPU 维护

使用 Python 3.10 或更新版本可运行仅依赖标准库的清单工具：

```bash
python tools/repository_inventory.py --check
python tools/repository_inventory.py --write
```

CPU 规划依赖 NumPy/SciPy；MILP 使用 SciPy/HiGHS。历史工作区常用 `envs/data_env/bin/python` 与 `/root/miniconda3/envs/gsdeploy/bin/python`，具体环境以对应阶段 README 和冻结清单为准。

若在独立开发环境中需要现有包的可编辑安装，可使用 `python -m pip install -e '.[planning]'`。这只声明 CPU 规划依赖，不安装所有模型推理依赖。现有代码包含依赖源码相对位置的路径，当前不承诺 wheel 安装后完整实验可移植。

已有 CPU fixture 测试的入口例如：

```bash
python -m unittest discover -s deployment_core/tests -v
python -m unittest discover -s joint_deploy/measurement/tests -v
```

其中部分测试会读取工作区实测画像或启动测试 worker；失败时先核对缺失材料与所用环境。历史 README 所列测试数量是该阶段记录，不代表本次上传重新运行了全部测试。

## 历史 GPU 复现需要什么

对每个具体实验，先检查：

1. 原源码及其冻结版本，必要的动态 router/launcher 和结果目录内包。
2. 准入后的模型检查点、精度、推理参数与专属 Python 环境。
3. 原数据池、输入长度/音频信息、冻结到达和路由轨迹。
4. catalog、容量/显存标定、质量准入证据和效用参考。
5. GPU/驱动身份、显存预算、启动/预热/测量/排空协议。
6. 原始请求记录、journal、遥测与失败材料，供审计及重算。

一些文件写死 `/root/autodl-tmp/generalist-specialist`，另一些 manifest 保存绝对路径。异机复现应使用明确的路径映射或新适配层，并记录新的环境与复现身份；不要通过修改历史 manifest 绕过校验。

如果只有源码与摘要，应表述为“源码查阅/摘要复核”。如果用原记录重算指标，应表述为“离线回放”。新的真实运行需单独冻结输入与记录全部结果，不能覆盖旧观察值。

## 文件与完整性核对

`docs/repository_inventory.json` 记录 Git 可跟踪源码的 SHA256。`method_v1` 和后期实验另有原始冻结验证；仓库清单不会代替这些实验特有的身份、数据和保护检查。

`.gitignore` 的结果 JSON 放行规则采用具体文件路径。新增科学配置/摘要应先确认内容与体积，再有意加入；原始数据应存放在外部归档，并在实验说明中记录来源。大文件只从 Git 跟踪范围排除，首轮整理不会删除本地材料。
