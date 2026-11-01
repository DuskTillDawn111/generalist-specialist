# T8 完成后的 V4 一次性接续草案

本草案尚未启动，也不创建运行 claim。固定 V4 CPU 协议已准备完成；GPU 仍由主任务统一启动。T8 continuation 已负责全部三项拟合，V4 不重复拟合，不增加服务模型或调参轮次。

所有下列路径相对 `artifacts/predictor-gpu-20261027`：

- T8 终态：`admission-training-continuation-v1/completed.json`。
- T8 原协议/审核：该目录的 `protocol.json`、`root-review.json`。
- 三个实际拟合 manifest：该目录的 `capacity-fit/manifest.json`、`resource-fit/manifest.json`、`quality-fit/manifest.json`。
- V4 协议：`admission-validation-v1/protocol.json`。
- 建议唯一新输出：`admission-validation-v1/sealed-v1`、`admission-validation-v1/collected-v1`、`admission-validation-v1/report-v1`。
- public plan 级永久 claim：`admission-validation-v1/target/independent-V4-attempt.json`。

必须先审计 `completed.json`，不能只检查文件存在、进程退出、单个 raw 或部分 manifest。其 schema 必须为 `gsdeploy.admission_T8_continuation.v1.completed`，source 为当前冻结的 `admission_training_continuation_v1.py`，protocol/root_review 对应原固定文件，`job_count=8`，`service_refitted=false`，`independent_or_final_launched=false`，`final_gain_eligible=false`。`jobs` 必须逐项等于原 `training.verify(campaign)['jobs']` 的固定 job_id 顺序及各原路径 `training-complete.json` receipt；每张 completion 仍须匹配该 campaign/job、1200 请求、controller 和原 native raw，并再执行严格 `training.map_training`。三个 manifest receipt 必须恰为上述三个固定目录、分别绑定 V4 protocol 所指的原 T8 协议。保留 upstream FAIL、native stability 和所有 UNKNOWN。

建议接续分两段，CPU seal 后由主任务检查结果并启动既定诊断 V4：

```python
from pathlib import Path
from tools.predictor_validation_20261027 import admission_training_continuation_v1 as tc
from tools.predictor_validation_20261027 import admission_validation_v1 as v
from tools.predictor_validation_20261027 import admission_validation_run_v1 as vr
from tools.predictor_validation_20261027.native_flow import receipt, checked, save
from tools.predictor_validation_20261027.probe_validation import read

base = Path('artifacts/predictor-gpu-20261027').resolve()
t = base / 'admission-training-continuation-v1'
target = base / 'admission-validation-v1'
# First validate the exact completed packet and every completion as described
# above. Freeze a root-reviewed continuation source and a complete transitive
# receipt closure before launching this once; do not substitute a new T8 run.
protocol, review = tc.audit_root_review(t/'protocol.json', t/'root-review.json')
done = read(t/'completed.json')
assert done['schema'] == tc.SCHEMA + '.completed'
assert done['source'] == receipt(tc.__file__)
assert done['protocol'] == receipt(t/'protocol.json')
assert done['root_review'] == receipt(t/'root-review.json')
assert done['job_count'] == 8
assert done['service_refitted'] is False
assert done['independent_or_final_launched'] is False
assert done['final_gain_eligible'] is False
campaign = v.training.verify(checked(protocol['campaign']))
expected = []
for job in campaign['jobs']:
    packet_path = Path(job['output'])/'training-complete.json'
    packet = read(packet_path)
    assert packet['schema'] == v.training.SCHEMA + '.completed'
    assert packet['campaign'] == protocol['campaign'] and packet['job'] == job
    assert packet['request_count'] == 1200 and packet['final_gain_eligible'] is False
    mapped = v.training.map_training(job, protocol['campaign']['path'])
    assert packet['native_raw'] == mapped['native_raw']
    assert packet['controller_sha256'] == job['controller_sha256']
    expected.append(dict(job_id=job['job_id'],completion=receipt(packet_path)))
assert done['jobs'] == expected
manifests = {key:receipt(t/(key.removesuffix('_protocol')+'-fit')/'manifest.json')
             for key in tc.MODULES}
assert done['manifests'] == manifests
for key, module in tc.MODULES.items():
    manifest = module.verify_manifest(checked(manifests[key]))
    assert manifest['protocol'] == protocol[key]
closure = vr.closure([receipt(t/'completed.json'),receipt(target/'protocol.json'),
                      receipt(v.__file__),receipt(vr.__file__)])
vr.recheck(closure)

# CPU only, in a process whose CUDA_VISIBLE_DEVICES is empty:
v.seal(target/'protocol.json',
       manifests['resource_protocol']['path'], manifests['quality_protocol']['path'],
       manifests['capacity_protocol']['path'], target/'sealed-v1')
vr.recheck(closure)
# v.seal revalidates all fits. vr.run performs the full numerical sealed audit
# once immediately before launch, then hashes the transitive closure per job.
```

上述 CPU seal 命令的精确 CLI：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.predictor_validation_20261027.admission_validation_v1 seal \
  --protocol artifacts/predictor-gpu-20261027/admission-validation-v1/protocol.json \
  --resource-manifest artifacts/predictor-gpu-20261027/admission-training-continuation-v1/resource-fit/manifest.json \
  --quality-manifest artifacts/predictor-gpu-20261027/admission-training-continuation-v1/quality-fit/manifest.json \
  --capacity-manifest artifacts/predictor-gpu-20261027/admission-training-continuation-v1/capacity-fit/manifest.json \
  --output artifacts/predictor-gpu-20261027/admission-validation-v1/sealed-v1
```

仅在精确 T8 completed 审计及上述 seal 成功后，主任务执行固定四路由诊断。GPU 命令应继承主任务已核实的设备环境，不应沿用 CPU 进程的空 `CUDA_VISIBLE_DEVICES`：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.predictor_validation_20261027.admission_validation_run_v1 run \
  --sealed-dir artifacts/predictor-gpu-20261027/admission-validation-v1/sealed-v1 \
  --output artifacts/predictor-gpu-20261027/admission-validation-v1/collected-v1 \
  --diagnostic
```

Python API 对应 `vr.run(target/'sealed-v1', target/'collected-v1', diagnostic=True)`。使用显式诊断意图保留失败服务模型的既有事实与 UNKNOWN，不自动放行最终部署或声称正收益。

无论四路由完成，还是部分完成后失败，后续单独 CPU report；失败时不能再次调用 run：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m tools.predictor_validation_20261027.admission_validation_run_v1 report \
  --sealed-dir artifacts/predictor-gpu-20261027/admission-validation-v1/sealed-v1 \
  --run-dir artifacts/predictor-gpu-20261027/admission-validation-v1/collected-v1 \
  --output artifacts/predictor-gpu-20261027/admission-validation-v1/report-v1
```

Python API 为 `vr.report(...)`。它重新执行独立 native 审计及原报告/门槛/资格重算，报告四候选、4800 请求的原固定域，未完成的候选不会消失。诊断与原指标均不得转为新训练数据。后续 F8 只能独立衔接，不能由该接续草案启动。

若实现自动等待版本，应绑定 T8 当前原始 PID、Linux start ticks、cwd、argv、已冻结原脚本及 completed 路径，拒绝 PID 复用/重复进程/无 completed 的异常退出。用 exclusive continuation started claim 且不恢复失败，轮询等待只读、不启动或替换 T8。该自动等待实现和主任务 source-bound review 需要另行完成；本草案不声称已经具备它。
