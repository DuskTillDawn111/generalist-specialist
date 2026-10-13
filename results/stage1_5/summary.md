# Stage 1.5 Feasibility Sanity Sweep

仅验证 Stage 1 可行集合及 routing；所有组合按模型数量和名称展示。
A/B 使用 rho_max=0.8；C 使用固定基准 workload。显存预算来自现有 snapshot。
完整 arrival rates、routing、utilization、淘汰原因保存在对应 JSON 中。

## Table 1：Load Intensity

Scale | Total RPS | Coverage | Memory | Capacity | Feasible Set
--- | ---: | ---: | ---: | ---: | ---
0.25 | 5.5 | 16 | 16 | 16 | qwen; distilbert + qwen; paddleocr + qwen; qwen + resnet50; qwen + whisper; distilbert + paddleocr + qwen; distilbert + qwen + resnet50; distilbert + qwen + whisper; paddleocr + qwen + resnet50; paddleocr + qwen + whisper; qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50; distilbert + paddleocr + qwen + whisper; distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
0.50 | 11 | 16 | 16 | 9 | qwen + whisper; distilbert + qwen + whisper; paddleocr + qwen + whisper; qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50; distilbert + paddleocr + qwen + whisper; distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
0.75 | 16.5 | 16 | 16 | 5 | qwen + resnet50 + whisper; distilbert + paddleocr + qwen + whisper; distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
1.00 | 22 | 16 | 16 | 3 | distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
1.25 | 27.5 | 16 | 16 | 3 | distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
1.50 | 33 | 16 | 16 | 2 | distilbert + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper

## Table 2：Composition

Workload | Total RPS | Capacity Feasible | Feasible Deployments
--- | ---: | ---: | ---
balanced | 20 | 4 | qwen + resnet50 + whisper; distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
text-heavy | 20 | 5 | distilbert + qwen + whisper; distilbert + paddleocr + qwen + whisper; distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
image-heavy | 20 | 4 | qwen + resnet50 + whisper; distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
ocr-heavy | 20 | 4 | paddleocr + qwen + whisper; distilbert + paddleocr + qwen + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
asr-heavy | 14 | 0 | ∅
vqa-heavy | 15 | 0 | ∅
lid-heavy | 20 | 8 | qwen + whisper; distilbert + qwen + whisper; paddleocr + qwen + whisper; qwen + resnet50 + whisper; distilbert + paddleocr + qwen + whisper; distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper

## Table 3：rho_max

rho_max | Capacity Feasible | Feasible Deployments
--- | ---: | ---
0.50 | 2 | distilbert + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
0.60 | 3 | distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
0.70 | 3 | distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
0.80 | 3 | distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
0.90 | 5 | qwen + resnet50 + whisper; distilbert + paddleocr + qwen + whisper; distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper
0.95 | 5 | qwen + resnet50 + whisper; distilbert + paddleocr + qwen + whisper; distilbert + qwen + resnet50 + whisper; paddleocr + qwen + resnet50 + whisper; distilbert + paddleocr + qwen + resnet50 + whisper

## Composition observations

本次 sweep 得到 5 种不同 feasible set。
VQA 正负载时包含 Qwen 的检查：PASS。
空集合中的该性质仅为真空成立，不表示存在可服务组合。

- asr-heavy / whisper：含该模型的可行组合 0 个，不含的 0 个；仅添加该模型就从 Capacity 淘汰变为可行的配对 0 个。
  当前负载无可行组合，不能由此证明该 specialist 在本场景恢复可行性。
- text-heavy / distilbert：含该模型的可行组合 4 个，不含的 1 个；仅添加该模型就从 Capacity 淘汰变为可行的配对 3 个。
- image-heavy / resnet50：含该模型的可行组合 4 个，不含的 0 个；仅添加该模型就从 Capacity 淘汰变为可行的配对 4 个。
- ocr-heavy / paddleocr：含该模型的可行组合 4 个，不含的 0 个；仅添加该模型就从 Capacity 淘汰变为可行的配对 4 个。

配对结果检验 specialist 对共享 Qwen capacity 约束的作用；零目标 LP 的 witness 不保证 Qwen 利用率最小，因此不从单个 witness 推断最优卸载量。
这些结论仅适用于 standalone service-demand screening，不代表真实共置性能。

## Validation

Load monotonicity: PASS  
rho monotonicity: PASS  
Routing conservation: PASS  
Capability validation: PASS  
Capacity validation: PASS  
Non-negativity: PASS  

Scenarios: 19; routing witnesses: 84; failures: 0.

集合单调性检查覆盖 A/C 各自所有 15 对参数；不是仅比较可行组合数量。
routing 守恒复用相对容差 1e-7（零到达率不加绝对余量）；capacity 容差 1e-9。

STAGE 1.5 VALIDATION: PASS
