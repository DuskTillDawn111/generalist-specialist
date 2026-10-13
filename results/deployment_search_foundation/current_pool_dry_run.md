# 当前 5-model pool：Module 3A dry run

使用 Module 2G 冻结的三个 workload；只读已有 standalone profiles。
Seed 由最小内存 MILP 实际求解；没有预设 seed。Canonical min-max LP 不代表真实稳定性。
Runtime evaluation = 0；未运行 GPU profiling。每组只检查 seed 的一跳邻域。

## W1_balanced

Status: `DRY_RUN_COMPLETE`

```text
Iteration 0
seed:
  qwen
canonical z=0.312392558
bottleneck:
  qwen rho=0.312392558
pressure tasks: asr, lid, image, ocr, text, visual_qa
candidates:
  +distilbert: PRUNED_TOP_B
    z=0.276681587 delta_z=+0.035710971 rank=4
    relief_per_memory=2.3397828388480004 memory_freed_mib=0.0
  +paddleocr: PRUNED_TOP_B
    z=0.274533279 delta_z=+0.037859279 rank=3
    relief_per_memory=3.196572148404124 memory_freed_mib=0.0
  +resnet50: SELECTED_TOP_B
    z=0.256547283 delta_z=+0.055845276 rank=2
    relief_per_memory=4.543438498756294 memory_freed_mib=0.0
  +whisper: SELECTED_TOP_B
    z=0.158317016 delta_z=+0.154075542 rank=1
    relief_per_memory=4.656378936954797 memory_freed_mib=0.0
Top-B:
  qwen, whisper
  qwen, resnet50
```

LP solves=5; runtime evaluations=0

## W2_generalist_heavy

Status: `DRY_RUN_COMPLETE`

```text
Iteration 0
seed:
  qwen
canonical z=0.542974868
bottleneck:
  qwen rho=0.542974868
pressure tasks: asr, lid, image, text, ocr, visual_qa
candidates:
  +distilbert: PRUNED_TOP_B
    z=0.476016797 delta_z=+0.066958071 rank=3
    relief_per_memory=4.3870928228400015 memory_freed_mib=0.0
  +paddleocr: PRUNED_TOP_B
    z=0.476721129 delta_z=+0.066253739 rank=4
    relief_per_memory=5.59400125970722 memory_freed_mib=0.0
  +resnet50: SELECTED_TOP_B
    z=0.445245635 delta_z=+0.097729233 rank=2
    relief_per_memory=7.9510173728235145 memory_freed_mib=0.0
  +whisper: SELECTED_TOP_B
    z=0.281518650 delta_z=+0.261456218 rank=1
    relief_per_memory=7.9015735379762 memory_freed_mib=0.0
Top-B:
  qwen, whisper
  qwen, resnet50
```

LP solves=5; runtime evaluations=0

## W3_specialist_friendly

Status: `DRY_RUN_COMPLETE`

```text
Iteration 0
seed:
  qwen
canonical z=0.330755497
bottleneck:
  qwen rho=0.330755497
pressure tasks: asr, image, ocr, text, lid, visual_qa
candidates:
  +distilbert: PRUNED_TOP_B
    z=0.286116783 delta_z=+0.044638714 rank=4
    relief_per_memory=2.9247285485599988 memory_freed_mib=0.0
  +paddleocr: PRUNED_TOP_B
    z=0.283431397 delta_z=+0.047324099 rank=3
    relief_per_memory=3.9957151855051527 memory_freed_mib=0.0
  +resnet50: SELECTED_TOP_B
    z=0.260948902 delta_z=+0.069806595 rank=2
    relief_per_memory=5.679298123445364 memory_freed_mib=0.0
  +whisper: SELECTED_TOP_B
    z=0.190670898 delta_z=+0.140084599 rank=1
    relief_per_memory=4.233553007800183 memory_freed_mib=0.0
Top-B:
  qwen, whisper
  qwen, resnet50
```

LP solves=5; runtime evaluations=0
