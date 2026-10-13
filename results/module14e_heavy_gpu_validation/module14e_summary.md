# Module14E：Bounded Efficiency vs Heavy GPU Validation

Gate: **STOP_GPU_MEMORY_BUDGET_REJECTION**。2个CPU选择并预先冻结的context，双方各3次独立运行，共12会话，无运行错误或OOM。

|context|E stability|H stability|E safe idle MiB|H safe idle MiB|frozen safe budget MiB|
|---|---|---|---:|---:|---:|
|no_vqa_phase/B3|3/3 steady|3/3 steady|24709.835–24763.156|19237.769–19664.337|32019.254|
|mix_03_I3/B6|3/3 unstable|3/3 unstable|49295.255–49461.883|49908.447–50763.804|49140|

高负载双方均存在持续尾部backlog增长；drain最终完成不构成steady。Heavy在三次运行中均超过原safe-memory budget，E也三次超预算。没有物理OOM，仍不能放宽原safe-budget。原狭义E-steady/H-unstable相对否定判据未触发；该CPU-feasible高负载joint deployment的绝对safe-memory/stability未通过GPU核验。

后续强制停止：未进入14F Stable Overflow Candidate Search，未运行hotspot或second-G GPU，不改模型、任务、SLO、rho、J、workload或memory budget。全部raw及原冻结选择、路由和阈值保留。
