# 共置干扰与 ASR 执行/排队

均值的 S 分母会随分流改变；下面额外使用保留 S 的同请求对照和被分流请求的 S-only 反事实，避免输入选择混淆。原生 wait=start-submit，service=end-start。CPU 占用来自 worker 树，100%=一 CPU 核；NVML 为采样占用，不能唯一归因为 GPU 因果。

- A0 qwen ASR: n=0，service mean/P95=None/Nonems，wait mean/P95=None/Nonems。
- A0 whisper-small ASR: n=595，service mean/P95=227.56721412941172/436.25355339999993ms，wait mean/P95=36.059269505882355/229.76183019999985ms。
- A1 qwen ASR: n=28，service mean/P95=947.6330576428572/1665.9299008999997ms，wait mean/P95=56.10667057142857/504.237342999999ms。
- A1 whisper-small ASR: n=567，service mean/P95=222.49415568959438/452.5412622999995ms，wait mean/P95=35.680296481481484/238.1382672999998ms。
- A2 qwen ASR: n=53，service mean/P95=957.2757058113207/1554.8958169999999ms，wait mean/P95=66.03672488679246/533.6570425999995ms。
- A2 whisper-small ASR: n=542，service mean/P95=223.761248099631/481.45215039999994ms，wait mean/P95=30.529843723247236/200.01182984999937ms。
- split05 qwen#0: same n=0，保留 S service/wait 差=None/Nonems；分流 G 相对同输入原 S service/wait 差=724.5514119642859/42.172773892857144ms。
- split05 whisper-small#0: same n=567，保留 S service/wait 差=-5.294567992945327/-1.4715840282186945ms；分流 G 相对同输入原 S service/wait 差=None/Nonems。
- split10 qwen#0: same n=0，保留 S service/wait 差=None/Nonems；分流 G 相对同输入原 S service/wait 差=736.1424081320757/22.677454377358494ms。
- split10 whisper-small#0: same n=542，保留 S service/wait 差=-4.435112841328414/-4.81558804612546ms；分流 G 相对同输入原 S service/wait 差=None/Nonems。
- A1 text: mean/P95 Δ=0.7242365321100941/-0.49620680000000306ms。
- A1 image: mean/P95 Δ=4.076614394495415/9.38074099999998ms。
- A1 ocr: mean/P95 Δ=-0.7062605859374997/-2.0792900000000003ms。
- A1 asr: mean/P95 Δ=29.635183284033587/131.4321010999994ms。
- A1 lid: mean/P95 Δ=-3.074746477272697/25.4211970500005ms。
- A1 visual_qa: mean/P95 Δ=19.734756036036032/90.21144950000001ms。
- A2 text: mean/P95 Δ=1.6739323302752283/3.0427541999999796ms。
- A2 image: mean/P95 Δ=-0.7242886513761455/-1.2116410000000037ms。
- A2 ocr: mean/P95 Δ=-0.6538283593749981/-0.24355720000001213ms。
- A2 asr: mean/P95 Δ=59.49814800336128/292.9712110999991ms。
- A2 lid: mean/P95 Δ=-8.056008977272725/15.168713450000496ms。
- A2 visual_qa: mean/P95 Δ=46.7344875135135/356.462484ms。

资源均值/峰值见 resource_comparison.csv，逐样本见 GPU_CPU_environment.csv；所有背景任务变化保留，不以改善任务替代完整六任务效用。单次序列 A0→A1→A2 未隔离运行次序/温漂；物理环境 Gate 单独检查。

## 精确 ASR 均值分解

- A1: S 保留等待变化 -834.388ms；G 额外执行 20287.440ms；ASR mean Δ=29.635ms，精确重建=29.635ms。
- A2: S 保留等待变化 -2610.049ms；G 额外执行 39015.548ms；ASR mean Δ=59.498ms，精确重建=59.498ms。

完整五项及分母见 native_latency_decomposition.csv。保留 S 负 wait delta 是真实配对等待下降；分流请求消失不冒充保留 S 排队改善。
