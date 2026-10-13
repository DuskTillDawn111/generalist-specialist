# Module13B — Real Specialist Local Measurement

Gate：**BASELINE_RETAINED**。完成9条冻结task-model paths的真实本机测量与27个context的两regime CPU回放。未执行完整replica profiling、最终deployment、low-QPS、multi-hotspot或Task-Rich。

## 1. 八个checkpoint是否成功本地运行

GPU完整任务路径执行成功：**8/8**。每个checkpoint的精确revision、下载字节、LFS SHA256、缓存位置、CPU/CUDA加载状态见download_manifest.json；版本、FP32、device、attention和TF32见local_environment.json。RTX4090为实际49140MiB设备，不能称作标准24GB实测。

## 2–5. 真实quality、s0、safe memory和gamma/mu

所有quality来自冻结200条本机结果；s0为3个独立进程各30条的加权均值，保留全部有效run。ASR WER越低越好；LID/Text accuracy越高越好。ASR≤0.042181；Text≥0.925；LID只有原semantic admission，无数值accuracy门槛。

| Task | Candidate | Q | Admission | s0 ms | Idle MiB | Safe MiB | γ_real | μ_real | Pareto |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| asr | facebook/wav2vec2-large-960h-lv60-self | 0.020193 | PASS | 21.062 | 1808.0 | 2008.4 | 0.1440 | 0.4924 | REAL_PARETO_OPTIMAL |
| asr | openai/whisper-large-v3 | 0.039488 | PASS | 494.913 | 7310.0 | 8120.3 | 3.3846 | 1.9907 | REAL_DOMINATED |
| asr | openai/whisper-medium | 0.037694 | PASS | 355.203 | 3734.0 | 4147.9 | 2.4292 | 1.0169 | REAL_DOMINATED |
| lid | facebook/mms-lid-256 | 0.940000 | PASS | 52.266 | 4540.0 | 5043.3 | 1.9298 | 2.8094 | REAL_DOMINATED |
| lid | openai/whisper-medium | 0.990000 | PASS | 69.333 | 3502.0 | 3890.2 | 2.5599 | 2.1671 | REAL_PARETO_OPTIMAL |
| lid | speechbrain/lang-id-voxlingua107-ecapa | 0.945000 | PASS | 17.372 | 662.0 | 735.4 | 0.6414 | 0.4097 | REAL_PARETO_OPTIMAL |
| text | philschmid/roberta-large-sst2 | 0.970000 | PASS | 8.643 | 1848.0 | 2052.9 | 2.2929 | 2.4640 | REAL_PARETO_OPTIMAL |
| text | textattack/facebook-bart-large-SST-2 | 0.965000 | PASS | 10.026 | 2048.0 | 2275.0 | 2.6600 | 2.7307 | REAL_DOMINATED |
| text | textattack/xlnet-base-cased-SST-2 | 0.945000 | PASS | 10.638 | 976.0 | 1084.2 | 2.8224 | 1.3013 | REAL_PARETO_OPTIMAL |

safe M = 实测post-warmup idle × 原冻结factor1.110853955254881；与原baseline safe M比较，factor相消。该经验safe定义不是candidate peak的保证。peak NVML、torch maxima、host RSS与idle均分开记录。ASR/LID Whisper-medium共用权重，task-path latency/quality分别测量。

旧immutable分母：ASR Distil-large-v3146.223666144ms/3672MiB；LID Whisper-small27.084134878ms/1616MiB；Text DistilBERT3.769309222ms/750MiB。旧quality与所有旧admitted models完整保留；同期reference control只另报drift，未重定义历史分母。

## 6. 哪些最接近γ≈1.2125–1.25

按γ单轴log-distance最近：lid/facebook/mms-lid-256 γ=1.9298。距离=0.434250；还须同时考虑真实μ、quality与task-local target，不能只靠γ声称落靶区。

## 7. 哪些最接近γ≈1.9375–2.0

按γ单轴log-distance最近：lid/facebook/mms-lid-256 γ=1.9298。距离=0.004005；还须同时考虑真实μ、quality与task-local target，不能只靠γ声称落靶区。

## 8. 真实Pareto-optimal候选

- asr: facebook/wav2vec2-large-960h-lv60-self@54074b1c16f4de6a5ad59affb4caa8f2ea03a119, openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d
- lid: openai/whisper-medium@abdf7c39ab9d0397620ccaea8974cc764cd0953e, speechbrain/lang-id-voxlingua107-ecapa@0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9, openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d, openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9
- text: philschmid/roberta-large-sst2@7d2599d698b7a805b6831e15e830e60a0b07bdb4, textattack/xlnet-base-cased-SST-2@9ceeb077dcd5cf5ae790572b2bd6aec755a263be, distilbert/distilbert-base-uncased-finetuned-sst-2-english@714eb0fa89d2f80546fda750413ed43d93601a13, textattack/roberta-base-SST-2@84ee248b91053ef5d0c748bbac4edfba1cf89584

## 9–13. 最终选择与BALANCED组成

9. asr: **openai/whisper-small** @973afd24965f72e36ca33b3055d56a652f456b4d；BASELINE_RETAINED；冻结log-distance=0.820783。
10. lid: **openai/whisper-small** @973afd24965f72e36ca33b3055d56a652f456b4d；BASELINE_RETAINED；冻结log-distance=0.000000。
11. text: **distilbert/distilbert-base-uncased-finetuned-sst-2-english** @714eb0fa89d2f80546fda750413ed43d93601a13；BASELINE_RETAINED；冻结log-distance=0.000000。
12. BASELINE_RETAINED tasks：asr, lid, text。所有旧reference加入Pareto；reference endpoint距离可为0，距离0不等于已发生transition。
13. BALANCED完整Specialist pool：{"distilbert": ["text"], "resnet50": ["image"], "convnext_tiny": ["image"], "paddleocr": ["ocr"], "whisper-small": ["asr", "lid"], "whisper_turbo": ["asr", "lid"], "distil_whisper": ["asr"], "roberta_sst2": ["text"]}。完整EFFICIENCY pool仍单独保留。仅移除所有task均被新winner替换的旧identity；baseline-retained task保持旧pool，shared checkpoint保留原admitted capabilities；新checkpoint保留全部已测且quality-admitted的冻结paths。Generalists/OCR/Image保持原样。

## 14–17. 真实结构变化

14. Specialist replica count减少：**0/27 contexts**。
15. Generalist总workload加权routing share增加：**0/27 contexts**。
16. 新的G/S task split：**0/27 contexts**。
17. 新产生second Generalist（原N_G<2，新N_G≥2）：**0/27 contexts**。
R4 Generalist多task新增consolidation：0/27；R5 N_G增加：0/27；R6 replica allocation变化：0/27。未要求一定新增second Generalist；原已有N_G≥2的context仍计其S/routing/composition变化。每context deployment、identity、routing、J*、rho、safe M与feasibility见real_structural_replay.csv。
Gate的NEAR报告尺度在GPU结果前锁定：log-L∞距离≤log(1.05)，取自Module12首个memory bracket的5%相对宽度；仅用于gate说明，不改变13A选择公式或任何实测值。
新模型只有真实g(1)=1和M(1)，g(x>1)未知且未复制旧模型曲线。同0/1searched-Specialist count上限诊断单独存real_structural_replay_matched_counts.csv；其是否仍有transition：False。单纯由证据上限差异造成的transition不能视作真实operating-point因果验证。CPU数学J* certificate与冻结Module9 router的运行结果分别保留；CPU回放不是最终GPU deployment evidence。

## 18. 与Module12 prediction一致性
PARTIAL_MATCH。All frozen-rule task winners are old baselines; no selected new operating point tests the counterfactual transition. The baseline endpoint has distance0 but is not an observed transition.

## 19. 下一步哪些selected Specialist需要g(x>1)
无新selected Specialist满足下一阶段profiling条件；本模块不启动profiling。

## 20. Gate
**BASELINE_RETAINED**。gate与局限见module13b_gate.json和real_transition_analysis.json；不强制得到balanced transition，不追加新模型。

反cherry-picking：**NO，未为了让Generalist赢挑模型。** 13A原quality admission→含旧baseline的真实三维Pareto→冻结target union log-L∞最近→higher quality→lower memory→deterministic ID@SHA；选择先于replay，未使用V3 outcome。specialist_selection_rule.json、module13b_execution_freeze.json与balanced_specialist_selection.json提供selection-rule evidence。

完整性审计：95个13A冻结文件、7156个历史受保护文件未变；Module9、原Specialist baseline与阈值未修改。所有失败与compatibility logs保留。完成后停止。

## 实测靶区与可比性补充

9条新路径均通过原准入，其中5条是REAL_PARETO_OPTIMAL：ASR Wav2Vec2；LID Whisper-medium与SpeechBrain；Text RoBERTa-large与XLNet-base。其余4条REAL_DOMINATED。

没有新候选的实测γ落入(1.2125,1.25]或(1.9375,2.0]。MMS-LID最接近这两个global γ区间，γ=1.929755834，距stronger下限约0.4%；但μ=2.809405941、accuracy=0.940，且被旧Whisper-small等支配，γ单轴邻近不足以支持选择。

Whisper-medium ASR的γ=2.429177773、μ=1.016884532，冻结joint log-distance=0.028737897，接近13A冻结的task-local target；其μ也落入首个memory bracket。它被Wav2Vec2以更低WER、更低s0和更低safe M严格支配，因此不能越过Pareto步骤使用它制造composition transition。

原ASR reference仍是Distil-Whisper large-v3；最终ASR选择规则代表为原pool中的Whisper-small，不重定义reference。三task均BASELINE_RETAINED，BALANCED回放目录与原EFFICIENCY目录完全一致；本模块没有建立含新checkpoint的第二个正式regime。27/27 context均可行，所有R1–R6与new-second-Generalist事件均为0。

同期baseline s0：asr 155.680758ms，较冻结分母+6.47%；lid 35.264023ms，较冻结分母+30.20%；text 4.048245ms，较冻结分母+7.40%。Text control CV=2.275083，所有有效样本和异常长请求均保留。γ/μ继续按冻结分母计算；该漂移限制了对窄历史transition bracket的精确本机定位。详见baseline_control_comparability.json、完整repetition statistics与raw logs。
