# Module 10 operational envelope 与 numerical certificate audit

状态：CPU 审计完成，10B GPU 暂停。未新增热点 GPU run、replica profiling，未修改 Module 9、模型池、质量、memory、g(x) 或 worker/runtime。

## 归档与旧 sweep

原目录中的 41 个文件已按原始字节归档到 `capacity_definition_mismatch_audit/`；原 H0–H3、planning、GPU manifest、cache、raw 文件、execution freeze 均保留。新增 archive_manifest.json 标记 `INVALID_FOR_OPERATIONAL_SWEEP_DUE_TO_CAPACITY_DEFINITION_MISMATCH`。原路径到归档路径映射及 SHA256 在该 manifest 中。

暂停前完成 1 次 canonical 运行，第二次已启动并被外部中断，第三次未启动。两次 raw 均保存且 GPU release verified；第二次 valid_execution=false。旧数据只作为 mismatch 诊断，不构成正式 hotspot benchmark。原 attempt ledger 第二次保留 STARTED，新增归档 metadata 解释 driver 被终止后的实际状态。

## Operational envelope

对原 5 个 overlap tasks（ASR/image/LID/OCR/text）、6 个原 budget、mix_03 I1/I2 全部重算：60 个 context、120 个 S-only/ALL 优化。使用同一 admitted paths、safe memory、实测 g(x)、证据支持 replica counts 和 background conservation；唯一 capacity 定义变化为所有 resident worker 的 D/g ≤ 0.8。MILP 四阶段 lexicographic tie-break 沿用旧审计；固定 witness 后 LP 复核。

10 个 context 的 replica audit PASS；其余 50 个仍需缺失 replica-count 证据，数值仅表示当前证据支持 count space 的容量，不声称 Specialist exhausted。两个合法正窗口均为 B1 ASR（I1/I2）；未发现通过 replica audit 的 multi-Generalist 正窗口。选定 primary 无需补测 replica。

原 deterministic selection rule 原样使用：合法 W>0 中优先 ALL 最大容量 witness ≥2G，再按 R、W、较低 H3 所需 rate、较低 budget、task/background/context ID 排序。此次容量定义统一后，primary 不变。

Primary：**ASR / mix_03 / I1 / B1**；safe budget **17007.174054952 MiB**。

| C_S^op (req/s) | C_A^op (req/s) | W^op (req/s) | R^op |
|---:|---:|---:|---:|
| 3.742602099238 | 4.347876366682 | 0.605274267444 | 0.161725519143 |

最大容量 witness：Phi + distilbert + resnet50 + paddleocr + whisper-small，各 1 replica，即 1G+4S。独立枚举 6912 deployment vectors + LP 与 MILP 的 primary 容量一致；重复 MILP deployment/capacity 一致。

## 新点机械冻结

| 点 | 旧 target req/s（作废，保留） | 新 operational target req/s | 最小 max rho | S-only operational feasible | ALL operational feasible |
|---|---:|---:|---:|---|---|
| H0 | 3.835452420922 | 2.994081679391 | 0.579650252 | True | True |
| H1 | 4.698429215629 | 3.667750057254 | 0.689299565 | True | True |
| H2 | 5.029014474940 | 3.924184379472 | 0.731037975 | False | True |
| H3 | 5.341946406657 | 4.166294086449 | 0.770444847 | False | True |
| H4 | 5.688178262554 | 4.434833894016 | 0.814153601 | False | False |

H0/H1 在 Specialist-only operational envelope 内；H2/H3 超过 Specialist-only operational envelope，但 ALL 在 ≤0.8 下仍 feasible。这里的 exhaustion 仅指冻结 headroom policy 下的 operational capacity，而非真实物理吞吐极限。H4 仅 CPU sanity，ALL 在 ≤0.8 下 infeasible。不移动这些点，也未启动新 GPU。

## 旧 H0 numerical certificate

源代码核查：frozen v2/V3 config 的 seed/candidate gate rho_max=0.8；e2e_router.py 内部 J solver 与 gradient-LP certificate 实际使用 rho≤1−1e−9。此次没有修改该代码。审计独立固定 rho≤0.8，因此要区分两个可行集合的 certificate；不能将两者数字当作同一个 certificate。

| 检查 | 结果 |
|---|---:|
| 旧 frozen solver gap（rho≤1−eps） | 12.243309371 ns |
| 同一旧 iterate 的 gap（rho≤0.8） | 7.964053839 ns |
| 原科学 certificate 阈值（保持） | 10.0 ns |
| 60 位 Decimal 旧 frozen-domain gap | 12.243309354 ns |
| 旧解距离独立最优 ASR split | 3.6982812489310703691711424470601532842789139921789652E-8 req/s |
| 旧解真实 J objective excess | 4.46918775542187373903791295134534048973446895E-16 s |
| 独立最优 max rho | 0.765217003784 |
| 独立最优 J | 0.1408771385597958 s |

- seconds scaling：gap=0.003832976 ns，constraints_valid=True，certificate_pass=True；两次固定初始化求解 routing 完全相同。
- milliseconds scaling：gap=0.003832853 ns，constraints_valid=True，certificate_pass=True；两次固定初始化求解 routing 完全相同。
- microseconds scaling：gap=0.178403174 ns，constraints_valid=True，certificate_pass=True；两次固定初始化求解 routing 完全相同。

结论：12.24 ns 可以由 60 位高精度计算复现，属于极小的优化停止残差，不是单纯 float64 subtraction 造成的错误拒绝。原较宽可行集合下超过 10 ns，原 certificate rejection 应保留。固定 operational 可行集合后，同一旧解为 7.96 ns，通过原阈值；单位 scaling、Brent root 与 Decimal 闭式解均验证同一 J objective 在 rho≤0.8 下有合法最优解。没有 physical routing infeasibility。诊断解未被安装为 fallback，Module 9 保持冻结。

## 所有 context 容量

PASS 为 replica envelope 已认证；REPLICA_EVIDENCE_REQUIRED 为 provisional，不能用于 exhaustion claim。

| context | task | C_S^op | C_A^op | W^op | replica audit |
|---|---|---:|---:|---:|---|
| B1_I1 | asr | 3.742602099 | 4.347876367 | 0.605274267 | PASS |
| B1_I1 | image | 145.512173861 | 145.512173861 | 0.000000000 | PASS |
| B1_I1 | lid | 29.537587359 | 29.537587359 | 0.000000000 | PASS |
| B1_I1 | ocr | 131.849479513 | 131.849479513 | 0.000000000 | PASS |
| B1_I1 | text | 212.240480373 | 212.240480373 | 0.000000000 | PASS |
| B1_I2 | asr | 2.411214798 | 2.721253399 | 0.310038601 | PASS |
| B1_I2 | image | 145.512173861 | 145.512173861 | 0.000000000 | PASS |
| B1_I2 | lid | 21.791616696 | 21.791616696 | 0.000000000 | PASS |
| B1_I2 | ocr | 131.849479513 | 131.849479513 | 0.000000000 | PASS |
| B1_I2 | text | 212.240480373 | 212.240480373 | 0.000000000 | PASS |
| B2_I1 | asr | 14.905559163 | 15.510833430 | 0.605274267 | REPLICA_EVIDENCE_REQUIRED |
| B2_I1 | image | 424.697241543 | 429.675621951 | 4.978380408 | REPLICA_EVIDENCE_REQUIRED |
| B2_I1 | lid | 46.179684128 | 46.179684128 | 0.000000000 | REPLICA_EVIDENCE_REQUIRED |
| B2_I1 | ocr | 131.849479513 | 135.951670364 | 4.102190851 | REPLICA_EVIDENCE_REQUIRED |
| B2_I1 | text | 357.525604296 | 373.096114810 | 15.570510515 | REPLICA_EVIDENCE_REQUIRED |
| B2_I2 | asr | 13.574171862 | 13.884210462 | 0.310038601 | REPLICA_EVIDENCE_REQUIRED |
| B2_I2 | image | 424.697241543 | 427.519779331 | 2.822537788 | REPLICA_EVIDENCE_REQUIRED |
| B2_I2 | lid | 46.179684128 | 46.179684128 | 0.000000000 | REPLICA_EVIDENCE_REQUIRED |
| B2_I2 | ocr | 131.849479513 | 133.410135722 | 1.560656209 | REPLICA_EVIDENCE_REQUIRED |
| B2_I2 | text | 357.525604296 | 366.353446024 | 8.827841729 | REPLICA_EVIDENCE_REQUIRED |
| B3_I1 | asr | 14.905559163 | 15.510833430 | 0.605274267 | REPLICA_EVIDENCE_REQUIRED |
| B3_I1 | image | 424.697241543 | 429.675621951 | 4.978380408 | REPLICA_EVIDENCE_REQUIRED |
| B3_I1 | lid | 46.179684128 | 50.398452699 | 4.218768571 | REPLICA_EVIDENCE_REQUIRED |
| B3_I1 | ocr | 131.849479513 | 135.951670364 | 4.102190851 | REPLICA_EVIDENCE_REQUIRED |
| B3_I1 | text | 357.525604296 | 373.096114810 | 15.570510515 | REPLICA_EVIDENCE_REQUIRED |
| B3_I2 | asr | 13.574171862 | 13.884210462 | 0.310038601 | REPLICA_EVIDENCE_REQUIRED |
| B3_I2 | image | 424.697241543 | 427.519779331 | 2.822537788 | REPLICA_EVIDENCE_REQUIRED |
| B3_I2 | lid | 46.179684128 | 46.179684128 | 0.000000000 | REPLICA_EVIDENCE_REQUIRED |
| B3_I2 | ocr | 131.849479513 | 133.410135722 | 1.560656209 | REPLICA_EVIDENCE_REQUIRED |
| B3_I2 | text | 357.525604296 | 366.353446024 | 8.827841729 | REPLICA_EVIDENCE_REQUIRED |
| B4_I1 | asr | 15.369810771 | 15.968195548 | 0.598384777 | REPLICA_EVIDENCE_REQUIRED |
| B4_I1 | image | 424.697241543 | 429.675621951 | 4.978380408 | REPLICA_EVIDENCE_REQUIRED |
| B4_I1 | lid | 46.179684128 | 51.147523289 | 4.967839161 | REPLICA_EVIDENCE_REQUIRED |
| B4_I1 | ocr | 131.849479513 | 135.951670364 | 4.102190851 | REPLICA_EVIDENCE_REQUIRED |
| B4_I1 | text | 357.525604296 | 373.096114810 | 15.570510515 | REPLICA_EVIDENCE_REQUIRED |
| B4_I2 | asr | 13.975317065 | 14.176643169 | 0.201326104 | REPLICA_EVIDENCE_REQUIRED |
| B4_I2 | image | 424.697241543 | 427.519779331 | 2.822537788 | REPLICA_EVIDENCE_REQUIRED |
| B4_I2 | lid | 46.179684128 | 48.996245461 | 2.816561333 | REPLICA_EVIDENCE_REQUIRED |
| B4_I2 | ocr | 131.849479513 | 133.410135722 | 1.560656209 | REPLICA_EVIDENCE_REQUIRED |
| B4_I2 | text | 357.525604296 | 366.353446024 | 8.827841729 | REPLICA_EVIDENCE_REQUIRED |
| B5_I1 | asr | 15.369810771 | 15.968195548 | 0.598384777 | REPLICA_EVIDENCE_REQUIRED |
| B5_I1 | image | 424.697241543 | 430.427359182 | 5.730117640 | REPLICA_EVIDENCE_REQUIRED |
| B5_I1 | lid | 46.179684128 | 51.897668789 | 5.717984661 | REPLICA_EVIDENCE_REQUIRED |
| B5_I1 | ocr | 131.849479513 | 136.837897493 | 4.988417980 | REPLICA_EVIDENCE_REQUIRED |
| B5_I1 | text | 357.525604296 | 375.447267496 | 17.921663200 | REPLICA_EVIDENCE_REQUIRED |
| B5_I2 | asr | 13.975317065 | 14.176643169 | 0.201326104 | REPLICA_EVIDENCE_REQUIRED |
| B5_I2 | image | 424.697241543 | 430.427359182 | 5.730117640 | REPLICA_EVIDENCE_REQUIRED |
| B5_I2 | lid | 46.179684128 | 49.636005373 | 3.456321245 | REPLICA_EVIDENCE_REQUIRED |
| B5_I2 | ocr | 131.849479513 | 136.837897493 | 4.988417980 | REPLICA_EVIDENCE_REQUIRED |
| B5_I2 | text | 357.525604296 | 375.447267496 | 17.921663200 | REPLICA_EVIDENCE_REQUIRED |
| B6_I1 | asr | 15.369810771 | 16.733928292 | 1.364117520 | REPLICA_EVIDENCE_REQUIRED |
| B6_I1 | image | 424.697241543 | 433.394706401 | 8.697464858 | REPLICA_EVIDENCE_REQUIRED |
| B6_I1 | lid | 46.179684128 | 51.897668789 | 5.717984661 | REPLICA_EVIDENCE_REQUIRED |
| B6_I1 | ocr | 131.849479513 | 136.837897493 | 4.988417980 | REPLICA_EVIDENCE_REQUIRED |
| B6_I1 | text | 357.525604296 | 376.965854514 | 19.440250218 | REPLICA_EVIDENCE_REQUIRED |
| B6_I2 | asr | 14.388548631 | 15.107305324 | 0.718756693 | REPLICA_EVIDENCE_REQUIRED |
| B6_I2 | image | 424.697241543 | 431.556270301 | 6.859028758 | REPLICA_EVIDENCE_REQUIRED |
| B6_I2 | lid | 46.179684128 | 51.897668789 | 5.717984661 | REPLICA_EVIDENCE_REQUIRED |
| B6_I2 | ocr | 131.849479513 | 136.837897493 | 4.988417980 | REPLICA_EVIDENCE_REQUIRED |
| B6_I2 | text | 357.525604296 | 375.447267496 | 17.921663200 | REPLICA_EVIDENCE_REQUIRED |

完整 witness/routing/pressure 在 capacity_envelope_operational.json/csv；primary 和点分别在 primary_hotspot_selection.json、hotspot_points.json；单位 scaling 与高精度细节在 numerical_certificate_audit.json。

审计到此停止。新的 GPU hotspot runs 需要用户确认；当前没有授权新 GPU 执行，也没有形成正式 Module 10 benchmark 结论。
