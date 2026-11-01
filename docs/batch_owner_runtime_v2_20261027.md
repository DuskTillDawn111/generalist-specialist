# 同 owner batch 到达与交付运行时 v2

[batch_owner.py](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/batch_owner.py) 将公开sample、到达表、[batch planner](/root/autodl-tmp/generalist-specialist-project/gsdeploy/runtime/batching.py) 与 [SharedTextBatchExecutor](/root/autodl-tmp/generalist-specialist-project/gsdeploy/backends/shared_text_batch.py) 接成完整执行循环；不改旧transport或其RESULT协议。CPU假模型已从公开流实际执行prepare、组批、一次批generate并逐请求交付，真实GPU尚未启动。

## 公共入口

```python
from gsdeploy.runtime.batching import BatchConfiguration
from gsdeploy.runtime.batch_owner import run_arrivals

report = run_arrivals(
    samples=[complete_text_sample_1, complete_text_sample_2],
    arrivals=[
        {"request_id": "r1", "sample_id": "sample-1", "arrival_ns": 0, "alias": "text#0"},
        {"request_id": "r2", "sample_id": "sample-2", "arrival_ns": 5_000_000, "alias": "text#1"},
    ],
    configuration=BatchConfiguration(2, 8192, 10_000_000, (1024, 4096)),
    backend_factory=create_one_shared_text_owner,
    owner="shared_text",
    parent_receipt={"path": "/absolute/parent.json", "sha256": "..."},
    output="/absolute/new-runtime-report.json",
)
```

`backend_factory`在有效公共输入和父receipt检查之后只调用一次；应返回同一个SharedTextBatchExecutor，模型load耗时独立计入。加载完成设置monotonic epoch，arrival_ns为之后的非负偏移。arrivals可以无序，按arrival_ns和原公开列表顺序稳定排序；request_id唯一，sample_id必须存在，可重复请求同一sample。原logical alias、pool_id和arrival_index逐请求保留，不因两alias共用一个owner而丢失。

`run_retained_arrivals(samples,arrivals,configuration,spec_path,source_root,execute=False,...)`是相同执行循环的真实模型便捷入口，默认拒绝模型加载；只有显式execute=True才把load_retained工厂交给run_arrivals。不能由CPU测试自动触发GPU。collector可以直接调用本模块，不需要先写一个中间helper才能形成有效执行。

## 实际调度语义

1. 已到达请求先按公开到达顺序进入arrived队列。owner空闲时逐个调用原native prepare，完成后的**真实时间**才是ready_ns；长度和compatibility_key由实际prepare返回，不窥探未来请求或输出长度。
2. pending按ready顺序送入现有select_batch。全批、token预算和长度bucket障碍、head超时遵循同一planner；不绕过FIFO head。未满且无障碍时等待min(head maxwait，下一公开arrival)。WAIT事件保存next_event_ns、worker_next_ready_ns和next_arrival_ns。
3. READY调用现有execute_selected_batch。局部adapter把其run_batch映射成**已准备项**的execute_prepared；不再次prepare/tokenize，顺序与身份在前后核验。真实backend一次共同generate，metrics的batch_size/padding_tokens须与planner一致。
4. generate期间到达的请求保留原arrival_ns，owner可用后再prepare；未提前准备或伪造ready。当前单owner也串行执行CPU prepare；非抢占式prepare/generate或Python开销可能使maxwait迟到，报告实际dispatch和timeout_lateness，不能称maxwait为硬实时保证。
5. 所有行在整批处理后一起交付；执行结束记录owner_next_ready_ns。pending被移除前保留实际结果，不漏行、不补虚拟完成、不按EOS时间提前交付。直到公开分母完整drain。

输出逐request记录pre_prepare_wait、preparation、formation_wait、queue_wait、end_to_end，以及到达/prepare开始/ready/dispatch/完成时间；batch记录实际owner_busy、planner选择、padding和backend成本/事件。service_ms为批共享执行时间不可逐行相加；完整端到端以运行时的arrival到交付为准。

父receipt在加载前、每次prepare/批执行前以及最终检查，文件hash变化会停止。公共sample与arrival表hash、configuration、owner身份、factory_calls、加载时间、全部事件、完成prefix均保留。output使用独占新文件创建；异常以BatchOwnerFailure保留原cause、report与traceback，报告active_prepare/active_batch/pending/arrived/future各身份及原requested分母，禁止自动重试/降级/换模型。

这个运行时默认标记not_for_training=true、final_gain_eligible=false；它负责真实执行闭环，不自行授予GPU/训练数据资格。CPU假执行没有物理证据；真实collector仍须绑定源闭包、模型/配置、四驻留与背景owner、NVML/功率/显存、质量及独立数据划分。当前模块只拥有一个text owner，不会暗中启动G2/OCR，也不能冒充已经完成四驻留共跑实验。

## CPU端到端验证

[test_batch_owner_runtime_v2.py](/root/autodl-tmp/generalist-specialist-project/tests/test_batch_owner_runtime_v2.py) 10/10 PASS。使用可推进monotonic clock以及**SharedTextBatchExecutor + fake tokenizer/model**；prepare与generate分别推进真实测试时钟，不直接伪造交付结果。

- 公开三请求：ready=2/7/14，dispatch=7/7/24，complete=12/12/29；一次owner load，三次prepare、两次generate，尾批maxwait触发，完整身份/等待计数核验。
- 长短bucket屏障、无序arrival/同到达tie、future不可提前参与、不可抢占准备导致超时迟到。
- prepare失败保留active/pending/future，第二批generate失败保留首批已完成prefix，原异常/分母/单次load不变。
- 父receipt漂移在下一个prepare前停止并保存失败report；新输出不覆盖；无效公开输入不启动factory；真实入口默认不load。

命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m unittest discover -s tests -p test_batch_owner_runtime_v2.py -v`。未启动GPU或修改任何冻结执行/训练文件。
