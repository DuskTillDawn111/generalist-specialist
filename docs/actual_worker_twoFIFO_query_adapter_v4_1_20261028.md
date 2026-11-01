# Query-bound v4 view（CPU）

`actual_worker_twoFIFO_query_adapter_v4_1.query_bound_view` 核验 caller 的完整有序 public cohort
与 v4 输出的 request ID、owner、task、pool、arrival 完全一致，再核验原 measurement window 与固定 drain。
SHA256 fingerprint 绑定这些字段及原边界。删请求、改内容键/route/arrival、改顺序或边界都会失败。
wrapper 保留 v4 schema、source FIFO match 与所有原始资格标志，新增 `source_FIFO_order_match_only=True`；
query identity 验证不授予 native 或预测资格。
`query_bound_resource_view` 在该验证后调用显式 resource schema adapter，不改实际 macro intervals。

补充非零 window start 回归：world start=100、arrival=100/101、显式 initial sender/worker release=0 时，
parent submit=100/101，worker START=101/112。旧 release 边界不会产生过去事件：
`submit=max(arrival+nonnegative_enqueue_extra,previous_write)+nonnegative_sender_extra >= arrival >= world_start`；
`START=max(previous_READY,line_available>=submit)+nonnegative_worker_pre >= world_start`。
初始已有 in-flight handler 的完整 macro 轨迹不在此初始边界接口中；历史状态认证仍不成立。

新增 2 项测试；连同 FIFO、partial-identification 和 v4 合计 24 项通过。
冻结 v4 源和原报告未变。仅 CPU，无 GPU、验证标签或新 native 采集；主 provider 资格仍为 false。
