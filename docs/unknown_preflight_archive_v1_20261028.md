# 四份 UNKNOWN 预检日志的精确无损归档

工具 `unknown_preflight_archive_v1.py` 只允许 v16 preflight02 的四份固定 forecast JSON，路径、SHA256、长度和对应 UNKNOWN report SHA 均在源码中逐项固定。它没有删除原始文件的入口，也不处理 training/raw、模型或 GPU。`plan` 只读小 report 与文件 stat；`archive --execute` 和 `restore --execute` 是分别显式执行的入口。本次只执行 tiny fixtures 和真实 metadata dryrun，未归档或删除原四文件。

归档逐文件以 1 MiB 缓冲读取原字节和计算 SHA256，写入新独占目录的 gzip level 6。压缩完成先 flush/fsync 文件和父目录，再完整解压一次，验证 gzip CRC、全部解压字节 SHA256 与长度；随后核验压缩文件自身 SHA 和 stat 在验证前后未改变。原路径、原 SHA、长度及 device/inode/mode/nlink/mtime/ctime 记录保存在 manifest。所有四份均验证通过、原 stat 和工具源仍一致，才能写入并 fsync 最终 manifest；部分失败只有失败回执，不产生成功 manifest。访问时间不在冻结 stat 中。

还原要求提供 manifest 的 SHA256，重新核对严格四文件 allowlist、压缩文件回执和 stat。压缩文件路径必须与 manifest 同目录并使用规定名称。只向原始固定路径还原，全部原路径必须不存在；先完整写入同目录的专用临时文件并验证解压 SHA/长度和归档未变，fsync 后通过拒绝覆盖的原子 `link()` 发布，再 fsync 目录。既有普通文件或任何 symlink 都拒绝覆盖，检查后的并发创建也不能被覆盖。还原保证原字节与原 SHA，不伪造历史 inode、mtime 或 ctime；这些保留在 manifest 中。

`plan` 给出原总字节数、当前磁盘空间以及“原大小加 1% 加每文件 1 MiB”的保守不可压缩场景。这只是存储场景，并非 gzip 编码的正式上界或实际压缩比例。已有 4 MiB 独立采样说明内容可压缩，但不能把片段比例当作完整日志测量。归档本身释放空间为零，因为原文件保持原位；后续若决定移除原始副本，必须另行授权和设计该步骤，并保持 manifest 与可恢复归档。这个工具不执行该步骤。

后续流程仍以原 JSON 回执为依据；压缩文件不能冒充原回执。需要读历史原路径的流程可先精确还原，或另行实现明确绑定原 SHA 与归档 SHA 的解压读接口，不能修改既有历史回执。UNKNOWN 日志归档不产生 finish、P95、效用或分流结论。
