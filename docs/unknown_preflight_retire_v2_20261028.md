# 四个 UNKNOWN 副本的可恢复退休工具

`unknown_preflight_retire_v2.py` 是独立入口，冻结的 `unknown_preflight_archive_v1.py` 保持原样。它只允许 v1 中硬编码的 preflight02 四个 UNKNOWN forecast 原路径、SHA 和长度。训练数据、raw、其他 forecast 不在名单内，不能作为目标。

入口为 `inspect`、`retire --execute`、`recover --execute`，均指定 `--manifest`、`--manifest-sha256`、`--output`。inspect 只读全部归档和原副本并写新报告；retire 的新审计目录只能是 artifacts 下 `predictor-v16-preflight02-lossless-retirement-*`，recover 是 `predictor-v16-preflight02-lossless-recovery-*`。所有输出目录独占新建，失败报告保留，重试恢复须换新的 output 后缀。

retire 完整检查 manifest SHA、四个固定原路径、原 UNKNOWN report SHA、v1 producer SHA；完整读取每个原文件的 SHA/长度并比对 lstat，检查压缩文件 SHA/长度/lstat，解压全流验证 gzip CRC、原始 SHA 与长度，并在验证前后重查归档身份。获取 archive-local flock 后再次完成以上检查；锁只序列化协作工具，不声称锁住任意外部写入者。

任何 unlink 前，工具写并 fsync intent、目录，记录所有固定目标及原身份、manifest、归档、完整验证和可直接运行的恢复 argv/shell 命令。每个 unlink 前再次验证该归档完整全流、原文件 SHA/lstat、manifest SHA，并最后重查原 lstat。unlink 后 fsync 原目录，再逐项写 fsync 进度；成功记录与失败记录另存。失败报告直接检查所有原路径是否缺失，因此 unlink 成功但 fsync 或进度写入失败也不会误报文件还存在。协作锁加最后 lstat 不提供对任意外部写入者的原子 compare-and-unlink；运行时必须保持该证据域无其他写入。

recover 支持全部或部分退休。它先完整验证四个归档和 manifest/producer/report，任何已存在的原文件必须是非 symlink 普通文件，内容 SHA/长度精确一致；已恢复文件的 inode/时间戳允许与历史不同。它在 fsync 恢复 intent 后只恢复缺失目标：完整解压、内容验证、fsync，再用拒绝覆盖的原子 link 发布。已有原路径从不覆盖。最后重新完整验证四个归档和四个原文件，写成功/失败报告。失败可以用同一 manifest 和新的 output 再恢复，归档始终保留。

CPU 小测试覆盖四个临时 opaque binary 流的退休与恢复、只读 inspect、durable intent、部分退休、unlink 后 fsync/receipt 故障、原文件/归档在 intent 后变化、gzip CRC 损坏但更新 compressed SHA/stat、report/producer 改变、原路径 symlink/内容改变、训练/raw 越界、manifest SHA 和执行开关。测试从未压缩、删除或修改真实四个大文件，GPU 未启动。真实操作仅由主对话在独审后排程。
