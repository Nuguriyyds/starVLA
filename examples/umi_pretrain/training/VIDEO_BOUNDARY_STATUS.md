# 视频时间映射：诊断完成，读取政策尚未验收

基线 `061639c`，复核日期 2026-09-20。Checkpoint basic/full 不改动。
本轮只扩展离线诊断，正式 `UMIIndexedDataset` / `VideoTimeWindow` 保持原样。
当前微秒向下量化是待复核政策，不能把其区间唯一性当作真实源帧归属证据。

## 实际结果

完整原始结果：`time_mapping_validation/video_time_diagnosis.json`。
便于阅读的派生汇总：`time_mapping_validation/summary.json`。

| 检查 | 结果 |
|---|---|
| 原 48 次请求，worker=0 | 48 条都有结果，39 成功、9 失败 |
| 原 48 次请求，worker=2 | 48 条都有结果，39 成功、9 失败 |
| 两组结果比较 | 顺序、错误、成功帧定位及像素摘要完全一致 |
| 去重失败 episode | 52630、154072、194878，各重复 3 次 |
| 16 episode × 首/中/尾 × 四路 | 192 路均完成候选诊断，4 路读取失败，全部在首部 |
| 首部读取成功但未选中捕获候选中最近帧 | 36 路；这不是已经判错的 36 帧 |
| 上述 36 路中，较近候选位于同源裁剪前观测范围内 | 20 路 |
| 中部、尾部 | 本组读取均成功，未出现上述最近候选差异；不能外推全量 |

首部的四路失败：

| Episode / 相机 | 被边界拒绝的前帧相对请求 | 下一帧相对请求 |
|---|---:|---:|
| 52630 / head_left | −14.991 µs | +33.427009 ms |
| 52630 / head_right | −1.998 µs | +33.454002 ms |
| 154072 / head_left | −9.330 µs | +33.347670 ms |
| 194878 / head_left | −8.333 µs | +33.456667 ms |

原始结果明确为 `status=failed`，工具退出码为 1。诊断记录完整不代表读取通过。
末尾一处候选搜索没有找到请求后的帧，已记录 `found_both_sides=false`；没有补帧。

## 已经查清的时间关系

对全部 64 路，关联原始 merged source 元数据、curation mapping 与清洗后元数据，
源 MCAP 身份一致，以下公式按保存的数值计算残差均为 0：

```text
curated.from = source.from + source_trim_frame_start / 30
curated.to   = source.from + source_trim_frame_stop  / 30
query        = curated.from + 当前低维行 timestamp
video time   = PTS × time_base
```

这是实际字段之间的核对结果，不是声称已读到转换函数的源代码。

例：episode 0 左头相机的 source.from 为 `1.000008232`，裁掉前 4 条观测，
curated.from 变成 `1.1333415653333334`。视频仍引用同一物理文件。
请求前 `67.565333 µs` 的候选因此落在清洗前的观测区间内，却在清洗后的区间外；
现读取器返回请求后 `33.319435 ms` 的下一帧。

因此，清洗后的 from/to 是随低维裁剪移动的逻辑观测范围，不能直接视为视频
中不同 MCAP 的精确切换边界。未裁剪 episode 的 source.from 自身如何对齐
拼接后的真实 PTS，仍需上游转换定义。

独立上游记录还确认：

- aggregation marker 的 `timestamp_mode` 为
  `ffmpeg_concat_demux_pts_rebase_with_per_source_duration_offsets`。
- `video_mode` 为 `h264_packet_stream_copy_no_reencode`。
- curation provenance 为 `reference_source`，不复制视频字节。
- flatten 记录 source hardlink、curated reference merged source。

这些记录的实际路径、选定字段和文件摘要保存在报告 `upstream_evidence`。
它们确认了处理链存在，但没有提供每个源片段的精确起止 PTS。

## 工具改变了什么

`RecordEveryRequest` 只在本诊断工具中捕获单次 Dataset 异常，保存错误后继续
下一个原请求；没有替换或过滤请求。训练的 `raw[index]` 仍抛出异常。

候选诊断逐相机运行，因此不会因 head_left 失败而漏掉同样有问题的 head_right。
每处保存请求前后各至多三帧、PTS/time_base、当前边界和匹配容差的拒绝原因。
首/中/尾 timestamp 读取实际低维行，不凭名义 FPS 构造请求。

相邻片段从同一 resolved MP4 的元数据中按时间排序查询，不用 episode ID ±1。
本次只扫 source/curated 各 366 个 episode 元数据文件，保留 64 个目标 MP4 的
成员；没有重扫十亿低维行或完整解码视频。每次候选探查最多解码 600 帧。

`nearest_candidate_ignoring_bounds` 仅用于诊断，未作为训练选帧规则。
同一 `contains()` 下“归属唯一”的检查已从正确性结论中移除。
此前对 episode 34300 右头所用“正确 PTS”“仅属于该 episode”的表述过强：
旧测试只证明该政策下可读取且区间归属唯一，没有独立证明物理源帧归属。

## 下一步只补一个上游接口

需要数据转换维护者提供以下任一项：

1. 生成 `videos/*/from_timestamp`、`to_timestamp`，以及 concat 每源时长偏移的代码；
2. MCAP / 源视频片段到合并 MP4 的实际起止 PTS、time_base、时间偏移映射。

优先定位 source episode `56329`（curated `52630`）左头视频
`videos/observation.images.head_left/chunk-002/file-348.mp4`，原采集文件为
`a898beb474714e45b6e230cc618c45ad.mcap`。报告已带其同文件前后片段。
再核对有明确裁剪的 source/curated episode `0` 即可覆盖两种情况。

拿到该定义后，分开“query 的逻辑时间偏移”和“允许读取的物理源片段范围”，
对同一组请求验收。当前不再扩大 epsilon、不重建窗口索引、不删除 episode，
也不另造一套 MCAP/H.264 转换器来猜测原流程。此项不阻碍阅读代码或处理其他
工程工作，但尚不能宣称这套视频读取政策适合正式全量训练。

## 复现（CPU；预期保留 failed 结果）

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python -u \
  examples/umi_pretrain/tools/check_umi_video_boundaries.py \
  --baseline /mnt/workspace/Native_Policy/user/wyt/runs/umi_performance_v1/loader_benchmark.json \
  --source-root /mnt/nas/public/roban_umi/restricted_data/source/umi \
  --output /mnt/workspace/Native_Policy/user/wyt/runs/umi_time_mapping_v2/video_time_diagnosis.json
```

当前真实报告已生成，不需要为了得到相同数字再运行。
