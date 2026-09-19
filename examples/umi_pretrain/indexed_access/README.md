# 全量 UMI 按索引读取

这版把已完成的有效窗口索引接到 QwenPI 的样本接口。沿用现有结构有效性规则，保留四视图、robot1 然后 robot2 的 world-pose-16D、当前 state 和未来第 1～16 帧 action。物理质量待复核项不在此处转化为删除或裁剪规则。

## 代码分工

| 文件 | 作用 |
|---|---|
| `tools/build_umi_access_index.py` | 离线读取全量 catalog 和窗口索引，验证来源和计数，编译运行时目录。只读取元数据及文件 size/mtime，不重扫低维 payload。 |
| `starVLA/dataloader/umi_indexed_dataset.py` | 二分查找有效起点，按真实 segment 和 row-group 读取当前＋未来 16 行；按各相机独立偏移解码当前四视图。 |
| `starVLA/dataloader/umi_sampler.py` | 分块打乱，避免创建十亿个索引的完整随机排列。提供 epoch 和显式消费游标接口。 |
| `starVLA/dataloader/__init__.py` | 增加 `dataset_py: umi_indexed`，原有读取入口仍保留。 |
| `train_files/umi_indexed_data.yaml` | 数据接入配置覆盖项；不是完整训练启动配置。 |

## 一次性构建全量运行时目录

在测试机私人仓库下执行，使用现有虚拟环境：

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python -u \
  examples/umi_pretrain/tools/build_umi_access_index.py
```

默认生成 `../data_preparation/roban_umi_access_v1/`：

- `ranges.npy`：每行是 episode、有效起点区间的首尾，不展开十亿个起点。
- `cumulative.npy`：每段累计窗口数，用于整数索引定位。
- `metadata.sqlite3`：episode 定位、每路视频偏移、任务文字和数值文件信息。
- `meta.json`：规则指纹、源文件与产物记录、全量核对计数。

目标目录已经存在时拒绝覆盖。当前默认全量目录用于数据接入，尚不是已确认的训练集。以后完成任务／场景语义标签与训练／验证划分确认，可使用 `--episode-list 清单.parquet --output 新目录` 构建相应读取视图；清单需要唯一的 `episode_index` 列。

## 一个样本如何产生

```text
整数样本编号
  → 累计窗口数二分查找
  → episode + 有效起点
  → episode_segments 映射到真正的文件行
  → 读取当前行和未来16行
  → 按 task_index 取指令
  → 按每路 video from_timestamp + 当前 timestamp 取图
  → image / lang / state / action / robot_tag
```

`image` 是按 head_left、head_right、wrist_left、wrist_right 排列的四张 RGB PIL 图像，默认 224×224；`state` 为 FP32 `(1,16)`，`action` 为 FP32 `(16,16)`。低维值原样传入，没有转相对坐标、归一化、补尾帧或四元数符号修改。

读取错误会给出具体样本编号并报错，不会悄悄换成另一个样本。`return_metadata: true` 时附带 episode、文件行、未来 frame_index 和视频请求/返回时间，便于排查；训练可关闭。

## 配置接入

将覆盖项与已有 QwenPI 配置合并：

```python
from omegaconf import OmegaConf
cfg = OmegaConf.merge(
    OmegaConf.load("现有QwenPI训练配置.yaml"),
    OmegaConf.load("examples/umi_pretrain/train_files/umi_indexed_data.yaml"),
)
```

现有 trainer 会通过 `build_dataloader()` 选择新入口。factory 在私人运行目录写 `dataset_access.json` 记录实际数据版本，不向公共源目录写统计、缓存或标注。

每个 worker 的 Arrow 保留缓存默认上限 128 MiB，最多 4 个数值文件句柄、8 个视频句柄；SQLite 只读。128 MiB 不是 worker 总内存上限，正在解码的 row-group、视频和预取样本也占内存。worker 数、batch 和缓存容量都是可配置项，没有写死 64 卡。

当前 curated 数据的 `videos/` 通过目录链接指向 `restricted_data/source/umi/videos`。配置中的 `allowed_video_roots` 明确允许只读访问这处已核实的源视频目录；数值 Parquet 仍限定在 `source_root` 内。迁移数据时需要同步更新这些路径，不修改公共链接。

## 当前边界

- 全量目录是候选池，尚未确定最终训练／验证归属。[划分工具](../splits/README.md)目前只生成 task ID／source set 层面的候选清单，尚未检查任务类别／场景语义均衡，不能据此固定正式划分。
- 候选清单默认位于 `../data_preparation/roban_umi_splits_metadata_candidate_v1/`。当前没有构建六个分片视图；可选接口检查需对 `build_umi_split_views.py` 显式传入 `--allow-candidate-splits`，其默认输出为 `../data_preparation/roban_umi_candidate_views_v1/`。接口通过不等于语义划分通过。
- `normalization: none` 保留原始 FP32。可选 `mean_std` 通过显式统计文件和包装层接入，见[归一化接口](../normalization/README.md)；当前只有工程统计，正式训练集统计仍未拟合，也不生成伪造的 `dataset_statistics.json`。
- sampler 的 state_dict 是可用接口，完整 trainer checkpoint 和分布式消费游标恢复尚未接入。Accelerate 可能在迭代入口调用 set_epoch，正式恢复时要统筹设置顺序；不能只保存 sampler 就宣称完整续训完成。
- sampler 产生的全局流不自行按 rank 分片，交给 Accelerate 一次分片。CPU 流测试已覆盖 `even_batches`/`drop_last`；各 rank 须有同一视图、seed、epoch 和消费游标，正式 trainer 仍需统一尾部及恢复计数。
- 分块打乱兼顾文件局部性，不等于十亿个窗口的全局均匀随机排列，也不提供任务均衡采样。
- 数值文件使用 size/mtime 校验。迁移阿里云后若 mtime 改变，需核对数据一致性后显式选择 `source_identity: size`；它不等于文件内容哈希验证。
- 视频按时间定位成功不证明传感器精确同步。解码超出 episode 范围或无法在时间容差内定位会报错，不自动删除窗口。

`view_fingerprint` 在 provenance 和样本 trace 中标识当前视图；旧全量目录可由已有哈希派生，毋须重建。语义标签 schema 由使用者确定任务／场景类别、查看视频和抽查纠错，随后再按确认的标签批量关联、分组分配并核对分布。标签接入与语义平衡尚未实现；工程接口开发可以并行推进。正式统计须等训练池确认后拟合，阶段训练及完整恢复仍待接入。
