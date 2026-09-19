# 基于元数据的候选验证集与五个训练阶段

当前工具仅根据有效窗口数量、task ID 和 source set 生成候选划分，尚未检查任务类别与场景语义的均衡，不能将候选清单冻结为最终训练／验证划分。task ID 不等于任务类别，source set 不等于场景。

当前只生成了基于元数据的候选清单，尚未构建六个分片视图。下一步先确定语义标签 schema 和分类口径：使用者负责查看任务文本／视频、确定类别并抽查纠错；之后由程序将确认的标签关联到数据、按来源组分配并报告语义分布，不需要手工逐条划分 32 万条 episode。语义标签导入、传播及其平衡约束尚未实现。

这一轮仍沿用 world-pose-16D、未来 1～16 帧、四视图顺序和既有有效性规则。待解释的物理极值继续保留；不做新的删除、裁剪、归一化或任务重采样。

## 划分定义

- 输入为冻结的 catalog、各文件完成标记、紧凑 anchor ranges 和 episode quality parts。校验哈希与计数，仅检查源数值文件 size/mtime，不读取原始低维 payload 或解码视频。
- `valid_windows > 0` 的 episode 才进入分配；零有效窗口单独登记。
- 不可拆分单位是已有 `source_recording_key`。先将整组分配到候选验证集／训练池，再将训练池分成五份互斥候选阶段。
- 默认种子 42、验证目标为全池有效窗口的 1%、训练阶段数 5。完整组不可拆开，所以报告实际比例，不保证精确 1%。
- 当前分配以有效窗口量为主；按真实 task ID 和 source set 的窗口量做软平衡，默认软约束各 0.15。这是元数据层面的参考结果，不能证明场景／任务语义均衡。语义类别及其目标比例仍需确认，不把稀有任务或小来源自动变成等概率。
- task 权重来自扫描器实际计算的 `task_counts_json.valid_windows`，不是用 episode 长度或任务列表平均估算。重复／遗漏任务权重会报错。
- 时长为有效窗口覆盖的原始行并集数除以记录 FPS；不能把窗口数乘 16 当成独立数据时长。

算法是大组优先、固定种子打破平局的贪心分配。它保留整组约束并限制总量偏差，不声称得到全局最优分布。单例任务不可能同时覆盖多个集合，程序不会为此复制样本。

## 代码和产物

| 文件 | 作用 |
|---|---|
| `tools/umi_split_allocation.py` | 仅处理组 ID、有效窗口量、task/source 权重的确定性分配器。 |
| `tools/build_umi_splits.py` | 验证冻结输入、聚合来源组、分配、核对守恒、读回新清单后原子发布。 |
| `tools/build_umi_split_views.py` | 可选的候选清单接口检查：复用 access builder 编译六个读取目录，核对数量、哈希与身份；须显式允许候选划分。 |
| `tests/test_umi_split_allocation.py` | 覆盖整组不拆、确定性、窗口守恒、巨大组和稀有标签。 |
| `tests/test_umi_sampler.py` | 覆盖顺序、块中恢复和实际 Accelerate BatchSamplerShard 的尾部语义。 |

默认候选清单目录 `../data_preparation/roban_umi_splits_metadata_candidate_v1/`：

- `split_config.json`：种子、目标比例、组键、规则与代码／输入哈希。
- `episode_assignment.parquet`：每条 episode 的唯一归属与来源组、有效窗口数、覆盖行数。
- `train_all_episodes.parquet`、`validation_episodes.parquet`、`stage_01_episodes.parquet`～`stage_05_episodes.parquet`。
- `excluded_no_valid_windows_episodes.parquet`：零有效窗口记录，可以是空表。
- `task_coverage.parquet`：shared、train_only、validation_only、no_valid_windows 四类 ID 覆盖。
- `window_distribution.parquet`：每个集合的 task/source 窗口比例与共同训练池参考比例。
- `split_report.json`：守恒检查、实际数量与比例、分布偏差和产物哈希。

若显式执行候选视图接口检查，默认输出 `../data_preparation/roban_umi_candidate_views_v1/`，含 validation、stage_01～stage_05 六个子目录；每个目录使用原有 ranges＋cumulative＋SQLite 接口。`view_build_report.json` 记录视图的身份、数量与接口验收状态，不证明语义划分已经通过。

## 执行与重复运行

在测试机私人仓库执行，沿用现有环境：

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
PY=/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python

# 仅生成元数据候选划分。已有清单时不覆盖；改变输入应指定新 --output。
"$PY" -u examples/umi_pretrain/tools/build_umi_splits.py

# 不加载 Qwen、不启动训练的性质检查。
CUDA_VISIBLE_DEVICES="" "$PY" tests/test_umi_split_allocation.py
CUDA_VISIBLE_DEVICES="" "$PY" tests/test_umi_sampler.py
```

当前下一步是语义标签 schema 与人工类别确认，不需要默认构建六个视图。只有为检查候选清单与读取器的接口时，才可选执行：

```bash
"$PY" -u examples/umi_pretrain/tools/build_umi_split_views.py \
  --allow-candidate-splits --workers 2
```

该显式参数只允许使用候选清单进行接口检查，不表示同意把它用于正式训练；不提供该参数时拒绝构建候选视图。

失败时保留日志和不完整目录，不把它们发布成可用视图。同一构建目录的配置／代码身份不一致会拒绝继续，应查看失败原因并选择新的版本。构建锁用于防止并发写同一产物，不要在任务尚未退出时移除锁。

读取配置仍使用 `umi_indexed_data.yaml` 的接口，以 `datasets.vla_data.index_dir` 指定视图。当前全量目录与元数据候选视图均不代表已确认的训练集；应完成语义划分确认后，再接入相应阶段与验证视图。验证的固定窗口子集和评估循环尚待实现。

## 身份与续训边界

每个视图增加 `view_fingerprint`：由规则、catalog 记录、ranges/cumulative/SQLite 的产物哈希和读取语义计算。运行时 provenance 和样本 trace 均带此值。相同整数编号在不同视图不代表同一原始样本，续训必须同时保存视图身份和采样状态。

旧全量视图没有该字段时，读取器由既有元数据派生身份，不改写旧产物。身份验证基于记录的哈希；大文件实际哈希在离线构建／验收中核对，不在每个 worker 启动时重读一遍。

采样器在 Accelerate 分片前必须使用相同视图、seed、epoch 和恢复位置；不能把 rank 加到 sampler seed。当前 CPU 流测试确认，在 `split_batches=False` 下：

| 配置 | 尾部行为 |
|---|---|
| `drop_last=False, even_batches=False` | 全局样本各一次，rank 的尾部步数可以不同。 |
| `drop_last=False, even_batches=True` | 用全局流前缀重复补齐完整 rank group。 |
| `drop_last=True` | 丢弃不完整 rank group，包括不能凑成一组的完整单个 batch。 |

这些是整数流契约检查，不是多机训练或完整数值续训验收。训练端的消费计数、Accelerate 的 set_epoch 调用、完整 checkpoint、阶段切换仍需统一实现。

来源组不跨集合只排除了当前映射能识别的同源泄漏；不同路径的内容重复或同 session 尚未排除。下一步先确定语义标签的粒度、类别口径、关联键和审核方式，再实现标签接入与按确认标签分配；具体类别尚未定稿。使用者可以看视频判断并纠错，程序负责批量关联、分组和分布核对。训练集统计／归一化、正式训练恢复及多卡验收继续留待后续。
