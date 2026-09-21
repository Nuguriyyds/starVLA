# 正式 UMI 预训练

模型继续使用双 LN QwenPI、连续 state 和修复后的注意力路径。首次启动加载原 Qwen 感知权重，动作头重新初始化，不加载 pilot checkpoint。五个阶段共用一套模型、Adam、学习率调度和正式统计。

## 已冻结的数据

`data_preparation/roban_umi_formal_v1/` 中：

| 产物 | 作用 |
| --- | --- |
| `formal_manifest.json` | 固定来源组、划分依据、数量与分布 |
| `manifests/train.csv.gz`、`validation.csv.gz` | 训练/验证 episode 清单 |
| `manifests/stage_01.csv.gz` 至 `stage_05.csv.gz` | 五个阶段的来源组与目录标签 |
| `views/train` | 五阶段并集，拟合正式统计用 |
| `views/stage_01` 至 `stage_05` | 全部有效窗口，直接供现有 Dataset 读取 |
| `views/validation` | 原120条验证轨迹同源组的全部有效窗口 |
| `views/validation_monitor` | 从原验证窗口固定选取每条4个，共480个，周期性监控 |
| `normalization_experiment.json` | 训练池批准依据、统计拟合视图和允许应用的视图 |
| `statistics.json` | 全部训练窗口加权 mean/std，完成计算后才出现 |
| `statistics.json.work/` | 按物理低维文件保存的统计进度，支持续算 |
| `distribution.csv` | 五阶段与共同训练池的任务/场景比例 |

只使用已经接受的目录标签。原始指令仍通过原任务表读取；不把细节指令替换成类别名。不增加有效性过滤，不删除待决质量项。原120条验证轨迹的同源组先隔离，其他来源组全部进训练，包括以前参与 pilot 的训练组；验证集仍是开发验证集。

每个视图的 `ranges.npy` 决定样本成员，区间 `[start,end)` 表示有效起点，不展开十亿行窗口。`metadata.sqlite3` 是只读完整父索引的硬链接（不支持硬链接时复制）；其中存在未选 episode 的查询元数据不表示它们参与了采样/统计。不要原地修改这些数据库。源视频、低维 Parquet 不复制、不修改。启动核对时，同一物理元数据文件只读一遍计算摘要，每个视图仍核对自己的预期摘要，避免重复读取约1.5GB的同一数据库。

正式统计对每个窗口的当前 state 计一次、未来16个 action 分别计一次；预期 state 权重为训练窗口数，action 权重为其16倍。只扫描低维数据，不解码视频。五阶段和验证都用这份统计，不重新拟合。

## 离线统计

本次已在私人目录生成 `run_statistics.sh` 并提交后台。查看：

```bash
PREP=/mnt/workspace/Native_Policy/user/wyt/data_preparation/roban_umi_formal_v1
tail -n 20 "$PREP/statistics.log"
find "$PREP/statistics.json.work/completed" -maxdepth 1 -type f -name '*.json' | wc -l
cat "$PREP/statistics.exit_code"
```

`exit_code` 运行中不存在。中断后，确认旧进程已结束，再执行 `bash "$PREP/run_statistics.sh" --resume`，复用已完成文件；不要重复并发启动。完成标志是 `statistics.json` 成功生成且退出码0，不是进程已提交。

## 填入训练预算

`plan.template.yaml` 中总更新数、各阶段更新数和 warmup 尚未填写，不能启动。总预算由团队决定，没有把一万小时解释成遍历一遍，也没有沿用 pilot 的10000步。

默认运行配置为4节点×16进程、每卡batch1、累积2，全局batch128。动作头LR1e-4、VLM LR1e-5沿用已固定配方；全程一次线性warmup再线性衰减，默认warmup占所填总更新数1%，可显式覆盖。每1000更新保存basic，每2000更新和最终更新评测480个固定验证窗口。它们是可在首次启动前调整的配置值，不是最优预算结论。

为避免十亿窗口逐元素随机排列的内存开销，正式训练使用现有 block shuffle，块长512：块顺序及块内顺序均打乱，窗口权重仍相同。每次完整epoch末不足全局batch的部分沿用 trainer 的丢弃并记账规则。五阶段更新预算按窗口数比例以最大余数法分配；不重置Adam或学习率。

团队给出总预算后，在仓库根目录执行（先设置真实 `TOTAL_UPDATES`）：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${TOTAL_UPDATES:?填写团队确定的正式总更新次数}"
python examples/umi_pretrain/tools/render_umi_formal_plan.py \
  --preparation-dir "$PREP" \
  --reference-plan /mnt/workspace/Native_Policy/user/wyt/data_preparation/roban_umi_pilot_v1/plan.yaml \
  --total-updates "$TOTAL_UPDATES" \
  --output "$PREP/plan.yaml"
```

这里 reference-plan 只读取已固定的模型/数据选项和学习率，不恢复 pilot 的权重、统计或优化器。也可用生成的 `plan.template.yaml` 作 reference-plan，避免上云后依赖 pilot 目录。模型迁移后使用 `--model-dir`，数据、视图、统计和输出路径需与正式节点挂载一致。首次启动前定稿并保存，续训时保持同一计划与代码版本。

## 四节点启动和暂停恢复

`formal/platform.env.example` 是平台变量模板，不含虚构地址。创建阿里云任务后，每个节点启动一次下列脚本；`NODE_RANK` 分别为0/1/2/3，不能每张卡再启动一次16进程脚本。

```bash
source /你填写后的路径/platform.env
bash examples/umi_pretrain/formal/start_node.sh
```

所有节点需要同一代码/环境、相同的公共数据/私有索引/模型挂载，以及共享持久化输出目录。通信网卡和PPU通信参数由实际云任务提供，本脚本不猜测、不安装环境、不自动换后端。当前仅支持普通DDP；4×16是配置目标，不是已经验证的64卡运行结论。已知PPU通信清理异常仍未解决，继续由平台处理。

计划在某次完整更新后暂停，在四节点同一任务命令中设置：

```bash
export PAUSE_AFTER_UPDATE=团队选定的全局更新数
bash examples/umi_pretrain/formal/start_node.sh
```

暂停后，四节点使用相同输出目录和原计划，去掉暂停变量再恢复：

```bash
unset PAUSE_AFTER_UPDATE
export RESUME=latest
bash examples/umi_pretrain/formal/start_node.sh
```

`RESUME` 也可指定该 run 的完整checkpoint目录。首次正式启动不设置RESUME。若已在运行，现有trainer捕获训练worker的SIGTERM，完成当前更新并保存；只对核对过的该任务worker PID发信号，不对共享机器批量kill，也不要把torchrun代理进程PID当成worker。云平台强制终止未必给保存留出时间，此时从最后完整checkpoint恢复。

评测复用 pilot 的位置RMSE、夹爪RMSE和四元数角度误差，并计算本验证集的保持状态基线、按任务均值。固定窗口分给各rank，不让64个rank重复评测全部窗口；评测保存/恢复RNG，不因软指标触发早停。480窗口结果是覆盖型开发监控，不等于全量自然分布平均误差，更不是机器人任务成功率。

## 本轮代码阅读顺序

1. `tools/build_umi_formal.py`：固定验证来源组→五阶段划分→紧凑读取视图。
2. `tools/umi_split_allocation.py::allocate_training_stages`：整组分配与比例平衡，不重采样。
3. `tools/compute_umi_normalization.py`：已有的流式窗口加权统计工具，本轮复用。
4. `tools/render_umi_formal_plan.py`：把数据身份、预算和固定模型配方写入plan。
5. `starVLA/training/train_umi_pretrain.py`：已有训练循环，仅补全局batch/进程数核对与正式监控接入。
6. `starVLA/training/trainer_utils/umi_action_evaluation.py`：复用动作指标，按rank分工、恢复随机状态。
7. `formal/start_node.sh`：把平台变量交给torchrun，暂停/恢复复用原trainer。

没有更改QwenPI/动作头、视频策略、有效窗口定义、归一化公式或checkpoint策略。本轮不启动新的pilot或模型消融。
