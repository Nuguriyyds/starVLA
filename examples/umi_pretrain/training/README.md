# UMI 分阶段训练与完整续训

后续有界长试跑见 [EXTENDED_LEARNING_STATUS.md](EXTENDED_LEARNING_STATUS.md)：解析采样检查与当前双 LN 配置的 2000 更新结果，取消 50 步软门槛。

后续对照见 [CONDITION_NORM_STATUS.md](CONDITION_NORM_STATUS.md)：条件 K/V 输入归一化已测，入口尺度恢复，但 50 步仍未学会有效去噪，按约定暂停。

最新动作模型诊断见 [DECODER_STATUS.md](DECODER_STATUS.md)：旧终态 bias-only 已确认；单独增加解码器输入 LayerNorm 后完成 200 步，但动作误差仍未达到基线。

入口：`python -m starVLA.training.train_umi_pretrain`。

这一版复用 QwenPI、索引 Dataset、固定归一化和 Accelerate。原有
`train_starvla.py`、debug 训练入口、动作头、字段、过滤规则均保留。
工程配置只用于训练程序验收，不代表语义划分获批，也不启动全量预训练。

本轮实际结果见[验收记录](VALIDATION.md)，其中区分CPU精确对照、真实模型
短程更新与尚未验证的后端/规模。

真实双 PPU 结果见 [DDP 工程验收](DDP_STATUS.md)：数据分工、阶段切换和恢复进度检查通过，但当前运行时清理发生 SIGSEGV，数值重复性也尚未通过；不能视为双卡完整验收通过。

单卡 200 次更新结果见 [可学习性试跑](LEARNABILITY_STATUS.md)：正常退出，但未胜保持姿态基线，输出形式提示动作解码器退化。视频现采用固定 1 ms 候选边界余量，详见 [视频策略](VIDEO_BOUNDARY_STATUS.md)。

## 支持范围

- 单设备以及普通 DDP；CPU 双进程用 Gloo，设备 DDP 用 NCCL 兼容接口。
- 参数及 AdamW 状态保留 FP32，AdamW `fused=False, foreach=False`；可选择
  Accelerate `no`/`bf16`。QwenPI 保留已验证的 VLM→动作头 FP32 桥接。
- 完整更新边界的保存、停止、恢复；同一计划中的多个数据阶段连续优化。
- 当前 Dataset 读取和缩放均为确定性操作；不承诺新增随机增强后的精确恢复。
- **本版明确拒绝 DeepSpeed/ZeRO、FSDP、FP16 scaler 和改变卡数的恢复。**
  这些后端的保存格式、优化器及 scheduler 所有权需另做验收，不由普通 DDP
  测试推导支持。此次不涉及 64 卡通信或吞吐。
- 共享 POSIX 文件系统；训练只写 `--output-dir`。同一个 run 用文件锁防止
  两个任务同时写入，进程退出后锁自动释放。

## 两种工程配置

`train_files/umi_training_tiny.yaml`：CPU 小模型、AdamW、12 次更新，A/B 各6次，
每进程 batch=2、梯度累积=2、worker=2。N=23/27 特意不能被全局 batch 整除，
用于验证尾部和 epoch。A/B 是不同的合成数据视图。

`train_files/umi_training_qwenpi_engineering.yaml`：真实 Qwen3-VL-2B-Instruct，
四路当前图像、世界系双手位姿与开度、未来1～16帧，沿用固定工程统计量。
两个工程视图对应 episode 325202/325203（通过原索引编译器生成）。A/B各2次
更新、累积=2，用于建立 Adam 状态、写完整 checkpoint、退出后加载及切阶段。
工程评测使用独立 loader，但与训练工程数据重叠，**不能解释为泛化性能**。

所有阶段共用同一个归一化文件；启动不拟合统计量。正式模式需要现有归一化
模块认可的 formal 统计与实验合同，工程统计不能直接改标签冒充。

## 启动和暂停

在仓库根目录、已经验证的私人环境中运行。例如：

```bash
python -m starVLA.training.train_umi_pretrain \
  --plan examples/umi_pretrain/train_files/umi_training_qwenpi_engineering.yaml \
  --output-dir /mnt/workspace/Native_Policy/user/wyt/runs/umi_staged_engineering \
  --stop-after-update 1
```

新进程继续同一 run：

```bash
python -m starVLA.training.train_umi_pretrain \
  --plan examples/umi_pretrain/train_files/umi_training_qwenpi_engineering.yaml \
  --output-dir /mnt/workspace/Native_Policy/user/wyt/runs/umi_staged_engineering \
  --resume latest
```

PPU 运行前仍需加载平台 SDK 环境并选择空闲设备；使用已有环境，不重新安装
依赖。CPU 加 `--cpu`，普通多进程使用 `torchrun`/`python -m torch.distributed.run`。

`--stop-after-update` 是本次进程暂停点，不改变完整计划。SIGINT/SIGTERM 只
设置停止请求，所有 rank 在下一次完整更新后保存退出；启动前/文件写入期
被强杀只能使用此前已发布的 checkpoint。首次运行输出目录必须不存在，
显式 `--resume` 缺失、损坏或身份不匹配必须失败，绝不退回重新初始化。

## 样本进度和阶段

全局更新 batch = 进程数 × 每进程 batch × 累积次数。每个 epoch 仅消费能
组成完整更新的前缀，剩余尾部计入 `dropped_tail_samples`，不补重复样本，
不跨 epoch 或 stage 累积。视图不足一次完整更新则报错。

采样器只生成全局流，Accelerate 按进程分一次；不另加 DistributedSampler
或 skip_first_batches。worker 预取不提交进度。一次有限 loss/梯度对应的
optimizer update 成功后，scheduler 只推进一次，再提交样本游标。

进度记录表示**下一次更新的位置**：A完成后保存时，stage_index已经指向B。
切阶段仅替换 loader，关闭旧 worker、文件和视频缓存；模型、Adam动量、
scheduler、归一化、全局 step 继续沿用。重复 set_epoch(同一个epoch) 不清空
已恢复游标；显式 `start_index=0` 才重置。

epoch 被预算截断时尚未访问的大段样本不记作尾部；只有完成可执行前缀后
不足一个全局更新的余数记作尾部。曝光次数不是去重帧数或数据小时数。

## 输出与完整性

```text
run_identity.json                 内容身份、代码版本、运行与参数组约束
plan_requested.json               完整计划，暂停点不在此计划中
normalization/statistics.json     实际统计文件的原字节副本
data_access/all_views.json        所有阶段和评测的来源
data_access/stage_A/...           每阶段来源记录，互不覆盖
checkpoints/update_XXXXXXXX/      模型+optimizer+进度+scheduler+每rank RNG
  manifest.json                  校验模式、逐文件大小/校验方式、保存触发原因
  COMPLETED.json                 仅完整写入后发布
latest.json                      指向最近完整发布点
exports/                         预留的推理导出位置，不可当作resume
updates.jsonl / progress.json     以已提交更新计数的日志与进度
trace_rank_N.jsonl                工程样本追踪，正式运行可关闭
```

保存时所有 rank 调用 `accelerator.save_state()`，进度和唯一所有者 scheduler
显式注册为 custom checkpoint。额外严格保存并恢复 Python、NumPy、Torch
CPU/设备与 loader generator：Accelerate 某些版本会把 RNG 恢复异常降为日志，
此入口不接受这种静默降级。数据迭代器创建和评测保留训练 RNG；当前 worker
只执行确定性变换，base seed由stage/epoch确定。

先在隐藏临时目录完成所有 rank 文件，再按校验模式处理摘要，写完成标记、发布
目录及 latest 指针。半成品目录不参与 latest 选择；不自动回退到更早点。
强制中断后未持久化的更新可能重算，恢复时工程 trace/日志去除这些未提交
记录。默认 basic 只对模型/optimizer 检查大小，其余状态文件保留 SHA-256；
full 对全部文件计算摘要。旧 v1 检查点继续按全量摘要验证，不自动降级。
文件与目录执行 fsync，但远端存储的持久性仍依赖其服务端保证。

配置、剩余风险和本轮验证见 [CHECKPOINT_POLICY.md](CHECKPOINT_POLICY.md)。
保存频率现在使用 `checkpoint.every_updates`，工程 YAML 仍保持高频验收；
旧 `training.save_every` 可用，但两者不能同时出现。

首次严格恢复要求相同代码内容/commit、模型工件、数据视图、规则、表示、
归一化内容、完整阶段计划、参数组、world size、batch、累积和关键库版本。
归一化和视图以内容识别，不能只验证路径。权重文件本身不满足完整恢复条件。

## 可重复验收

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python \
  examples/umi_pretrain/tools/check_umi_training_resume.py \
  --output-dir /mnt/workspace/Native_Policy/user/wyt/runs/umi_cpu_acceptance \
  --world-size 1

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python \
  examples/umi_pretrain/tools/check_umi_training_resume.py \
  --output-dir /mnt/workspace/Native_Policy/user/wyt/runs/umi_ddp_acceptance \
  --world-size 2
```

两条测试都运行实际入口：连续12次更新，对比在3/6/9步退出并用新进程恢复；
比较最终模型、Adam状态、scheduler、所有rank RNG，以及逐更新样本与LR。
另将各rank样本交织还原，独立核对 sampler 的全局流，防止两条路径犯同样
的重复/跳样本错误。拒绝半成品、权重冒充、改变视图/计划和formal误用合成数据。

`tests/test_umi_training_state.py` 和 `tests/test_umi_checkpoint.py` 另覆盖尾部、
无效状态、保存中途故障注入、latest不被半成品覆盖，以及统计哈希/卡数变更
和等长文件损坏。旧 sampler/normalization factory 测试作为本轮改动的回归。

## 可关闭的性能测量

默认没有性能采集。在工程 plan 中设置 `performance.enabled: true` 才启用。
`warmup_updates: 2`、`detail_updates: 2` 表示前两次更新用设备 events 分解
forward/backward/clip/optimizer；后续只在连续测量区间边界同步设备。
这些时间不属于训练恢复状态。性能开启/关闭不消耗模型或采样器 RNG。

`max_training_seconds` 在完整更新边界触发正常暂停并保存完整 checkpoint；
不是强制杀进程期限，预加载、单次更新、评测和安全保存仍可能超出它。
修改 performance 配置也属于新 plan，请用新 run，勿修改旧 checkpoint 身份。

三种测量共用以下参数（同一输出目录的配置必须一致）：

```bash
ROOT=/mnt/workspace/Native_Policy/user/wyt
PLAN=examples/umi_pretrain/train_files/umi_training_qwenpi_engineering.yaml
OUT=$ROOT/runs/umi_performance_new
INDEX=$ROOT/data_preparation/roban_umi_access_v1

python examples/umi_pretrain/tools/profile_umi_pipeline.py loader \
  --plan "$PLAN" --output-dir "$OUT" --candidate-index "$INDEX"
python examples/umi_pretrain/tools/profile_umi_pipeline.py training \
  --plan "$PLAN" --output-dir "$OUT" --candidate-index "$INDEX"
python examples/umi_pretrain/tools/profile_umi_pipeline.py report \
  --plan "$PLAN" --output-dir "$OUT" --candidate-index "$INDEX"
```

先使用现有 PPU SDK 环境和私人 Python。loader 模式只读全量候选池，比较
sampler 实际顺序与固定跨文件压力序列，各测 worker=0/2/4、prefetch=2。
默认实际顺序256窗口、跨文件48窗口，每组稳定读取最多约180秒。记录实际
交付前缀，未完成相同数量时不能忽视内容差异直接比较速度。

training 模式只对已有工程视图更新参数，调用本训练入口，不另写优化循环。
两组均 batch=1、累积=2，继承 FP32 参数/AdamW 和现有 VLM 内部 BF16 路径，
默认2次预热+20次更新，最后一次跨入B阶段，结尾评测一次。阶段结束与最终
结束各完整保存一次，按 plan 的 basic/full 策略校验，保留 fsync/完成标记。
历史 PERFORMANCE.md 的 42 GiB 结果使用旧版全量 SHA，不应直接当作新版测量。
另启动新进程加载已完成的
端到端 run，单独记录恢复校验与加载成本，不继续更新。

回放组提前解码各阶段 sampler 前64个样本（PIL/语言/归一化数值），仍使用
原样本 ID、原 sampler、原视图长度；没有缓存 VLM 特征或冻结参数。
缓存外访问直接报错，不替换样本。该模式仅支持工程单进程、worker=0，避免
多 worker 复制缓存。端到端组 worker=2。报告核对两组样本流指纹；若时间
预算导致更新数不同，需要匹配共同前缀另作比较。

产物为 `performance_config.json`、`loader_benchmark.json`、
`training_benchmark.json`、`PERFORMANCE.md`。训练 run 保留每进程独立性能
报告，包含启动/阶段首批、等待/rank检查、设备/主机资源、评测、保存和恢复。
RSS 为进程 RSS 相加可能重复计算共享页；设备总占用不限于 Torch 分配。
host 小段可能与设备或 worker 重叠，不能相加冒充纯设备时间。稳定窗口曝光
吞吐与包含启动/保存的进程吞吐分别报告，均不是独立数据小时数。

重启 worker 不是冷存储测试，本工具不会清空共享 OS/存储缓存。
`check_umi_training_resume.py --performance` 额外逐位比较开启/关闭计时的
最终参数、Adam、scheduler 与 RNG，并保留原先的重启续训对照。
