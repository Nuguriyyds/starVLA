# Checkpoint 校验策略

完整续训状态与大文件内容校验分别配置。模型、Adam、scheduler、随机状态、
阶段和已消费样本位置，在 basic/full 两种模式下都完整保存。

```yaml
checkpoint:
  integrity: basic       # 日常保存；设为 full 时校验全部文件内容
  every_updates: 4       # 工程验收示例；正式间隔另按训练计划设置
```

配置纳入 run identity。改变策略或周期使用新 run，不改旧 manifest 来绕过
身份检查。兼容只设置 `training.save_every` 的旧计划，默认校验为 basic；
不能同时设置新旧两个周期字段。full 可用于需要全量校验的归档运行。
本轮没有增加自动清理旧检查点，也没有后台异步保存。

| 检查对象 | basic | full |
|---|---|---|
| 模型 `pytorch_model.bin`、Adam `optimizer.bin` | 存在、非空、大小一致 | 左侧检查＋SHA-256 |
| 进度、scheduler、各 rank RNG 等其他状态文件 | SHA-256 | SHA-256 |
| manifest | 完成标记中的 SHA-256 | 完成标记中的 SHA-256 |
| 必要文件、身份、更新步、发布顺序 | 保留 | 保留 |

basic 无法主动检出两个大文件的等长内容损坏；它只记录 `check: size`，
不生成、不假装验证文件内容摘要。full 记录 `check: sha256` 和参考摘要。
缺摘要、未知模式或不符合该模式的逐文件检查方式均报错。

新格式为 `umi-full-checkpoint-v2`。读取时依据检查点自身记录的策略，
而非本次进程期望的策略。旧 `umi-full-checkpoint-v1` 必须继续验证所有
参考摘要。兼容解析旧格式不等于允许用新代码越过旧 run 的代码身份约束。

写入仍遵守：临时目录 → 完整状态 → fsync → manifest → 完成标记 →
目录发布 → latest。周期、暂停、阶段边界、完成可同时触发，但同一更新只
调用一次保存，并把全部原因写入 manifest 和 updates 日志。

首次写后生成摘要只是建立后续比对基准，不证明内存中的状态逻辑正确。
启动时对预训练模型文件建立身份的校验保持原样；本次调整针对 checkpoint
每次保存/恢复额外读取大文件的成本。

## 验证结果（2026-09-20）

- checkpoint/进度/sampler/时间边界共 32 项定向单元测试通过。
- basic 和 full 分别完成真实训练入口的 CPU 12 次更新，worker=2、累积=2；
  在更新 3/6/9 后退出重启，与连续训练的参数、Adam、scheduler、RNG 逐位
  一致，样本顺序与独立重建的 sampler 一致，每种模式 5 类拒绝测试通过。
- 测试明确覆盖 basic 不读取模型/optimizer 计算摘要、full 的等长损坏检测、
  两种模式的缺文件/截断/小状态文件变更，以及旧版参考摘要缺失时拒绝。
- 未为测保存而重新训练 QwenPI 或生成另一份 42 GiB 检查点；新版完整
  QwenPI 保存耗时尚未实测，不能把理论减少的读取量当作已测加速比。

## 只读 I/O 诊断

工具 `examples/umi_pretrain/tools/profile_umi_checkpoint_io.py` 对已有检查点
按文件报告大小，交替执行 read、read+SHA、read+SHA、read。默认每次只读
每个文件前 64 MiB，传 `--max-bytes-per-file 0` 才是全文件。输出必须位于
检查点目录之外；不改旧 manifest、不清共享缓存、不加载模型。

本次检查点位于 NFS。旧 optimizer 约 29.97 GB，模型约 15.22 GB；对各自
64 MiB 前缀，首次读取约 0.47/0.54 秒，随后热缓存 read+SHA 约 0.054～
0.061 秒，最后纯读约 0.011～0.016 秒。这里既有缓存效应，也有摘要计算
成本，不能把不同缓存状态直接相减当作 SHA 的独立成本，更不能外推全文件
耗时。原报告 475.8 秒包含存储读取和摘要计算。

原始报告位于私人目录 `runs/umi_boundary_checkpoint_v1/`；提交中的
`boundary_checkpoint_validation/` 保存本轮验收 JSON。视频回归的剩余失败
独立记录于 `VIDEO_BOUNDARY_STATUS.md`，不把 checkpoint 通过当成数据全通过。
