# 双 PPU DDP 工程验收（2026-09-20）

测试使用生产代码基线 `34023b56bc6275baff0f08cde51835af8321d657`，保持已有字段、归一化、视频政策、模型和 checkpoint 实现。新工具仅组织运行并采集训练状态；未修改共享环境。

## 测试范围

- 同一节点，两张 PPU-ZW810E；普通 DDP，NCCL 兼容后端。
- 每卡 batch 1、梯度累积 2，每次更新全局消费 4 个窗口。
- 已有工程数据视图、固定统计量：A 阶段 2 次更新，B 阶段 2 次更新。
- 比较连续 A→B 与 A 完成后保存、由新进程恢复 B。checkpoint 为 basic。
- 使用真实 Qwen3-VL-2B + QwenPI，参数与 Adam 保留 FP32，沿用 VLM BF16 计算和已有 FP32 特征桥接。

工程视图仅用于训练系统验收，不验证视频物理同步、正式数据划分或模型效果。

## 真实结果

**两组均完成四次更新及 checkpoint 发布，数据进度检查通过；完整验收未通过，因进程清理崩溃且未取得数值逐位重复性。**

| 检查 | 结果 |
|---|---|
| 全局样本流 | 两个 rank 合并后与独立生成的预期流一致，每次更新 4 个不同窗口 |
| 连续与恢复路径 | A,A,B,B；样本、阶段游标、全局步数、学习率一致 |
| 两个 rank 的梯度范数 | 每次更新一致 |
| live 参数/Adam 探针 | 恢复组第 2、4 次更新，1041 个参数张量及已有 Adam 状态抽查一致 |
| 调度器、进度、每 rank 随机状态 | 两组第 2、4 次 checkpoint 中对应状态一致 |
| 最终 Adam | 两组参数分组一致；1025 个有状态参数的 step 均为 4 |
| 模型数值重复性 | differences_measured_not_yet_explained |
| 清理退出 | 三个真实训练进程组均返回 1，worker SIGSEGV |

最终模型逐张量抽查：抽查元素最大绝对差 **0.000290463679**，
差值 RMS **3.4310431e-05**；
optimizer 抽查元素最大绝对差 **0.000108492626**。
这些只代表抽查元素，没有据此声明误差可接受、全量最大误差或跨设备逐位一致。

重启之前，第 2 次更新的 loss 已分别是 **30395.386719 / 30380.564453**，
该时点参数抽查最大差约 **1.50e-4**。因此不能把最终差异全部归因于恢复。
当前仅确认数据/调度/RNG进度对齐；设备运算或其他数值差异的具体来源未定位。

## 保存和恢复成本

同一次更新中不同 rank 的等待时间不能相加。下表列 rank 0 实际区间：

| 进程 | 区间 | 秒 |
|---|---|---:|
| continuous / performance_rank_0_662437.json | checkpoint_total | 88.021 |
| continuous / performance_rank_0_662437.json | checkpoint_total | 87.979 |
| resumed / performance_rank_0_667203.json | checkpoint_total | 90.014 |
| resumed / performance_rank_0_669228.json | resume_verify | 0.099 |
| resumed / performance_rank_0_669228.json | resume_load | 607.527 |
| resumed / performance_rank_0_669228.json | checkpoint_total | 89.249 |

每份完整 checkpoint 约 42.08 GiB。basic 保存约 **88～90 秒**；
恢复校验约 **0.10 秒**，恢复加载约 **607.53 秒**。
保存主要耗时仍是写模型/Adam，恢复包含实际读入、反序列化、状态装载和同步；
不能把整个恢复区间都解释成纯存储传输时间。
设备峰值 allocated 显存约 **76.50 GiB/卡**；DDP 每卡保留完整模型和优化器，
增加卡数不会把这部分显存简单除以卡数。

这是一组短程工程测量，不是长期吞吐或云端性能承诺。

## 通信运行时问题

训练完成并发布 checkpoint 后，当前 PPU 运行时在标准 `destroy_process_group()` 清理时发生 SIGSEGV，torchrun 返回 1。因此本轮不能称为整项 DDP 验收通过。

不加载 Qwen、UMI、Accelerate 或 checkpoint 的 16×16 Linear + AdamW 双卡程序也能复现，栈落在已安装 Torch 的 `distributed_c10d.py:2135`，调用 `pg_to_shutdown.shutdown()`。这将问题缩小到当前分布式运行时的清理路径，尚不能定位具体库的内部根因。

额外只检查了两个变体：让进程自然退出、在最终 barrier 后再同步设备，均仍以 SIGSEGV 退出。本轮到此收尾，不继续切换环境、升级依赖或尝试绕过清理。

原始复现代码和运行日志保留在私人运行目录。仓库的 `probe_ppu_ddp_shutdown.py` 是补充说明和 main 入口后的同一计算步骤，未额外运行这一整理版。

## 报告边界

- `rank_probe` 抽查每个参数张量及已有 Adam 一、二阶矩的首、中、尾元素；抽查相等不等于全张量逐位相等。
- 第一组连续运行在清理后才准备采集探针，因崩溃未取得该组 live rank probe；缺失明确保留。
- 后两次运行将探针移到标准清理之前，仍调用原清理并保留失败退出码。没有用强制成功退出掩盖问题。
- 保存成本来自训练中的计时；额外离线比较的成本单独记录，不计入 checkpoint 保存或恢复，也不是新增的周期性哈希要求。

## 完整复现入口

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
source /usr/local/PPU_SDK/envsetup.sh
CUDA_VISIBLE_DEVICES=0,1 \
/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python -u \
examples/umi_pretrain/tools/check_qwenpi_ddp.py run \
--output-dir /mnt/workspace/Native_Policy/user/wyt/runs/umi_qwenpi_ddp_NEW
```

输出目录必须是新目录。默认遇到进程错误停止；本次后两阶段是确认最小复现后显式继续执行的，退出码仍保留，不是工具自动忽略错误。

当前运行目录：`/mnt/workspace/Native_Policy/user/wyt/runs/umi_qwenpi_ddp_v1`。
实际运行命令及退出码见 `processes.json`；原始 plan、trace、性能记录、checkpoint 和 probe 均保留于该目录。

本次 run identity 记录的是测试时的代码提交 `34023b5`。后续提交新增验收工具与文档不会改写旧 identity；在新提交上进行新的恢复验收应新建运行目录，不能修改旧身份文件使其强行兼容。

## 本轮收尾与下一步

视频上游问题暂挂，不重复原诊断，也不修改微秒阈值。已有生产训练路径保持不变。
当前通信运行时问题保留最小复现，后续在正式云端环境核验时处理；本轮不继续试装环境。
数值重复性仍列为未验证，不拿本轮模型指标证明预训练有效。数据语义划分和正式计划可继续准备。

结果文件：[acceptance.json](ddp_validation/acceptance.json)，实际命令与退出码
[processes.json](ddp_validation/processes.json)，独立复现证据
[shutdown_diagnosis.json](ddp_validation/shutdown_diagnosis.json)。

完整逐参数 identity 与逐区间 performance 保留在原运行目录，仓库只收录 identity 摘要、验收结果和 trace，避免重复提交大型清单。已中止的额外全张量比较不作为验收结果，最终报告仅含逐张量抽查。
