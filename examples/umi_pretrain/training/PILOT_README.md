# 60 类任务覆盖型 pilot

本轮把已有目录标签直接用于训练清单，不等待人工分类或视频复核。模型、动作表示、视频时间策略和完整 checkpoint 沿用此前实现。

## 数据选择

- 按 60 个任务目录类别，各选 4 条训练、2 条验证 episode；优先划分原始 MCAP 来源组。
- 当前目录的任务类别本身包含场景信息。保留场景与任务交叉表，不额外强制七个场景等量。
- 从历史 QwenPI 训练计划读出旧训练视图，保守排除其中全部来源；另纳入早期 private debug 使用的 episode 0。验证来源不得与它们重叠。
- 用种子 42 的内容排序固定选择。原始任务文本不改写为类别名。
- 每条 episode 从已有有效起点中等距选最多 32 个；起点间至少相差 16 帧，短轨迹不重复补齐。
- 训练视图逐窗口打乱。完整验证视图固定。此 pilot 是任务覆盖型选择，不代表全量自然分布。
- 全量第一版的五个阶段今后按共同训练池的自然有效窗口分布匹配，不由这个 pilot 的配额决定。

## 文件分工

1. `build_umi_pilot.py`：读取既有目录清单、紧凑有效范围、历史训练计划，写入两个独立的小索引、episode 清单、配额报告和训练计划。不复制视频。
2. 既有 `compute_umi_normalization.py`：仅在训练索引上拟合 state/action mean/std；验证复用同一统计文件。统计按训练窗口中的实际曝光加权。
3. `train_qwenpi_pilot.py`：为已有正式 trainer 提供固定评测回调。训练期间固定训练子集、独立验证子集；完成时完整验证。
4. `train_umi_pretrain.py`：仅新增可选启动和评测回调，默认行为不变；优化器、学习率、采样恢复和 checkpoint 仍由原程序负责。

## 训练和评测

全新 Qwen3-VL-2B-Instruct 感知初始化与随机动作头，双 LayerNorm、连续 state；单 PPU，batch 1、累积 2，共 10,000 次更新。VLM 峰值学习率 1e-5，动作头 1e-4，预热 10 步，之后线性下降，4 步动作采样。沿用完整 FP32 参数与已验证的精度接口。

每 1,000 次更新保存 basic 完整续训状态。它仍包含 Adam、scheduler、RNG 和数据游标；大权重文件不做额外全文件内容摘要。

评测时点为 0、1,000、2,500、5,000、7,500、10,000：

- 曲线训练子集：每任务固定 1 条训练 episode，各取 4 个分散窗口，目标 240。
- 曲线验证子集：每条独立验证 episode 各取 4 个分散窗口，目标 480。
- 第 10,000 次更新额外评估完整验证视图，目标 3,840 窗口。

报告 pooled 位置/夹爪 RMSE、旋转角误差、逐任务指标、各任务指标的算术平均和本视图自己的保持状态基线。固定评测种子并恢复训练 RNG。报告原始物理表示上的误差；四元数角度计算处理 q/-q 等价。无效四元数另计数，不伪装为正确旋转。

曲线与完整验证分开命名，不把前者称为完整验证。没有按软指标强制早停；读取异常或非有限数值仍明确报错。两条验证轨迹不能支撑稳定的单任务性能结论，离线动作误差也不是闭环任务成功率。

## 执行

以下路径均在私人空间。公共数据和共享环境不修改。

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
source /usr/local/PPU_SDK/envsetup.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python
PILOT=/mnt/workspace/Native_Policy/user/wyt/data_preparation/roban_umi_pilot_v1

# 两个固定视图生成后，只拟合训练视图。
$PY -u examples/umi_pretrain/tools/compute_umi_normalization.py \
  --index-dir "$PILOT/train" --output "$PILOT/statistics.json" --purpose engineering

# 新任务；输出目录应尚不存在。
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
$PY -u examples/umi_pretrain/tools/train_qwenpi_pilot.py \
  --plan "$PILOT/plan.yaml" \
  --output-dir /mnt/workspace/Native_Policy/user/wyt/runs/qwenpi_task_coverage_pilot_v1
```

恢复时保持同一计划、代码和统计，向最后一条命令增加 `--resume latest`；不要再次执行生成工具覆盖清单，也不要把旧 32 窗口 checkpoint 当成这次的初始化。

清单、索引、统计在服务器私人目录；代码提交只包含工具与说明，不把样本或模型权重 push 到 GitHub。
