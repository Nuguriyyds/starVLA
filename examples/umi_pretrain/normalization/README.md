# 低维归一化：工程接口

这版在原始 world-pose-16D 与 QwenPI 之间增加显式的可逆缩放。现有语义划分仍是候选；本轮只用合成数据和明确选择的小型真实视图验证工程接口，不对全量候选池拟合正式统计量。

## 使用位置

```text
原始 Parquet → UMIIndexedDataset（始终原值）
           → 可选 UMINormalizedDataset → QwenPI

原始 state → normalize_state
原始 action → normalize_action → FM 加噪与训练
动作噪声 → 在归一化空间完成所有采样步 → inverse_action → 原数值空间
```

`UMIIndexedDataset.read_lowdim()` 的含义不变，也不会解码图像。包装层通过 `raw_dataset` 暴露底层读取器；`__getitem__()` 另建 state/action 数组，不能污染原始缓存。旧 `normalization: none` 保留原路径，不要求伪造统计文件。

`mean_std` 使用 state/action 各自一套 16 维参数。动作的 16 个未来位置共用同一套逐维尺度；不是逐样本、逐 batch 或逐未来步拟合参数。

计算形式为 `normalized = (raw - mean) / scale`，逆变换为 `raw = normalized * scale + mean`。方差采用总体方差；默认标准差小于 `1e-6` 的维度使用有效 `scale=1`，统计文件明确记录阈值及处理维度。均值和尺度固定，训练启动时不会自动拟合。

这版不调整四元数符号、不变换姿态坐标、不裁剪开度或极值。反归一化只是恢复数值坐标，不保证模型生成的四元数单位范数。

## 文件分工

| 文件 | 职责 |
|---|---|
| `starVLA/dataloader/umi_normalization.py` | FP64 增量矩、统计身份校验、NumPy/Torch 正反变换。 |
| `tools/compute_umi_normalization.py` | 显式选定视图的离线低维统计、文件级进度与恢复。 |
| `starVLA/dataloader/umi_indexed_dataset.py` | factory 按配置添加可选包装层，记录实际模型数值空间。 |
| `train_files/umi_normalization_engineering.yaml` | 工程覆盖配置，不覆盖现有 raw 配置。 |
| `tools/check_qwenpi_normalization.py` | 少量样本前向／反向、完整采样及最终反变换的连接检查。 |

## 统计口径

对 N 个选中有效窗口：state 计数 N；action 将每窗未来 1～16 行合并计数，计数为 16N。同一原始行被多个目标窗口使用时，按出现次数加权，不能偷偷改成去重行等权。

统计工具从 compact ranges 和真实 segments 推导每行整数权重，按物理文件、row-group／批次读取所需低维字段。没有展开十亿窗口，没有调用含视频解码的 `__getitem__()`。选中行先按模型入口转换为 FP32，再用 FP64 count/mean/M2 聚合；未选行不参与统计。

每个完成文件的结果原子保存，最后按固定文件顺序合并。恢复必须匹配视图、源文件身份、代码和参数；失败但未提交的文件重新计算，不重复累计已完成文件。

## 工程用法

先准备明确的小视图，再显式给出路径。下面路径只对应本轮的两条真实 episode 工程选择，不代表最终训练集。

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
PY=/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python
ENGINEERING=/mnt/workspace/Native_Policy/user/wyt/data_preparation/umi_normalization_engineering_v1

"$PY" -u examples/umi_pretrain/tools/compute_umi_normalization.py \
  --index-dir "$ENGINEERING/access" \
  --output "$ENGINEERING/statistics.json" --purpose engineering

# 中断后用相同参数加 --resume；不更换视图或改变数值定义。
```

将模型配置、`umi_indexed_data.yaml`、`umi_normalization_engineering.yaml` 依次合并。factory 必须得到 `normalization_statistics`，文件缺失或不兼容会报错；不会退回原值模式。`dataset_access.json` 同时记录 raw reader 的表示、模型使用的缩放方式、统计文件哈希、拟合视图和当前视图。

推理侧直接使用加载同一份统计的 `UMINormalizer`。先归一化 state，保持 FM 的全部迭代都在归一化动作空间，仅对最终 `normalized_actions` 调用 `inverse_action()`。当前没有修改通用部署服务或训练器，调用方仍须使用这两个明确接口。

## 数据身份与正式使用

拟合视图和当前读取视图可以不同。五个训练阶段和验证集应应用同一训练池统计，不能在每个阶段或验证集重新拟合。模块核对字段／表示顺序、horizon、父数据版本以及数值统计身份，不要求两个 view fingerprint 必须相等。

统计文件区分 `engineering` 与 `formal`。正式使用需要显式实验合同，绑定获准训练池的 fit view、父数据／表示版本和允许应用的视图。正式配置拒绝工程统计及不匹配合同；本轮不会创建已批准的正式合同。代码校验合同一致性，不能代替你对任务、场景与最终划分的科学判断。

候选数据用于工程验证，不会自动升级为正式数据。最终缩放策略、正式训练清单及正式统计量仍待确定。通过连接测试只说明输入、统计、变换和采样接口相容，不说明归一化策略最优或模型已经学会任务。

## 本轮工程验收

现有 PPU 测试环境中完成 40 项归一化专项测试，包括加权统计与逐窗口参考一致、跨 row-group／文件、文件中断恢复、锁与结果损坏检查、正式／工程用途拒绝、NumPy/Torch 正反变换和 factory 缓存隔离。没有升级环境。

真实工程视图只选两条 episode，共 3,010 个有效窗口；拟合统计的 state count=3,010、action count=48,160，统计用途为 `engineering`。未生成全量或正式统计。

一次真实 Qwen3-VL-2B QwenPI 前向／反向和四步动作采样通过，VLM、动作头与 state encoder 均有有限非零梯度。Hook 核对归一化标签在 FM 加噪之前进入动作头；原值正反变换以 `rtol=1e-5, atol=1e-6` 验证，原始再次读取不变。最终动作反变换后为 FP32 `[1,16,16]`，峰值张量显存约 28.15 GiB。

该次测试 optimizer steps=0，动作头尚未训练。初始 loss 约 9258.74、预测误差较大，只用于接口检查；不作为学习效果或数值训练稳定性已通过的证据。完整报告和数组保存在私人 runs 的 `qwenpi_normalization_engineering_20260919T171709_905689Z/`，未提交原始样本或工程统计参数到仓库。

复现模型连接检查（统计已存在后）：

```bash
source /usr/local/PPU_SDK/envsetup.sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  "$PY" -u examples/umi_pretrain/tools/check_qwenpi_normalization.py \
  --index-dir "$ENGINEERING/access" \
  --statistics "$ENGINEERING/statistics.json" --samples 1 --device cuda:0
```

正式训练入口、优化器／随机数／采样消费进度的完整 checkpoint 和阶段续训仍未接入。
