# Roban UMI 数据适配与小样本检查

这套适配供 QwenPI 使用。`gr00t_lerobot` 是共享的数据加载器，并不代表选择了 QwenGR00T 模型。模型选择仍由训练配置中的 `framework.name: QwenPI` 决定。

## 当前数据约定

- 四路图像固定顺序：head_left、head_right、wrist_left、wrist_right。
- 每个操作端保留 finger_relative_eef_pose 的 7 维以及磁编码器的 1 维；先 robot1，后 robot2。
- state 读取当前帧，形状 `[1, 16]`。
- action 读取未来第 1 到第 8 帧，形状 `[8, 16]`；来自示范轨迹测量，而非原始机器人控制命令。
- 保留数据原有的固定参考系和四元数表示；`action_mode: abs` 不进行额外差分。
- 当前 transform 只转 Tensor，尚未做训练归一化。
- 轨迹末尾不足 8 帧时，加载器重复最后一帧。本检查会显式核对这一行为。

## 文件作用

- `train_files/modality.json`：原始列、列内切片与逻辑名称的映射。
- `train_files/data_registry/data_config.py`：自动注册、字段顺序、时间索引。
- `tools/prepare_debug_dataset.py`：从精选版提取完整 episode 0；输出独立 parquet/metadata，四路视频使用符号链接并保留原偏移。
- `tools/check_debug_dataset.py`：仅检查私人小样本，实际解码四路图像，验证语言、state/action 数值及时间索引，生成检查报告和四视角预览。
- `tools/requirements-data-check.txt`：数据检查环境的依赖；不作为模型训练环境配置。

## 数据位置

源数据：
`/mnt/workspace/public/roban_umi/restricted_data/derived/umi_v30_curated_v1`

已生成的私人小样本（1 条轨迹、755 帧）：
`/mnt/workspace/lpy/Native_Policy/user/wyt/datasets/roban_umi_debug`

脚本不会覆写已存在的数据目录。需要重新提取时通过 `--output` 指定另一个私人目录。

## 运行检查

下面只执行数据检查，不加载 VLM，不启动训练。

```bash
cd /mnt/workspace/lpy/Native_Policy/user/wyt/starVLA-umi-pretrain
export PYTHONDONTWRITEBYTECODE=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
ROBAN_CHECK_PY=/mnt/workspace/lpy/Native_Policy/user/wyt/.venvs/roban-data-check-py310/bin/python

"$ROBAN_CHECK_PY" -m unittest discover -s tests -p test_lerobot_v3_metadata.py -v
"$ROBAN_CHECK_PY" examples/umi_pretrain/tools/check_debug_dataset.py
```

检查报告写入小样本目录下的 `debug_check.json`，四视角预览为 `debug_preview.jpg`。小样本统计和步骤缓存也只写入私人小样本目录；不能把这些统计用作全量训练统计。

## 数据加载代码的修改

`starVLA/dataloader/gr00t_lerobot/datasets.py`：

1. `_get_tasks()` 同时支持任务文本存于普通列或 pandas 索引，统一以 `task_index` 建立查询索引。
2. `get_video_path()` 使用各个相机自己的 chunk/file 编号，避免错误地套用传感器数据文件编号。
3. `get_state_or_action()` 支持标量列，将磁编码器这类 `(T,)` 数据作为 `(T, 1)` 单维特征读取；不修改数值或时间索引。

另外，`starVLA/dataloader/__init__.py` 将 VLM 数据加载器改为按需导入：仅在选择 `vlm_datasets` 时导入，避免机器人数据检查提前加载无关的 Transformers 模型依赖。

对应回归测试：`tests/test_lerobot_v3_metadata.py`，包含非连续任务 ID 与相机/数据文件编号不一致的情况。

## 已验证结果（2026-09-18）

- 9 项回归测试全部通过：任务表格式与 ID、相机文件编号、标量和多维信号、时间索引与末尾补齐。
- 实际读取私人轨迹的第 0、100、754 帧，每帧包含四路 224×224 图像、正确任务文本、`state [1, 16]`、`action [8, 16]`。
- state/action 数值与原始 parquet 对应行逐项一致；末尾 action 重复最后一帧。四路视频路径与各自时间偏移均已检查。
- `debug_check.json` 的 `passed` 为 `true`，并已生成 `debug_preview.jpg`。
- 检查环境为私人 `.venvs/roban-data-check-py310`，只读复用现有 Python 3.10 / PyTorch 2.1.2 / NumPy 1.24.4；新增依赖装在私人目录，公共环境和源数据未修改。
- 这是 CPU 数据检查环境，未加载 QwenPI 权重或验证模型前向。torchvision 的可选 CUDA 图像扩展提示不影响本次实际通过的 PyAV 读帧检查。

## 后续训练配置需要的字段

```yaml
framework:
  name: QwenPI
  action_model:
    action_dim: 16
    state_dim: 16
    action_horizon: 8

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: /mnt/workspace/lpy/Native_Policy/user/wyt/datasets
    data_mix: roban_umi_debug
    lerobot_version: v3.0
    include_state: true
    action_mode: abs
    video_backend: torchvision_av
```

这只是数据相关配置片段，不是完整的训练 YAML。全量预训练前还需确定归一化、实际训练环境及训练参数。
