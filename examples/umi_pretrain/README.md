# Roban UMI 世界位姿 16 维调试训练

本适配供 QwenPI 使用；共享加载器目录名 `gr00t_lerobot` 不决定模型。当前数据契约为 `world-pose-16D`，用于私人小样本的数据核对和短程训练检查。

## 当前契约

- 四路当前帧图像固定顺序：head_left、head_right、wrist_left、wrist_right。
- 每端读取 `finger_eef_pose` 的 7 维 `[x, y, z, qx, qy, qz, qw]`，再拼接 `sensor_magnetic_encoder` 的 1 维；先 robot1，后 robot2，共 16 维。当前契约不使用 `finger_relative_eef_pose`。
- state 为当前帧 `[1, 16]`；action 为同一 episode 内未来第 1 到第 16 帧 `[16, 16]`，不含当前帧。
- state/action 都是测量轨迹的原始 FP32 数值；action 是未来测量目标，并非原始机器人控制命令。
- `action_mode: abs`；不做差分、归一化、参考系转换、四元数改号或重新归一化，也不裁剪磁编码器数值。
- 世界位姿版本直接选择 `observation.umi.robot1_finger_eef_pose` 与 `observation.umi.robot2_finger_eef_pose` 源列，保留其原始参考系表示；本次适配不新增 world 到机器人坐标系的变换。

仓库 `train_files/modality.json` 与实际私人数据的 `meta/modality.json` 必须一致。审计函数严格验证字段、切片、顺序、absolute 和 dtype。仅修改仓库模板不会改变已提取数据的映射。

## 完整窗口过滤

`tools/umi_valid_windows.py` 在创建训练 DataLoader 或加载模型之前扫描全部原始行，仅保留“当前帧 + 未来 16 帧”全部有效的锚点：

- frame_validity 有效，16 维信号存在且有限，四元数非零，任务文本可用。
- 已有 schema_presence 标记不能表明所需字段缺失；已有源时间戳不能是无效值或配置的哨兵值。
- 17 帧属于同一 episode 和同一任务，frame_index 连续，时间戳递增且与 fps 相符。默认相邻间隔容差为 `0.20 / fps + 1e-6` 秒，报告记录具体规则。
- 末尾不足 16 帧、窗口内存在无效帧或时间中断时，整个锚点被排除，不使用末帧重复补齐。

过滤仅建立 `torch.utils.data.Subset` 的有效索引，不删除或重排 parquet 行，不重写视频，不调整视频时间偏移。`dataset.all_steps` 的第二项是 episode 内行偏移，报告另外记录原始 frame_index，避免缺帧时混用两者。四元数范数、相邻旋转变化和磁编码器分布用于诊断，不自动修正或裁剪数值。

## 文件与路径

当前仓库：

`/mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain`

当前精选源数据：

`/mnt/nas/public/roban_umi/restricted_data/derived/umi_v30_curated_v1`

默认私人数据、模型与输出：

- `../datasets/roban_umi_debug`：现有 episode 0 小样本，原始 755 行；有效窗口数以本次报告为准。
- `../models/Qwen3-VL-2B-Instruct`：既有本地模型。
- `../runs/qwenpi_world-pose-16D_<UTC时间>`：每次新建的独立输出目录。

脚本默认私人路径由仓库位置计算，不依赖旧的 `/mnt/workspace/lpy` 挂载。训练输出必须位于私人 runs 目录下，已有目录不会覆盖。

- `tools/prepare_debug_dataset.py`：提取完整 episode 0，输出独立数据与 metadata，视频使用符号链接并保留原偏移。输出目录必须尚不存在；现有小样本不需要再次提取。源数据根目录可通过 `ROBAN_SOURCE_ROOT` 指定。
- `tools/umi_valid_windows.py`：共享映射校验、逐帧审计、完整窗口过滤及索引映射。
- `tools/check_debug_dataset.py`：CPU 数据检查，实际解码四视角，对照原始 FP32 行核对 state/action、语言和视频偏移，生成报告与预览。
- `tools/train_qwenpi_debug.py`：默认 20 步的训练检查；`--data-only` 走同一过滤、Subset 与实际加载器预检，不导入模型或运行训练。
- `tools/requirements-data-check.txt`：历史数据检查环境依赖说明，不是训练环境配置。

## 数据检查命令

直接使用已经存在的私人 Python 环境，不需要激活或安装环境：

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
export PYTHONDONTWRITEBYTECODE=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python \
  examples/umi_pretrain/tools/check_debug_dataset.py

CUDA_VISIBLE_DEVICES="" \
/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python \
  examples/umi_pretrain/tools/train_qwenpi_debug.py --data-only
```

独立检查器在私人数据目录保存 `debug_check.world-pose-16D.json`（含完整窗口报告）和 `debug_preview.world-pose-16D.jpg`。训练入口的数据预检另建私人 runs 目录，保存请求配置、精确映射、允许索引及其 SHA-256、窗口报告和实际加载器样本核对结果。

全部原始行参与过滤审计；实际图像解码和逐项 FP32 对照采用代表性有效锚点。预检要求加载器输出已经是 NumPy FP32，不能通过事后转换掩盖之前的 FP16 舍入。

## 手动启动 20 步训练

下面为明确指定设备的启动命令；命令本身不代表训练已经通过：

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
# 这台 PPU 测试机需要载入已有 SDK；只影响当前 shell。
source /usr/local/PPU_SDK/envsetup.sh
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python \
  examples/umi_pretrain/tools/train_qwenpi_debug.py \
  --device cuda:0 --max-steps 20 --batch-size 1 --num-workers 0 --save-every 20
```

保留既有 `QwenPITrainCheck` 精度桥接：参数为 FP32，VLM 保留框架内部 BF16 autocast，输出隐藏状态转 FP32 后进入动作头，转换保留计算图。AdamW 设置不变：VLM 学习率 `1e-5`，动作头 `1e-4`，betas `(0.9, 0.95)`，eps `1e-8`，无 weight decay，梯度裁剪为 1.0。动作头使用下文所述 canonical self/cross 路径；不额外更换解码器或激活函数。

每步检查有限标量 loss、VLM、动作头与 state_encoder 各自有限且非零的梯度范数，以及每组一个有梯度参数张量中最多 32 个选定元素的更新。state_encoder 属于原动作头优化器组，不会被重复加入优化器。差值仅证明抽查元素发生更新，不证明全部参数更新，也不证明模型对 state 存在因果依赖或已具备任务成功能力。

每次运行保存：

- `config_requested.yaml` 与模型加载后的 `config.yaml`。
- `run_metadata.json`：契约版本、精确映射、有效索引及指纹、窗口报告、模型位置、可获得的 revision 和本地配置哈希。
- `valid_windows.json`：完整窗口审计。
- `progress.json`：步数、loss、梯度、有限范围的参数更新证据、运行状态及最近 checkpoint 步数。
- `pytorch_model.pt`：沿用 `model.state_dict()` 格式，先写临时文件再替换；不含完整断点续训状态。

## 数据配置与共享加载器

以下为与当前入口一致的数据相关配置片段，相对路径以仓库根目录为基准：

```yaml
framework:
  name: QwenPI
  qwenvl:
    base_vlm: ../models/Qwen3-VL-2B-Instruct
    attn_implementation: sdpa
  action_model:
    action_dim: 16
    state_dim: 16
    action_horizon: 16
    diffusion_model_cfg:
      interleave_self_attention: true
      use_canonical_forward: true

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: ../datasets
    data_mix: roban_umi_debug
    lerobot_version: v3.0
    include_state: true
    action_mode: abs
    lowdim_dtype: float32
    strict_window_sampling: true
    video_backend: torchvision_av
```

`lowdim_dtype: float32` 是共享加载器的显式选择项，使原始低维数据直接转换为 FP32；其他未设置该项的数据配置保留既有默认行为。单独使用此 YAML 不会自动过滤窗口，过滤由本版检查器和调试训练入口调用。

`strict_window_sampling: true` 使低维读取仅堆叠当前窗口的行，并拒绝越界补齐，避免窗口外的畸形信号影响有效窗口；默认关闭，不改变其他数据集的补齐行为。

两个 debug 入口还自动传入 `raw_lowdim_statistics`：仅对有效窗口涉及的原始行并集计算小样本元信息统计，在内存中供加载器初始化使用。它不用于归一化，不写入 `stats_gr00t.json`，也不作为全量训练统计；这样被排除的畸形行不会在初始化时阻断有效样本读取。未传入此项的其他数据集沿用原统计路径。

共享加载器保留既有任务表索引兼容、各相机独立视频 chunk/file 路径和标量磁编码器支持。数据变换只转换 Tensor；私人小样本统计不能作为全量训练归一化统计。

## 历史结果适用范围

2026-09-18 文档中的 9 项回归测试、第 0/100/754 帧检查与末帧补齐结果属于旧 relative-pose、8 帧目标检查，不能当作本版 world-pose、16 帧完整窗口、FP32 契约的通过证据。旧检查报告按 `debug_check.legacy-relative_<时间>.json` 命名归档。

当前结果应以新报告中的 data_version、映射、形状、dtype、过滤数量和实际运行状态为准。

## 数据适配验收（动作头修复前，2026-09-19）

- 34 项 CPU 回归测试通过，包括没有统计缓存、窗口外存在畸形 pose 的真实数据集初始化及有效样本读取；公共默认路径的既有测试仍通过。
- 独立检查器通过：原始 755 帧保留，739 个完整有效窗口，末尾 16 个不完整锚点排除；抽查第 0、369、738 帧的 state/action 与源列 FP32 值逐项相等。
- 实际 batch：state `[2, 1, 16]`，action `[2, 16, 16]`；四视图路径、顺序、时间偏移及任务文本核对通过。
- 最终训练入口 `--data-only` 通过，报告位于 `/mnt/workspace/Native_Policy/user/wyt/runs/qwenpi_world-pose-16D_20260919T060139_103847Z`；未加载模型。
- 20 步训练尝试在第 0 步因正在单独修改的动作头引用未定义 `timesteps_tensor` 而停止，未生成 checkpoint。记录位于 `../runs/qwenpi_world-pose-16D_20260919T055709_526834Z/progress.json`。用户确认先完成数据适配；本次没有修改动作头，也没有继续启动训练。

以上属于动作头修复前的数据验收记录。此次动作头代码补齐后，尚未重新运行模型测试。

## 动作头修复与待运行检查

按上游 [c521dec](https://github.com/starVLA/starVLA/commit/c521decb7441c7dfea282c61dc758456bcffbb8f) 定向补齐两个模型文件，未引入 RTC 等其他功能：

- `LayerwiseFM_ActionHeader.py` 的训练与普通推理都调用 `DiT.forward(timestep=...)`，保留 VLM mask 与 `return_pre_output=True`，继续使用既有 action_decoder。
- `cross_attention_dit.py` 兼容单个特征 Tensor 和完整逐层列表；列表按原层号取值。新训练显式启用 `interleave_self_attention=True` 与 `use_canonical_forward=True`：零起始偶数层 cross，奇数层 self；self 层不接收 VLM 特征或 VLM padding mask。
- `use_canonical_forward=False` 保留上游 legacy all-cross 行为，供旧路径复现；本 UMI 新训练不选择该模式。
- `train_qwenpi_debug.py` 和 `check_qwenpi_state.py` 共用 `make_debug_config`。新 run 加载基础 Qwen 权重、重新初始化动作头，不加载旧动作 checkpoint。

本次只修改并阅读代码，未执行以下诊断或训练。由使用者在测试机依次运行：

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
source /usr/local/PPU_SDK/envsetup.sh
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export USE_TF=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/mnt/workspace/Native_Policy/user/wyt/.cache/huggingface
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PY_BIN=/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python

"$PY_BIN" -B -u examples/umi_pretrain/tools/check_qwenpi_state.py --device cuda:0 && "$PY_BIN" -B -u examples/umi_pretrain/tools/train_qwenpi_debug.py   --device cuda:0 --max-steps 20 --save-every 20 --batch-size 1 --num-workers 0
```

第一项检查实际训练/推理的各层 attention 路径，固定随机状态比较 state 平移扰动前后的速度及完整动作预测，并检查 state_encoder 梯度；不更新参数。第一项失败时不会启动后面的短训练。诊断和训练分别创建私人 runs 子目录，输出位置由脚本打印。

这两项通过只说明运行路径和短程更新可用，不代表原生动作表示的最终效果，也不作为旧 checkpoint 已恢复正常的证据。全量预训练适配不在本次修改范围内。
