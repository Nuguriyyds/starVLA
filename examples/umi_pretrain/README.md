# Roban UMI 世界位姿 16 维调试训练

正式全量训练准备、固定清单及四节点启动见 [formal/README.md](formal/README.md)。下面保留早期 debug 接口说明；正式入口使用预编译索引，不在启动时重扫全部低维数据。

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

## 第一阶段：全量元数据目录

`tools/build_umi_catalog.py` 只读取 `meta/info.json`、任务表、
episode 元数据及已有的 merge/curation 来源映射，不读取逐帧数值、
解码视频、加载模型、过滤窗口或计算归一化。已有 debug 入口保持不变。

使用已有私人 Python 环境。输出目录必须尚不存在；工具拒绝覆盖已有目录
或写入公共源数据。命令行输出只允许放在私人 `../data_preparation/` 下。

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
PY_BIN=/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python
CATALOG_ROOT=/mnt/workspace/Native_Policy/user/wyt/data_preparation
CATALOG_RUN=$(date -u +%Y%m%dT%H%M%SZ)

# 少量 episode 元数据检查；任务和已有来源映射表仍完整读取。
"$PY_BIN" -B -u examples/umi_pretrain/tools/build_umi_catalog.py \
  --output "$CATALOG_ROOT/roban_umi_world_pose_v1_probe_$CATALOG_RUN" \
  --max-meta-files 3 --check-files

# 全量元数据盘点；--check-files 只检查引用文件是否存在。
"$PY_BIN" -B -u examples/umi_pretrain/tools/build_umi_catalog.py \
  --output "$CATALOG_ROOT/roban_umi_world_pose_v1_$CATALOG_RUN" \
  --check-files

# 合成数据的 CPU 测试，不读取生产数据的逐帧内容或加载模型。
"$PY_BIN" -B -m unittest discover -s tests -p test_umi_catalog.py -v
```

默认源目录为本文记录的精选数据目录，可通过 `--source` 覆盖。
`--batch-size` 与 `--part-rows` 控制元数据读取和输出分片大小。
关联与汇总使用输出目录内的 SQLite，不在内存中展开逐帧索引。

主要产物：

- `catalog_config.json`：源信息、选项、代码 commit、工作区状态及工具哈希。
- `episode_manifest/part-*.parquet`：episode 身份、全局范围、数据文件、
  四相机独立偏移、任务关联、原始来源映射及逐条异常。
- `task_catalog.parquet`、`task_distribution.parquet`：任务原文与 episode 级分布。
- `schema_report.json`：原始 schema、输入元数据哈希及缺失字段。
- `catalog_report.json`：总量、分布、来源覆盖及异常计数。
- `source_groups.parquet`、`file_inventory.parquet`、`catalog_index.sqlite3`：
  来源路径分组、引用文件清单及用于审计关联的磁盘索引。

`dataset_from_index`/`dataset_to_index` 是全局索引；
curation 的 `local_dataset_*` 是 set 内累计索引，不保证是文件内行号。
因此清单保留 `data_locator=episode_index_filter`，文件内行范围留空。
后续索引阶段确定真实文件内偏移后，才能直接按行切片。

`source_mcap` 按已有说明映射为规范 OSS 路径，但不同路径不证明内容
或 session 不同。场景、session、采集者只保留明确提供的字段，不从任务文本、
set ID 或路径层级猜测。多任务 episode 会计入每个关联任务，其帧数不是
互斥的逐任务标注时长。

有限扫描标记 `scan_complete=false`。完整扫描检查总量、全局范围连续性、
首尾位置及映射一致性。元数据异常保留记录，标记 `completed_with_issues`
并以退出码 2 结束；失败产物保留在独立目录。文件存在不代表内容有效。
这个目录供下一阶段建立有效窗口索引使用，尚不是可直接训练的数据集。


### 第一阶段验收（2026-09-19）

- 14 项元数据工具 CPU 测试通过。
- 全量扫描 366 个 episode 元数据文件，输出 40 份清单：327,667 条 episode、1,023,415,387 帧、99,027 个任务。
- 按 30 fps 计算的名义时长为 9,476.07 小时，未进行有效窗口过滤。
- 引用的 49,236 个数值/视频文件均存在；元数据一致性检查未发现异常或警告。没有读取这些文件的逐帧内容或解码视频。
- 每条 episode 均关联 source_mcap；源路径全部唯一。没有明确的 scene/session/collector 字段，不能据此排除不同源路径下的同次采集或重复内容泄漏。
- 结果目录：`/mnt/workspace/Native_Policy/user/wyt/data_preparation/roban_umi_world_pose_v1`。源 commit 为 `7866233`，实际扫描工具哈希与元数据哈希已保存在报告中。


## 第二阶段：全量低维有效窗口索引

`tools/build_umi_window_index.py` 读取第一阶段的完整 catalog，直接扫描全部
3,286 个物理低维 Parquet。`tools/umi_window_rules.py` 是已有过滤规则的
NumPy/Arrow 实现；未修改小样本检查器、模型或正式训练入口。
本轮按用户要求不运行小样本或合成测试，完成静态审查后直接启动全量扫描。

每个物理文件的必要列在内存中处理，实际最大 341,712 行；默认单文件
100 万行资源保护，超限记为失败，不截断。默认 2 个 CPU worker，不用训练卡。
四路视频、IMU、触觉以及不参与已有规则的 gap_ns 列不读取。

规则保持 world-pose-16D、robot1 后接 robot2、原始 FP32、未来 1..16 帧。
当前行及未来 16 行都必须有效；任务、帧序号及时间连续性按已有规则检查。
不改四元数、不裁剪开度、不补齐、不删除无效行后拼接。根时间戳先按整数
纳秒求差，传感器自己的时间戳不要求 30Hz，也不新增 gap_ns 阈值。

工具核验每文件的 episode 集合、每 episode 实测长度和全局 index 序列。
不将 frame_index 当作原始行位置；多个非连续物理片段保存在 segment 表。
若真实文件分配与 catalog 不符，该文件失败并报告，不猜测跨文件映射。
读取批次和 row group 边界不会截断窗口。

首次启动（输出目录必须尚不存在）：

```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python -B -u \
  examples/umi_pretrain/tools/build_umi_window_index.py \
  --catalog /mnt/workspace/Native_Policy/user/wyt/data_preparation/roban_umi_world_pose_v1 \
  --output /mnt/workspace/Native_Policy/user/wyt/data_preparation/roban_umi_world_pose_windows_v1 \
  --workers 2
```

已有后台扫描不要重复启动。运行进度位于输出目录的
`scan_progress/progress.json`，其中有父进程 PID、完成文件数、处理行数、
有效窗口数和吞吐。需要平稳停止时向这个父进程发送
`kill -TERM <父PID>`，停止提交新文件并等待当前文件完成。
前台 Ctrl+C 可能同时打断 worker，尚未提交的文件在恢复时重做。

恢复时使用上面同一命令并增加 `--resume`。规则、代码、catalog、
字段映射和 horizon 指纹必须一致；worker 数可以改变。
已完成文件会复核源文件 size/mtime 和输出 SHA256。源数据 size/mtime
不是内容哈希；源数据在整个准备过程中应保持不变。
不要在扫描期间修改两个新工具或旧规则模块，再尝试混合恢复。

主要产物：

- `window_index_config.json`：固定输入、规则、字段映射和代码哈希。
- `episode_segments/`：episode 原始行位置到实际文件行段的对应关系。
- `valid_anchor_ranges/`：有效起点的半开区间，坐标是 episode 原始行位置。
- `episode_quality_parts/`：逐文件落盘的 episode 质量结果。
- `episode_quality.parquet`：全量成功后合并的质量表。
- `task_window_distribution.parquet`：全量成功后的任务行数及有效窗口数量。
- `scan_progress/completed/`：每物理文件的原子完成记录，包含结果哈希。
- `scan_progress/failed/`：失败诊断；恢复成功后以 completed 标记为准。
- `window_index_report.json`：完成或平稳停止时的全局报告。

完整候选窗口数 = 有效窗口数 + 因质量/连续性拒绝的候选数。
末尾不够 16 帧的起点单列，不混入完整候选；多标签原因可能重叠，
主要拒绝原因互斥。时长分别统计原始行、有效行、有效窗口覆盖的去重行，
不能用窗口数乘 16 计算独立数据时长。

此结果只说明低维条件和目标满足当前规则，未验证全量图像内容或时间对齐，
也尚未划分训练/验证集、计算归一化或改变采样权重。


## 第二阶段补充：训练范围内的质量审计

全量索引已完成：3,286 个文件成功，1,013,325,887 个有效窗口，504,659 个
起点区间，覆盖 1,021,400,431 个去重原始行，约 9,457.41 名义小时。

新增 `tools/audit_umi_trainable_quality.py` 和 `tools/umi_quality_metrics.py`。
旧位置/开度/范数诊断包含可解析原始行；旧相邻变化包含有效连续边，
不限定完整窗口。旧范数 std 的 sum-squares 差分存在消减误差，旧报告
保留作为来源记录；新审计使用 FP64 count/mean/M2（Chan）合并。

新审计按冻结的 segment/range 分开统计 raw、row_valid、window_covered，
并单独构造窗口内部边的覆盖范围。先计算源存储四元数范数，仅在旋转诊断
中单位化并取 abs(dot)。分位数只给固定直方图的区间，阈值桶不用于清洗。
它重读必要低维列，但不重新生成窗口、不修改原始数值、模型或旧指纹。

已有后台任务不要重复启动。默认输出：
`/mnt/workspace/Native_Policy/user/wyt/data_preparation/roban_umi_trainable_quality_v1`。
入口命令为：
```bash
cd /mnt/workspace/Native_Policy/user/wyt/starVLA-umi-pretrain
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/mnt/workspace/Native_Policy/user/wyt/.venvs/qwenpi-ppu-py312/bin/python -B -u \
  examples/umi_pretrain/tools/audit_umi_trainable_quality.py --workers 2
```
恢复增加 `--resume`，规则/工具版本须不变。平稳停止向 progress.json
记录的父 PID 发送 SIGTERM。

产物包括 `trainable_quality_report.json`、`quality_cases.parquet`、
`quality_review.md`，分别保存统计、可定位上下文和待解释问题。
已知极值文件优先处理，初步报告 scan_complete=false，不能当全量频率。
来源分布用 source_set_id，不能冒充设备/session/场景分类。
仅做静态审查后直接运行实际全量审计，没有重跑小样本训练测试。


## Indexed full-dataset access

See [indexed access documentation](indexed_access/README.md) for the immutable runtime index, lazy UMI Dataset, block sampler, and configuration overlay.

## 元数据候选划分与待确认的语义标签

见[划分与视图构建说明](splits/README.md)。`build_umi_splits.py` 基于冻结元数据按原始采集来源组生成约 1% 候选验证池和五份候选训练清单，默认输出 `../data_preparation/roban_umi_splits_metadata_candidate_v1/`。当前均衡只使用有效窗口量、task ID 和 source set；task ID 不等于任务类别，source set 不等于场景，尚未检验任务／场景语义均衡，不能据此固定最终划分。

当前没有构建六个分片视图。`build_umi_split_views.py` 仅在显式传入 `--allow-candidate-splits` 时允许候选清单接口检查，默认输出 `../data_preparation/roban_umi_candidate_views_v1/`；这不是当前默认执行步骤，也不意味着可以启动正式训练。

最终划分仍需确定语义标签 schema 和分类口径。使用者负责查看文本／视频、确定类别并抽查纠错；之后程序按确认的标签批量关联、分组分配和核对分布，不需要手工逐条划分 32 万条 episode。语义标签接入与相应平衡约束尚未实现。保留现有质量政策，不做新的数据清洗；归一化与阶段训练接口先用工程视图验收。

## 可逆归一化与离线统计接口

见[归一化工程说明](normalization/README.md)。原始 Dataset 和 `read_lowdim()` 保持原值语义，factory 可用独立包装层加载 `mean_std` 统计；支持 state/action 正变换和最终动作反变换。离线工具按实际有效窗口出现次数加权，FP64 增量统计、文件级恢复，不解码视频。本轮仅在明确的小型工程视图上拟合和验证；正式配置拒绝工程统计，正式训练池与统计量仍待确认。

## 分阶段训练与完整续训

见[训练入口说明](training/README.md)。新增 `starVLA.training.train_umi_pretrain`，将完整计划与本次进程暂停点分开；只在成功 optimizer update 后提交全局窗口游标。使用 Accelerate 保存模型、Adam、scheduler、阶段进度和每 rank 随机状态，完成标记发布后才更新 latest。阶段切换保留优化状态并释放旧 loader。当前支持单设备和普通 DDP；DeepSpeed/FSDP、64 卡通信与正式语义划分另行验收。
