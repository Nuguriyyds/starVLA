# 动作解码器退化诊断与单变量 LayerNorm 试验

后续对照见 [CONDITION_NORM_STATUS.md](CONDITION_NORM_STATUS.md)：条件 K/V 输入归一化已测，入口尺度恢复，但 50 步仍未学会有效去噪，按约定暂停。

2026-09-20。基于 `fa8ee31`；最小实现和本次归一化试跑代码为 `b45ea930a32d82c3d5a29a76b4f059c9dae742e4`。

**旧终态的 bias-only 退化已得到直接证据。LayerNorm 候选完成 200 次更新，避免了所测全零激活，但小数据可学习性仍未通过。**

## 旧 checkpoint：直接观察，不更新参数

仅加载 200 步模型权重，不加载 Adam。固定窗口 0/16/31，分别进行 eval/train 模式的真实动作前向、反向和完整四步生成；所有统计只覆盖末尾 16 个 action token。梯度在裁剪前测量。

- 三个 eval 探针中 ReLU 输出全零，模型 FM loss 与同实际噪声的 bias-only loss 相同。
- 三个完整生成中，每一步速度严格等于末层 bias；直接捕获初始噪声，终点与 noise+bias 的最大误差约 4.71e-7。
- 固定噪声和时间，将第一个归一化 state 分量加 0.5，速度变化为零。
- eval 探针中，除末层 bias 外，所查解码器/state/DiT/VLM 梯度为零。
- train 模式窗口 16 有 1/32768 个正激活，并出现微弱上游梯度。因此不能说所有训练输入永久失活。

## 50 步短重现

总 scheduler 仍定义为 200 更新，原 seed、数据顺序、学习率、batch 和梯度累积保持一致；第 50 步提前停止。只在工具内省略此次非续训诊断的 checkpoint，不修改生产保存逻辑。

| 更新 | action 位置 h RMS | ReLU 非零比例（eval） | h 梯度 RMS |
|---:|---:|---:|---:|
| 0 | 319.462 | 49.6216% | 0.0064 |
| 1 | 316.297 | 48.9197% | 0.00406 |
| 2 | 327.227 | 48.3398% | 0.00388 |
| 5 | 321.795 | 43.9178% | 0.00193 |
| 10 | 320.359 | 32.0312% | 0.00454 |
| 20 | 29.411 | 22.9980% | 0.000292 |
| 50 | 100.354 | 0.1953% | 9.75e-06 |

初始 32 窗口评测与原始运行相同，但后续数值并非逐位重现。首步 loss 相同，梯度范数已有差异；不把具体退化时刻写成可逐位复制的结果。所查 50 次的数据序列、学习率均一致。尺度并非单调增长，现有证据支持入口尺度偏大与激活关闭这一嫌疑链，不能宣称已完整证明因果链。

## 唯一模型改动

`LayerwiseFM_ActionHeader.py` 增加可选的 `decoder_input_norm: layer_norm`：

```text
DiT 原始残差输出 → 非仿射 LayerNorm(hidden_dim, eps=1e-5) → 原 Linear → ReLU → Linear
```

默认 `none`，保留旧行为。新层没有可学习参数，不消耗随机数；公共 MLP、state encoder、DiT 内部计算、初始化、学习率和四步采样均未改。训练和推理使用同一入口。三项单元测试通过：原参数与 RNG 初始化一致；训练及四次采样均走归一化且梯度有效；未知配置拒绝。实际大模型的初始参数探针也一致。

**使用候选权重时必须携带模型配置中的这个开关。** 它不增加 state_dict 键，单看权重文件无法推断是否开启；训练身份与保存的 plan 已记录该设置。本轮只在候选新 run 中开启，没有把正式配置默认切换为该方案。

## 同 32 窗口、200 更新结果

| 更新 | 固定 FM loss | normalized 生成 MSE | 位置 RMSE | 夹爪 RMSE | 旋转角误差 ° |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.788922 | 1.785282 | 0.162177 | 0.017001 | 37.849 |
| 50 | 1.623536 | 1.633574 | 0.130213 | 0.016889 | 37.317 |
| 100 | 1.612373 | 1.627400 | 0.130437 | 0.016829 | 37.109 |
| 150 | 1.606786 | 1.620013 | 0.129702 | 0.016808 | 37.105 |
| 200 | 1.592428 | 1.605294 | 0.129181 | 0.016697 | 36.745 |

保持状态基线：位置 0.019161、夹爪 0.008730、旋转 6.315°。RMSE 使用原标签数值单位。

终态窗口 0 的 eval 探针：raw DiT RMS=1638.081，归一化后 h RMS=1.000000，ReLU 非零比例=4.3945%，速度减 bias RMS=0.326134。
同实际噪声的 FM loss：模型 1.211214，bias-only 1.372116。
固定噪声/时间的 state 干预，速度最大变化 2.3841858e-07。

可学习性检查：`{"fixed_fm_loss_decreased": true, "generated_action_mse_decreased": true, "position_beats_hold": false, "gripper_beats_hold": false, "rotation_beats_hold": false}`。

评测与训练重叠，不能证明泛化或策略成功；非零激活、非零梯度和低于 bias-only 的 loss 也不能替代保持状态基线。只完成这一项机制试验，不继续叠加激活替换、逐层归一化或更大学习率。视频、数据表示、归一化统计、正式划分和分布式环境未改。

## 当前结论与下一处定位

不要把“速度减 bias 不为零”直接写成“已经学会读取条件”。终态 state 单坐标干预只产生约 2.38e-7 的速度差，响应很弱；state_encoder.layer1.weight 的动作梯度 RMS 从初始约 4.60e-3 降到约 1.19e-9。该探针不代表所有 state 分量，但足以提醒：非零梯度不是有效学习的充分证据。

现有逐层统计已经给出下一处定位依据，无须再训练来得到这些数字：

| eval 探针位置（层从 0 计） | 初始化 RMS | 更新 200 后 RMS |
|---|---:|---:|
| DiT block 1 输出 | 0.533 | 1.681 |
| DiT block 2 输出（跨注意力） | 97.579 | 134.573 |
| DiT block 27 原始输出 | 319.462 | 1638.081 |
| 新增的解码器输入 LayerNorm 后 | 1.000 | 1.000 |

后续应优先检查 VLM 条件进入跨注意力时的幅值，以及带噪动作/state 对各层输出的敏感度。**“条件注入掩盖动作信号”目前是待检验假设，不是已证实根因。** 本轮没有再加第二种归一化或改激活。候选开关保持可选，不把它宣称为已验收的正式训练修复。

本次全部 200 次 loss/全局梯度范数有限，进程退出码 0；峰值 allocated 显存 62.18 GiB，末尾 basic 保存 88.27 秒。诊断回调有额外成本，本轮不用于重新评估吞吐。

## 复现入口与证据

在仓库根目录、已有私人 PPU 环境运行，设置 `CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 PYTHONPATH="$PWD"`：

```bash
python examples/umi_pretrain/tools/diagnose_qwenpi_action_decoder.py --run-dir /mnt/workspace/Native_Policy/user/wyt/runs/qwenpi_learnability_v1 --output 新的诊断JSON路径
python examples/umi_pretrain/tools/check_qwenpi_decoder_trial.py --reference-run /mnt/workspace/Native_Policy/user/wyt/runs/qwenpi_learnability_v1 --output-dir 新的私人运行目录 --variant reproduce
python examples/umi_pretrain/tools/check_qwenpi_decoder_trial.py --reference-run /mnt/workspace/Native_Policy/user/wyt/runs/qwenpi_learnability_v1 --output-dir 另一个新的私人运行目录 --variant layer_norm
```

汇总和激活/梯度统计在 `decoder_validation/`。`control_comparison.json` 核对实际样本顺序、学习率和初始参数探针；`training_summary.json` 记录资源与保存信息。
原始预测 NPZ、checkpoint 和运行日志保留在私人 `runs/qwenpi_decoder_layernorm_v1`，不提交原始图像或动作标签。
