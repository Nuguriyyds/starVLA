# VLM → Action 条件归一化诊断与学习对照

后续有界长试跑见 [EXTENDED_LEARNING_STATUS.md](EXTENDED_LEARNING_STATUS.md)：解析采样检查与当前双 LN 配置的 2000 更新结果，取消 50 步软门槛。

2026-09-20，基线 `d0ce5bc`。本次已完成更新数：50。全部最终可学习性标准：`False`。

**条件输入归一化消除了初始化时 block 2 的巨大注入，但当前配方的学习退化仍未解决；按预定筛查在 50 步正常暂停，没有追加到 200 步。** 噪声变化的预测方向已有正相关，但沿理想方向的增益只有约 0.13%，state 变化仍很弱。不能把上游已报告的归一化缺失，等同于本地所有现象已经找到唯一根因；也不能由这个最小候选失败推断完整 QwenPI_v3 不可用。

## 上游依据与本轮唯一变量

[Issue #356](https://github.com/starVLA/starVLA/issues/356) 及[维护者回复](https://github.com/starVLA/starVLA/issues/356#issuecomment-4599479900)明确指出旧 QwenPI 缺少 VLM embedding normalization。现有 v3 的 `project_layers` 只在维度不同时使用 LayerNorm+Linear，维度相同则 Identity；它还把 state 离散化进文本。因此本轮没有整体换 v3。

参照已有接口位置，仅在 `cross_attention_dit.py` 添加 `cross_condition_norm: none / layer_norm`，未知值直接报错。候选为 token 内 hidden 维非仿射 LN（eps=1e-5），同一结果进入 K/V，不 detach；self-attention 不归一化外部条件；训练和采样共用 block。默认 none，参数和 RNG 初始化不变。它不是 v3 的整套实现，也不是 QK normalization。

对照背景保持 `decoder_input_norm: layer_norm`，连续 state、头部维度、动作表示、数据、统计量、优化器、200 步 scheduler、四步采样均不变。候选开关写入 plan 和 checkpoint 身份；它没有权重键，不能仅凭 state_dict 判断是否启用。生产工程默认配置未自动切换。

## 无更新对照

同一初始化与上一轮 decoder-LN 200 步终态，窗口 0/16/31，时间 0/.25/.75。每窗口缓存一次 VLM 特征；直接在真实采样路径的 action encoder 输入替换带噪动作，并同时指定 action encoder 与 DiT 的离散时间；取第一次真实 decoder 速度后停止，不复制生成网络公式。三个策略：正常、外部条件 LN、实际 cross 残差分支置零。后者不改变 self-attention，也不以空条件或零条件冒充关闭分支。

有效 token 上记录条件和 K/V 统计；block 1/2/3/27 的统计与扰动差异只覆盖末尾 16 个 action 位置。Attention residual 是实际相加的分支，FFN 另记。state 坐标 0/8/15 各加 0.5（归一化单位）；噪声使用两份固定张量。重复前向差异作为数值参照，诊断恢复 RNG/mode，不更新参数。

窗口 0、t=0 的代表性结果（完整三窗口、三时间在 JSON）：

| 权重 | 条件策略 | block2 输入 RMS | Attention 注入 RMS | FFN 注入 RMS | 最后层 RMS | 噪声响应 RMS 比 | state0 速度变化 RMS |
|---|---|---:|---:|---:|---:|---:|---:|
| initial | normal | 0.5355 | 91.71 | 0.2004 | 316 | 0.08117 | 0.01856 |
| initial | layer_norm | 0.4971 | 0.2686 | 0.1949 | 1.816 | 0.02108 | 0.0003133 |
| initial | zero_cross | 0.3328 | 0 | 0.2039 | 1.473 | 0.06805 | 0.001658 |
| decoder_ln_200 | normal | 1.648 | 134.1 | 0.889 | 1638 | 4.43e-06 | 3.849e-08 |
| decoder_ln_200 | layer_norm | 1.698 | 0.3093 | 0.2238 | 20.52 | 0.0003612 | 1.057e-06 |
| decoder_ln_200 | zero_cross | 0.7862 | 0 | 0.214 | 17.5 | 0.000763 | 7.154e-06 |

噪声响应比是 RMS(Δ速度)/RMS(Δ噪声)，并不表示方向正确。t=0 的理想差分为 −Δ噪声，故同时记录 cosine、沿理想方向增益及相对误差。随机初始化不应满足该参照；在旧权重上临时改条件尺度也不能替代重新训练。

## 同 32 窗口的学习对照

从新初始化训练；初始化参数探针、实际样本序列、每步学习率与上一轮一致。PPU 跨运行不宣称逐位相同。50 步筛查在试跑前写入身份，兼看噪声反向响应、state 响应高于重复差异、固定 loss 优于 bias 和生成误差改善。若失败由现有训练器正常暂停并保存，不用非零梯度代替可学习性。50 步筛查是是否继续的工程决策，不是最终验收。

| 更新 | 固定 FM loss | 生成动作 normalized MSE | 位置 RMSE | 夹爪 RMSE | 旋转误差 ° |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.771808 | 1.774981 | 0.156388 | 0.017252 | 37.659 |
| 50 | 1.622396 | 1.630932 | 0.131158 | 0.016852 | 37.327 |

上一轮仅 decoder LN 的同样 50 步：固定 FM loss 1.623536、位置 0.130213、夹爪 0.016889、旋转 37.317°。与本次差别很小，不宣称统计显著的提升或下降。
保持当前状态基线：位置 0.019161、夹爪 0.008730、旋转 6.315°。RMSE 为原标签单位。

| 更新 | 窗口 | 噪声反向 cosine | 沿理想方向增益 | 相对理想误差 | 最大 state 干预 RMS | 重复速度 RMS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | -0.04710 | -0.00099 | 1.00121 | 0.00042229 | 0 |
| 0 | 16 | -0.03124 | -0.00065 | 1.00086 | 0.00036136 | 0 |
| 0 | 31 | -0.08568 | -0.00162 | 1.00180 | 0.00041048 | 0 |
| 50 | 0 | 0.71298 | 0.00135 | 0.99866 | 3.6098e-07 | 0 |
| 50 | 16 | 0.72591 | 0.00140 | 0.99860 | 3.3472e-07 | 0 |
| 50 | 31 | 0.71609 | 0.00129 | 0.99871 | 3.7505e-07 | 0 |

最终检查：`{"fixed_fm_decreased": true, "generated_mse_decreased": true, "position_beats_hold": false, "gripper_beats_hold": false, "rotation_beats_hold": false}`。

这个失败结果不只是卡在任意的 20% loss 阈值：噪声端的实际响应幅度几乎消失、state 扰动响应落至约 2e-7～4e-7、位置误差仍显著差于保持状态基线，这三类证据一致。初始窗口 0 的末层 RMS 为 1.82，50 步后为 42.09；block 2 的 attention 注入仍只有约 0.72，已经不是原来百倍注入的情形。当前证据不足以把剩余退化继续全归因于外部条件尺度。

下一步应隔离最小动作去噪子问题，确认模型能学习对 noisy action 的负向依赖，再逐步接回条件；本轮不执行新的架构/学习率搜索。若采用完整 v3，必须单列为另一配置，同时记录 state 文本化与动作头宽度变化，不能声称仅补了归一化。

训练和评测窗口重叠；这些结果只判断小数据可学习性，不证明泛化、完整预训练收益或机器人闭环成功。梯度探针在 `decoder_*.json`，完整进度、固定评测和控制比较在 `condition_validation/`；原始预测与完整 checkpoint 留在私人 runs，不提交数据。

## 如何读代码

1. `cross_attention_dit.py` 的 `BasicTransformerBlock.forward`：只看 external condition → LN → Attention 三步；再看 DiT 如何传开关。
2. `diagnose_qwenpi_condition.py`：看真实 noisy input 和两个 timestep 的干预，再看三种条件策略。它是诊断工具，不是生产模型。
3. `check_qwenpi_condition_trial.py`：复用生产 trainer 与既有 FixedEvaluation；看配置差异和 50 步决策。
4. `test_qwenpi_condition_norm.py`：验证初始参数/RNG、self-attention、K/V 同输入、梯度、训练/采样和真实首步探针。

## 复现

已有私人环境，仓库根目录，单 PPU；两条命令均要求新的输出目录：

```bash
python examples/umi_pretrain/tools/diagnose_qwenpi_condition.py --reference-run /mnt/workspace/Native_Policy/user/wyt/runs/qwenpi_decoder_layernorm_v1 --output-dir 新的诊断目录
python examples/umi_pretrain/tools/check_qwenpi_condition_trial.py --reference-run /mnt/workspace/Native_Policy/user/wyt/runs/qwenpi_decoder_layernorm_v1 --output-dir 新的试跑目录
```

继续使用 `CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 PYTHONPATH="$PWD"`。本轮没有改视频、数据索引、归一化统计、公共环境或分布式实现。

5 项定向单元测试通过；真实试跑退出码 0，状态 `paused`，loss/全局梯度范数全部有限。峰值 allocated 显存 62.20 GiB。保存计时：`[{"name": "checkpoint_total", "host_seconds": 94.17660517001059}]`。只在第 50 步正常保存一次 basic 完整续训状态。

实际训练代码为 `a0281ea7b8c4d8fbf32dc0fd4f5f84aae58bf436`。结果提交仅修正诊断 JSON 中按通道数组的省略位置并添加文档，不改变模型函数或试跑结论。
