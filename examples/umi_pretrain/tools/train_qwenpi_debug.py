import json
from datetime import datetime

from torch.utils.data import DataLoader
from starVLA.dataloader.lerobot_datasets import collate_fn

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI


# 测试入口：把传给 FP32 动作头的隐藏状态转为 FP32。
# 转换保留计算图，梯度仍能返回 VLM。
class QwenPITrainCheck(Qwen_PI):
    def _encode_vl_hidden_states(self, *args, **kwargs):
        hidden_states, attention_mask = super()._encode_vl_hidden_states(
            *args, **kwargs
        )
        return [h.float() for h in hidden_states], attention_mask


root = Path("/mnt/workspace/lpy/Native_Policy/user/wyt")
torch.manual_seed(42)

data_cfg = OmegaConf.create({
    "lerobot_version": "v3.0",
    "include_state": True,
    "action_mode": "abs",
    "video_backend": "torchvision_av",
})

cfg = OmegaConf.create({
    "framework": {
        "name": "QwenPI",
        "qwenvl": {
            "base_vlm": str(root / "models/Qwen3-VL-2B-Instruct"),
            "attn_implementation": "sdpa",
        },
        "action_model": {
            "action_dim": 16,
            "state_dim": 16,
            "action_horizon": 16,
        },
    },
    "datasets": {"vla_data": data_cfg},
})

print("读取数据集", flush=True)
dataset = make_LeRobotSingleDataset(
    data_root_dir=root / "datasets",
    data_name="roban_umi_debug",
    robot_type="roban_umi",
    delete_pause_frame=False,
    data_cfg=data_cfg,
)

train_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
    generator=torch.Generator().manual_seed(42),
)

print(f"私人数据集：{len(dataset)} 个训练位置", flush=True)

print("加载模型，参数保留 FP32", flush=True)
model = QwenPITrainCheck(cfg).to(device="cuda", dtype=torch.float32)
model.qwen_vl_interface.model.config.use_cache = False

optimizer = torch.optim.AdamW(
    [
        {"params": model.qwen_vl_interface.parameters(), "lr": 1e-5},
        {"params": model.action_model.parameters(), "lr": 1e-4},
    ],
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.0,
    fused=False,
    foreach=False,
)


max_steps = 1000
save_every = 200

run_dir = root / "runs" / (
    "qwenpi_umi_test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
run_dir.mkdir(parents=True, exist_ok=False)
OmegaConf.save(model.config, run_dir / "config.yaml")

print("训练输出目录：", run_dir, flush=True)

model.train()
data_iterator = iter(train_loader)
recent_losses = []

for step in range(1, max_steps + 1):
    try:
        batch = next(data_iterator)
    except StopIteration:
        data_iterator = iter(train_loader)
        batch = next(data_iterator)

    for sample in batch:
        sample["state"] = np.asarray(sample["state"], dtype=np.float32)
        sample["action"] = np.asarray(sample["action"], dtype=np.float32)

    optimizer.zero_grad(set_to_none=True)

    loss = model(examples=batch)["action_loss"]
    assert torch.isfinite(loss).item(), f"step={step} 损失出现 NaN/Inf"

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
        foreach=False,
    )
    optimizer.step()

    loss_value = loss.item()
    recent_losses.append(loss_value)

    if step == 1 or step % 10 == 0:
        print(
            f"step={step}/{max_steps}  "
            f"loss={loss_value:.6f}  "
            f"区间平均loss={np.mean(recent_losses):.6f}  "
            f"裁剪前梯度范数={grad_norm.item():.4f}",
            flush=True,
        )
        recent_losses.clear()

    if step % save_every == 0 or step == max_steps:
        optimizer.zero_grad(set_to_none=True)

        # 先写临时文件，写完后替换最新权重。
        temporary_path = run_dir / "pytorch_model.pt.tmp"
        checkpoint_path = run_dir / "pytorch_model.pt"

        print(f"step={step}，开始保存权重", flush=True)
        torch.save(model.state_dict(), temporary_path)
        temporary_path.replace(checkpoint_path)

        (run_dir / "progress.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "max_steps": max_steps,
                    "last_train_loss": loss_value,
                    "dataset_size": len(dataset),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("已保存：", checkpoint_path, flush=True)

print(
    "峰值张量显存 GiB：",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
)
print("训练完成，输出目录：", run_dir)