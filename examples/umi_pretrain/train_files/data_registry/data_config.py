"""Roban UMI: four views, 16D measured state, and next-eight-frame targets."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class RobanUMIDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.left_head",
        "video.right_head",
        "video.left_hand",
        "video.right_hand",
    ]

    state_keys = [
        "state.robot1_pose",
        "state.robot1_gripper",
        "state.robot2_pose",
        "state.robot2_gripper",
    ]

    action_keys = [
        "action.robot1_pose",
        "action.robot1_gripper",
        "action.robot2_pose",
        "action.robot2_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(1, 17))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # Raw-value smoke check only; choose training normalization separately.
        return ComposedModalityTransform(
            transforms=[StateActionToTensor(apply_to=self.state_keys + self.action_keys)]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "roban_umi": RobanUMIDataConfig(),
}

# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------

DATASET_NAMED_MIXTURES = {
    "roban_umi_debug": [
        ("roban_umi_debug", 1.0, "roban_umi"),
    ],
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}