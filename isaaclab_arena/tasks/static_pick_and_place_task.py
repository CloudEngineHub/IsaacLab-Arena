# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pick-and-place task for the static-base G1 (WBC stands the robot in place; no nav)."""

from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.locomanip_pick_and_place_task import (
    LocomanipPickAndPlaceMimicEnvCfg,
    LocomanipPickAndPlaceTask,
)


class StaticPickAndPlaceTask(LocomanipPickAndPlaceTask):
    """Locomanip pick-and-place where the robot stands in place (WBC for balance only).

    Identical termination / scene / event behaviour to ``LocomanipPickAndPlaceTask``; only
    overrides ``get_mimic_env_cfg`` to return a Mimic env cfg whose body subtask group is
    collapsed to a single no-op (no navigation phases to segment).
    """

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        # The G1 WBC Pink action layout is dual-arm by construction; single-arm flows
        # would require a separate embodiment.
        assert arm_mode == ArmMode.DUAL_ARM, "Static pick and place task only supports dual arm mode"
        return StaticPickAndPlaceMimicEnvCfg(
            pick_up_object_name=self.pick_up_object.name,
            destination_name=self.destination_location.name,
        )


@configclass
class StaticPickAndPlaceMimicEnvCfg(LocomanipPickAndPlaceMimicEnvCfg):
    """Mimic env cfg for the static-base G1 pick-and-place task.

    Inherits arm subtasks (``idle_{left,right}`` -> ``grasp_and_idle_{left,right}`` -> final)
    and all datagen knobs from ``LocomanipPickAndPlaceMimicEnvCfg`` so generated dataset
    semantics line up with the locomanip variant. Two overrides:

    1. ``datagen_config.name`` is rebranded ``static_pick_and_place_*`` so generated
       datasets are not confused with locomanip ones in the converter / training pipeline.
    2. ``subtask_configs["body"]`` is replaced with a single subtask spanning the whole
       episode (no ``subtask_term_signal``, so it never triggers segmentation). The
       locomanip version expects the env to emit ``navigate_to_table`` /
       ``navigate_turn_inplace`` / ``navigate_to_bin`` term signals as the robot drives
       between waypoints; in the static env the robot never moves its base, so those
       signals never fire and Mimic would deadlock waiting for them. Collapsing to a
       single no-op subtask lets Mimic treat the body channel as one homogeneous block
       (the recorded body actions are constant ``stand-in-place`` commands anyway).
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = f"static_pick_and_place_{self.pick_up_object_name}_to_{self.destination_name}_D0"

        # Replace the locomanip's 4-step nav body subtask sequence with a single no-op.
        # Common knobs match the locomanip body subtasks (action_noise=0, no interpolation)
        # so the body channel is never perturbed during data generation.
        self.subtask_configs["body"] = [
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                # No subtask_term_signal -> Mimic treats this as the final body subtask
                # (runs to end of demo), matching the "last subtask has no term signal"
                # convention used by the per-arm subtask lists.
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        ]
