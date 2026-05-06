# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NIST gear insertion task."""

from __future__ import annotations

import math
import numpy as np
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.events import place_gear_in_gripper
from isaaclab_arena.tasks.observations import gear_insertion_observations
from isaaclab_arena.tasks.rewards import gear_insertion_rewards
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import gear_dropped_from_gripper, gear_mesh_insertion_success
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

_DEFAULT_PEG_OFFSET = (2.025e-2, 0.0, 0.0)


@dataclass
class GearInsertionGeometryCfg:
    """Geometry parameters for the insertion target."""

    peg_offset_from_board: list[float] = field(default_factory=lambda: list(_DEFAULT_PEG_OFFSET))
    peg_offset_for_obs: list[float] | None = None
    held_gear_base_offset: list[float] = field(default_factory=lambda: list(_DEFAULT_PEG_OFFSET))
    gear_peg_height: float = 0.02
    success_z_fraction: float = 0.30
    xy_threshold: float = 0.0025
    peg_offset_xy_noise: float = 0.005


@dataclass
class GraspCfg:
    """Embodiment-specific reset grasp parameters."""

    hand_grasp_width: float = 0.03
    hand_close_width: float = 0.0
    gripper_joint_setter_func: Callable | None = None
    end_effector_body_name: str = "panda_hand"
    finger_joint_names: str = "panda_finger_joint[1-2]"
    grasp_rot_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    grasp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    arm_joint_names: str = "panda_joint.*"


class NistGearInsertionTask(TaskBase):
    """Task for inserting the held gear onto a fixed NIST peg."""

    def __init__(
        self,
        assembled_board: Asset,
        held_gear: Asset,
        background_scene: Asset,
        gear_base_asset: Asset | None = None,
        geometry_cfg: GearInsertionGeometryCfg | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        grasp_cfg: GraspCfg | None = None,
        enable_randomization: bool = False,
        disable_drop_terminations: bool = True,
        rl_training_mode: bool = False,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.assembled_board = assembled_board
        self.held_gear = held_gear
        self.background_scene = background_scene
        self.held_gear.disable_reset_pose()
        self._gear_base_asset = gear_base_asset if gear_base_asset is not None else assembled_board
        self.geometry_cfg = geometry_cfg if geometry_cfg is not None else GearInsertionGeometryCfg()
        self.grasp_cfg = grasp_cfg
        self.enable_randomization = enable_randomization
        self.disable_drop_terminations = disable_drop_terminations
        self.rl_training_mode = rl_training_mode
        if self.task_description is None:
            self.task_description = f"Insert the {held_gear.name} onto the gear base on the {assembled_board.name}"

    def get_scene_cfg(self):
        return None

    def get_observation_cfg(self):
        geometry_cfg = self.geometry_cfg
        peg_obs = (
            geometry_cfg.peg_offset_for_obs
            if geometry_cfg.peg_offset_for_obs is not None
            else geometry_cfg.peg_offset_from_board
        )
        return GearInsertionObservationsCfg(
            gear_name=self.held_gear.name,
            board_name=self._gear_base_asset.name,
            peg_offset=peg_obs,
            held_gear_base_offset=geometry_cfg.held_gear_base_offset,
        )

    def get_rewards_cfg(self):
        geometry_cfg = self.geometry_cfg
        return GearInsertionRewardsCfg(
            gear_name=self.held_gear.name,
            board_name=self._gear_base_asset.name,
            peg_offset=geometry_cfg.peg_offset_from_board,
            held_gear_base_offset=geometry_cfg.held_gear_base_offset,
            gear_peg_height=geometry_cfg.gear_peg_height,
            success_z_fraction=geometry_cfg.success_z_fraction,
            xy_threshold=geometry_cfg.xy_threshold,
            peg_offset_xy_noise=geometry_cfg.peg_offset_xy_noise,
        )

    def get_termination_cfg(self):
        geometry_cfg = self.geometry_cfg
        success = TerminationTermCfg(
            func=gear_mesh_insertion_success,
            params={
                "held_object_cfg": SceneEntityCfg(self.held_gear.name),
                "fixed_object_cfg": SceneEntityCfg(self._gear_base_asset.name),
                "gear_base_offset": geometry_cfg.peg_offset_from_board,
                "held_gear_base_offset": geometry_cfg.held_gear_base_offset,
                "gear_peg_height": geometry_cfg.gear_peg_height,
                "success_z_fraction": geometry_cfg.success_z_fraction,
                "xy_threshold": geometry_cfg.xy_threshold,
                "rl_training": self.rl_training_mode,
            },
        )

        cfg = GearInsertionTerminationsCfg(success=success, object_dropped=None)
        # Drop checks are disabled during training to allow recovery.
        if not self.disable_drop_terminations:
            cfg.object_dropped = TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={
                    "minimum_height": self.background_scene.object_min_z,
                    "asset_cfg": SceneEntityCfg(self.held_gear.name),
                },
            )
        if not self.disable_drop_terminations and self.grasp_cfg is not None:
            cfg.gear_dropped_from_gripper = TerminationTermCfg(
                func=gear_dropped_from_gripper,
                params={
                    "gear_cfg": SceneEntityCfg(self.held_gear.name),
                    "robot_cfg": SceneEntityCfg("robot"),
                    "ee_body_name": self.grasp_cfg.end_effector_body_name,
                    "distance_threshold": 0.15,
                },
            )
        return cfg

    def get_events_cfg(self):
        cfg = GearInsertionEventsCfg()
        grasp_cfg = self.grasp_cfg
        if grasp_cfg is not None and grasp_cfg.gripper_joint_setter_func is not None:
            cfg.place_gear = EventTermCfg(
                func=place_gear_in_gripper,
                mode="reset",
                params={
                    "gear_cfg": SceneEntityCfg(self.held_gear.name),
                    "hand_grasp_width": grasp_cfg.hand_grasp_width,
                    "hand_close_width": grasp_cfg.hand_close_width,
                    "gripper_joint_setter_func": grasp_cfg.gripper_joint_setter_func,
                    "end_effector_body_name": grasp_cfg.end_effector_body_name,
                    "finger_joint_names": grasp_cfg.finger_joint_names,
                    "grasp_rot_offset": grasp_cfg.grasp_rot_offset,
                    "grasp_offset": grasp_cfg.grasp_offset,
                },
            )
        if self.enable_randomization:
            if grasp_cfg is None:
                raise ValueError("NIST gear insertion randomization requires an embodiment grasp configuration.")
            arm_joints = grasp_cfg.arm_joint_names
            cfg.held_object_mass = EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.held_gear.name),
                    "mass_distribution_params": (-0.005, 0.005),
                    "operation": "add",
                    "distribution": "uniform",
                },
            )
            cfg.fixed_asset_pose = EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self._gear_base_asset.name),
                    "pose_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, math.radians(15.0)),
                    },
                    "velocity_range": {},
                },
            )
            cfg.robot_actuator_gains = EventTermCfg(
                func=mdp_isaac_lab.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints),
                    "stiffness_distribution_params": (0.75, 1.5),
                    "damping_distribution_params": (0.3, 3.0),
                    "operation": "scale",
                    "distribution": "log_uniform",
                },
            )
            cfg.robot_joint_friction = EventTermCfg(
                func=mdp_isaac_lab.randomize_joint_parameters,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints),
                    "friction_distribution_params": (0.3, 0.7),
                    "operation": "add",
                    "distribution": "uniform",
                },
            )
        return cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        del arm_mode
        raise NotImplementedError("NIST gear insertion does not define a Mimic configuration yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.held_gear,
            offset=np.array([1.5, -0.5, 1.0]),
        )


@configclass
class GearInsertionTerminationsCfg:
    """Termination terms for the gear insertion task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg | None = MISSING
    gear_dropped_from_gripper: TerminationTermCfg | None = None


@configclass
class GearInsertionEventsCfg:
    """Reset and randomization events for gear insertion."""

    reset_all: EventTermCfg = EventTermCfg(
        func=mdp_isaac_lab.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    place_gear: EventTermCfg | None = None
    fixed_asset_pose: EventTermCfg | None = None
    held_object_mass: EventTermCfg | None = None
    robot_actuator_gains: EventTermCfg | None = None
    robot_joint_friction: EventTermCfg | None = None


@configclass
class GearInsertionObservationsCfg:
    """Task-specific observations for the gear insertion task."""

    task_obs: ObsGroup = MISSING

    def __init__(
        self,
        gear_name: str,
        board_name: str,
        peg_offset: list[float],
        held_gear_base_offset: list[float] | None = None,
    ):
        held_offset = held_gear_base_offset if held_gear_base_offset is not None else [2.025e-2, 0.0, 0.0]

        @configclass
        class TaskObsCfg(ObsGroup):
            gear_pos = ObsTerm(
                func=mdp_isaac_lab.root_pos_w,
                params={"asset_cfg": SceneEntityCfg(gear_name)},
            )
            gear_quat = ObsTerm(
                func=mdp_isaac_lab.root_quat_w,
                params={"make_quat_unique": True, "asset_cfg": SceneEntityCfg(gear_name)},
            )
            peg_pos = ObsTerm(
                func=gear_insertion_observations.peg_pos_in_env_frame,
                params={"board_cfg": SceneEntityCfg(board_name), "peg_offset": peg_offset},
            )
            board_quat = ObsTerm(
                func=mdp_isaac_lab.root_quat_w,
                params={"make_quat_unique": True, "asset_cfg": SceneEntityCfg(board_name)},
            )
            peg_delta = ObsTerm(
                func=gear_insertion_observations.peg_delta_from_held_gear_base,
                params={
                    "gear_cfg": SceneEntityCfg(gear_name),
                    "board_cfg": SceneEntityCfg(board_name),
                    "peg_offset": peg_offset,
                    "held_gear_base_offset": held_offset,
                },
            )
            joint_pos = ObsTerm(
                func=mdp_isaac_lab.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            joint_vel = ObsTerm(
                func=mdp_isaac_lab.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            ee_pos_noiseless = ObsTerm(
                func=gear_insertion_observations.body_pos_in_env_frame,
                params={"body_name": "panda_fingertip_centered"},
            )
            ee_quat_noiseless = ObsTerm(
                func=gear_insertion_observations.body_quat_canonical,
                params={"body_name": "panda_fingertip_centered"},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


@configclass
class GearInsertionRewardsCfg:
    """Reward terms for gear insertion."""

    kp_baseline: RewardTermCfg = MISSING
    kp_coarse: RewardTermCfg = MISSING
    kp_fine: RewardTermCfg = MISSING
    engagement_bonus: RewardTermCfg = MISSING
    success_bonus: RewardTermCfg = MISSING
    action_penalty_asset: RewardTermCfg = MISSING
    action_grad_penalty: RewardTermCfg = MISSING
    contact_penalty: RewardTermCfg = MISSING
    success_pred_error: RewardTermCfg = MISSING

    def __init__(
        self,
        gear_name: str,
        board_name: str,
        peg_offset: list[float],
        held_gear_base_offset: list[float],
        gear_peg_height: float,
        success_z_fraction: float,
        xy_threshold: float,
        peg_offset_xy_noise: float = 0.005,
    ):
        gear_cfg = SceneEntityCfg(gear_name)
        board_cfg = SceneEntityCfg(board_name)
        common_params = {
            "gear_cfg": gear_cfg,
            "board_cfg": board_cfg,
            "peg_offset": peg_offset,
            "held_gear_base_offset": held_gear_base_offset,
            "keypoint_scale": 0.15,
            "num_keypoints": 4,
            "peg_offset_xy_noise": peg_offset_xy_noise,
        }
        bonus_params = {
            "gear_cfg": gear_cfg,
            "board_cfg": board_cfg,
            "peg_offset": peg_offset,
            "held_gear_base_offset": held_gear_base_offset,
            "gear_peg_height": gear_peg_height,
            "xy_threshold": xy_threshold,
        }

        self.kp_baseline = RewardTermCfg(
            func=gear_insertion_rewards.gear_peg_keypoint_squashing,
            weight=1.0,
            params={**common_params, "squash_a": 5.0, "squash_b": 4.0},
        )
        self.kp_coarse = RewardTermCfg(
            func=gear_insertion_rewards.gear_peg_keypoint_squashing,
            weight=1.0,
            params={**common_params, "squash_a": 50.0, "squash_b": 2.0},
        )
        self.kp_fine = RewardTermCfg(
            func=gear_insertion_rewards.gear_peg_keypoint_squashing,
            weight=1.0,
            params={**common_params, "squash_a": 100.0, "squash_b": 0.0},
        )
        self.engagement_bonus = RewardTermCfg(
            func=gear_insertion_rewards.gear_insertion_geometry_bonus,
            weight=1.0,
            params={**bonus_params, "z_fraction": 0.90},
        )
        self.success_bonus = RewardTermCfg(
            func=gear_insertion_rewards.gear_insertion_geometry_bonus,
            weight=1.0,
            params={**bonus_params, "z_fraction": success_z_fraction},
        )
        self.action_penalty_asset = RewardTermCfg(
            func=gear_insertion_rewards.osc_action_magnitude_penalty,
            weight=-0.0005,
            params={},
        )
        self.action_grad_penalty = RewardTermCfg(
            func=gear_insertion_rewards.osc_action_delta_penalty,
            weight=-0.01,
            params={},
        )
        self.contact_penalty = RewardTermCfg(
            func=gear_insertion_rewards.wrist_contact_force_penalty,
            weight=-0.001,
            params={},
        )
        self.success_pred_error = RewardTermCfg(
            func=gear_insertion_rewards.success_prediction_error,
            weight=-1.0,
            params={
                "gear_cfg": gear_cfg,
                "board_cfg": board_cfg,
                "peg_offset": peg_offset,
                "held_gear_base_offset": held_gear_base_offset,
                "gear_peg_height": gear_peg_height,
                "success_z_fraction": success_z_fraction,
                "xy_threshold": xy_threshold,
                "delay_until_ratio": 0.25,
            },
        )
