# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Operational-space action term for NIST gear insertion with a Franka arm.

This module converts normalized policy commands into end-effector pose targets
for the OSC controller and applies task-specific filtering around peg contact.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab_tasks.direct.factory.factory_utils import wrap_yaw
from isaaclab_tasks.direct.forge.forge_utils import get_random_prop_gains

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _action_to_target_yaw(action: torch.Tensor) -> torch.Tensor:
    """Map normalized policy action to the commanded yaw interval."""
    return -math.pi + math.radians(270.0) * (action + 1.0) / 2.0


def _target_yaw_to_action(yaw: torch.Tensor) -> torch.Tensor:
    """Map commanded yaw back to the normalized policy interval."""
    return (yaw + math.pi) / math.radians(270.0) * 2.0 - 1.0


def _wrap_yaw_to_action_range(yaw: torch.Tensor) -> torch.Tensor:
    """Represent wrapped yaw in the policy command interval."""
    yaw = torch.where(yaw > math.pi / 2, yaw - 2 * math.pi, yaw)
    return torch.where(yaw < -math.pi, yaw + 2 * math.pi, yaw)


def _gripper_down_to_yaw_frame_quat(num_envs: int, device: torch.device) -> torch.Tensor:
    """Return the fixed rotation from the gripper-down frame to the yaw frame."""
    return math_utils.quat_from_euler_xyz(
        torch.full((num_envs,), -math.pi, device=device),
        torch.zeros(num_envs, device=device),
        torch.zeros(num_envs, device=device),
    )


def get_nist_gear_insertion_arm_action(
    env: ManagerBasedEnv,
    term_name: str = "arm_action",
) -> NistGearInsertionOscAction:
    """Return the NIST gear insertion OSC action term from an environment."""
    try:
        action_term = env.action_manager.get_term(term_name)
    except KeyError as exc:
        raise KeyError(f"Action term '{term_name}' is required for NIST gear insertion.") from exc
    if not isinstance(action_term, NistGearInsertionOscAction):
        raise TypeError(
            f"Action term '{term_name}' must be {NistGearInsertionOscAction.__name__}; "
            f"got {type(action_term).__name__}."
        )
    return action_term


class NistGearInsertionOscAction(OperationalSpaceControllerAction):
    """Operational-space control action term for NIST gear insertion."""

    cfg: NistGearInsertionOscActionCfg

    def __init__(self, cfg: NistGearInsertionOscActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._smoothed_actions = torch.zeros(self.num_envs, 7, device=self.device)
        self.ema_factor = torch.full((self.num_envs, 1), 0.05, device=self.device)
        self._pos_bounds = torch.tensor(cfg.pos_action_bounds, device=self.device)

        pos_threshold = torch.tensor(cfg.pos_action_threshold, device=self.device)
        rot_threshold = torch.tensor(cfg.rot_action_threshold, device=self.device)
        self._default_pos_thresh = pos_threshold.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._default_rot_thresh = rot_threshold.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._pos_thresh = self._default_pos_thresh.clone()
        self._rot_thresh = self._default_rot_thresh.clone()

        self._peg_offset = torch.tensor(cfg.peg_offset, device=self.device)
        self._fixed_pos_noise_levels = torch.tensor(cfg.fixed_pos_noise_levels, device=self.device)
        self.fixed_pos_noise = torch.zeros(self.num_envs, 3, device=self.device)
        self.contact_thresholds = torch.full((self.num_envs,), 7.5, device=self.device)
        self.force_smooth = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_smoothed_actions = torch.zeros(self.num_envs, 7, device=self.device)

        self.delta_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.success_pred = torch.full((self.num_envs,), -1.0, device=self.device)
        self._pos_dead_zone = torch.tensor(cfg.pos_dead_zone, device=self.device).unsqueeze(0)
        self._rot_dead_zone = cfg.rot_dead_zone
        self._force_body_idx: int | None = None
        self._force_smoothing_factor = cfg.force_smoothing_factor

    def _get_peg_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return the peg target position in each environment frame."""
        origins = self._env.scene.env_origins
        board = self._env.scene[self.cfg.fixed_asset_name]
        pos = wp.to_torch(board.data.root_pos_w) - origins
        quat = wp.to_torch(board.data.root_quat_w)
        offset = self._peg_offset.unsqueeze(0).expand(pos.shape[0], 3)
        peg_pos = pos + math_utils.quat_apply(quat, offset)
        if env_ids is not None:
            return peg_pos[env_ids]
        return peg_pos

    @property
    def action_dim(self) -> int:
        """Number of policy actions consumed by the OSC term."""
        return 7

    @property
    def smoothed_actions(self) -> torch.Tensor:
        """EMA-filtered policy actions used by observations and penalties."""
        return self._smoothed_actions

    @property
    def previous_smoothed_actions(self) -> torch.Tensor:
        """EMA-filtered policy actions from the previous environment step."""
        return self._prev_smoothed_actions

    @property
    def position_thresholds(self) -> torch.Tensor:
        """Per-environment position command limits after reset randomization."""
        return self._pos_thresh

    @property
    def rotation_thresholds(self) -> torch.Tensor:
        """Per-environment orientation command limits after reset randomization."""
        return self._rot_thresh

    def process_actions(self, actions: torch.Tensor):
        self._update_smoothed_force()
        self._update_smoothed_actions(actions)
        self._compute_ee_pose()

        # Convert normalized policy output into a bounded OSC pose command.
        ee_pos_b = self._ee_pose_b[:, :3]
        ee_quat_b = self._ee_pose_b[:, 3:7]
        self._processed_actions[:, :3] = self._compute_target_position(ee_pos_b)
        self._processed_actions[:, 3:7] = self._compute_target_orientation(ee_quat_b)

        self._compute_task_frame_pose()
        self._osc.set_command(
            command=self._processed_actions,
            current_ee_pose_b=self._ee_pose_b,
            current_task_frame_pose_b=self._task_frame_pose_b,
        )

    def _update_smoothed_actions(self, actions: torch.Tensor) -> None:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self._raw_actions[:] = actions
        self._prev_smoothed_actions[:] = self._smoothed_actions
        self._smoothed_actions[:] = self.ema_factor * actions + (1.0 - self.ema_factor) * self._smoothed_actions
        self.success_pred[:] = self._smoothed_actions[:, 6]

    def _ensure_force_body_idx(self) -> None:
        """Resolve the wrist force-sensor body index."""
        if self._force_body_idx is not None:
            return
        body_ids, body_names = self._asset.find_bodies(self.cfg.force_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Body '{self.cfg.force_body_name}' is required by {self.__class__.__name__} on asset "
                f"'{self.cfg.asset_name}'. Use a USD with a wrist force-sensor body or override "
                f"'force_body_name'. Found {len(body_ids)} matches: {body_names}."
            )
        self._force_body_idx = body_ids[0]

    def _update_smoothed_force(self) -> None:
        """Update the wrist-force EMA used by observations and rewards."""
        self._ensure_force_body_idx()
        raw_force = wp.to_torch(self._asset.root_view.get_link_incoming_joint_force())[:, self._force_body_idx, :3]
        raw_force = torch.nan_to_num(raw_force, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
        self.force_smooth[:] = (
            self._force_smoothing_factor * raw_force + (1.0 - self._force_smoothing_factor) * self.force_smooth
        )

    def _compute_target_position(self, ee_pos_b: torch.Tensor) -> torch.Tensor:
        """Return the next end-effector position command in the robot base frame."""
        peg_pos_b = self._get_peg_pos() + self.fixed_pos_noise

        pos_actions = self._smoothed_actions[:, :3]
        target_pos = peg_pos_b + pos_actions * self._pos_bounds

        self.delta_pos[:] = target_pos - ee_pos_b
        clipped_delta = self._clip_delta_with_dead_zone(self.delta_pos, self._pos_thresh, self._pos_dead_zone)
        return ee_pos_b + clipped_delta

    def _compute_target_orientation(self, ee_quat_b: torch.Tensor) -> torch.Tensor:
        """Return the next end-effector orientation command in the robot base frame."""
        target_yaw = _action_to_target_yaw(self._smoothed_actions[:, 5])
        target_quat = self._target_quat_from_yaw(target_yaw)
        desired_xyz = self._clip_orientation_delta(ee_quat_b, target_quat)
        return math_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0],
            pitch=desired_xyz[:, 1],
            yaw=desired_xyz[:, 2],
        )

    def _target_quat_from_yaw(self, target_yaw: torch.Tensor) -> torch.Tensor:
        """Return the gripper-down target orientation for the commanded yaw."""
        zero = torch.zeros_like(target_yaw)
        target_yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, target_yaw)
        gripper_down_quat = math_utils.quat_from_euler_xyz(torch.full_like(target_yaw, math.pi), zero, zero)
        return math_utils.quat_mul(gripper_down_quat, target_yaw_quat)

    def _clip_orientation_delta(self, ee_quat_b: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Clip roll, pitch, and yaw deltas before sending the OSC command."""
        curr_roll, curr_pitch, curr_yaw = math_utils.euler_xyz_from_quat(ee_quat_b, wrap_to_2pi=True)
        desired_roll, desired_pitch, desired_yaw = math_utils.euler_xyz_from_quat(target_quat, wrap_to_2pi=True)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        desired_xyz[:, 0] = self._clip_roll_delta(curr_roll, desired_roll)
        desired_xyz[:, 1] = self._clip_pitch_delta(curr_pitch, desired_pitch)
        desired_xyz[:, 2] = self._clip_yaw_delta(curr_yaw, desired_yaw)
        return desired_xyz

    def _clip_roll_delta(self, curr_roll: torch.Tensor, desired_roll: torch.Tensor) -> torch.Tensor:
        """Return roll target after applying the per-step rotation limit."""
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * math.pi, desired_roll)
        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clamp(delta_roll, -self._rot_thresh[:, 0], self._rot_thresh[:, 0])
        return curr_roll + clipped_roll

    def _clip_pitch_delta(self, curr_pitch: torch.Tensor, desired_pitch: torch.Tensor) -> torch.Tensor:
        """Return pitch target after wrapping into the signed interval and clipping."""
        curr_pitch_w = torch.where(curr_pitch > math.pi, curr_pitch - 2 * math.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * math.pi, desired_pitch)
        desired_pitch_w = torch.where(desired_pitch > math.pi, desired_pitch - 2 * math.pi, desired_pitch)
        delta_pitch = desired_pitch_w - curr_pitch_w
        clipped_pitch = torch.clamp(delta_pitch, -self._rot_thresh[:, 1], self._rot_thresh[:, 1])
        return curr_pitch_w + clipped_pitch

    def _clip_yaw_delta(self, curr_yaw: torch.Tensor, desired_yaw: torch.Tensor) -> torch.Tensor:
        """Return yaw target after wrapping, clipping, and applying the dead zone."""
        curr_yaw = wrap_yaw(curr_yaw)
        desired_yaw = wrap_yaw(desired_yaw)

        self.delta_yaw[:] = desired_yaw - curr_yaw
        clipped_yaw = self._clip_delta_with_dead_zone(
            self.delta_yaw,
            self._rot_thresh[:, 2],
            self._rot_dead_zone,
        )
        return curr_yaw + clipped_yaw

    def _clip_delta_with_dead_zone(
        self,
        delta: torch.Tensor,
        limits: torch.Tensor | float,
        dead_zone: torch.Tensor | float,
    ) -> torch.Tensor:
        """Clamp a requested delta and zero small values."""
        clipped = torch.clamp(delta, -limits, limits)
        return torch.where(torch.abs(clipped) > dead_zone, clipped, torch.zeros_like(clipped))

    def _ee_quat_to_yaw_action(self, ee_quat: torch.Tensor) -> torch.Tensor:
        """Convert the current EE orientation to the normalized policy yaw."""
        n = ee_quat.shape[0]
        gripper_down_to_yaw_frame = _gripper_down_to_yaw_frame_quat(n, self.device)
        yaw_frame_quat = math_utils.quat_mul(gripper_down_to_yaw_frame, ee_quat)
        target_yaw = math_utils.euler_xyz_from_quat(yaw_frame_quat, wrap_to_2pi=True)[2]
        target_yaw = _wrap_yaw_to_action_range(target_yaw)
        return _target_yaw_to_action(target_yaw)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            n = self.num_envs
        elif isinstance(env_ids, slice):
            n = len(range(*env_ids.indices(self.num_envs)))
        elif len(env_ids) == 0:
            return
        else:
            n = len(env_ids)

        lo, hi = self.cfg.ema_factor_range
        self.ema_factor[env_ids] = lo + torch.rand(n, 1, device=self.device) * (hi - lo)

        self._pos_thresh[env_ids] = get_random_prop_gains(
            self._default_pos_thresh[env_ids],
            self.cfg.pos_threshold_noise_level,
            n,
            self.device,
        )
        self._rot_thresh[env_ids] = get_random_prop_gains(
            self._default_rot_thresh[env_ids],
            self.cfg.rot_threshold_noise_level,
            n,
            self.device,
        )

        self.fixed_pos_noise[env_ids] = torch.randn(n, 3, device=self.device) * self._fixed_pos_noise_levels

        ct_lo, ct_hi = self.cfg.contact_threshold_range
        self.contact_thresholds[env_ids] = ct_lo + torch.rand(n, device=self.device) * (ct_hi - ct_lo)

        self.force_smooth[env_ids] = 0.0

        self._compute_ee_pose()
        ee_pos = self._ee_pose_b[env_ids, :3]
        ee_quat = self._ee_pose_b[env_ids, 3:7]

        peg_pos = self._get_peg_pos(env_ids) + self.fixed_pos_noise[env_ids]

        pos_actions = (ee_pos - peg_pos) / self._pos_bounds
        self._smoothed_actions[env_ids, 0:3] = pos_actions

        yaw_action = self._ee_quat_to_yaw_action(ee_quat)

        self._smoothed_actions[env_ids, 3:5] = 0.0
        self._smoothed_actions[env_ids, 5] = yaw_action
        self._smoothed_actions[env_ids, 6] = -1.0

        self._prev_smoothed_actions[env_ids] = self._smoothed_actions[env_ids]
        self.delta_pos[env_ids] = 0.0
        self.delta_yaw[env_ids] = 0.0
        self.success_pred[env_ids] = -1.0


@configclass
class NistGearInsertionOscActionCfg(OperationalSpaceControllerActionCfg):
    """Config for :class:`NistGearInsertionOscAction`."""

    class_type: type[ActionTerm] = NistGearInsertionOscAction

    fixed_asset_name: str = "gears_and_base"
    """Name of the fixed asset that contains the insertion peg."""

    peg_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Local-frame peg offset on :attr:`fixed_asset_name`."""

    fixed_pos_noise_levels: tuple[float, float, float] = (0.001, 0.001, 0.001)
    """Per-axis standard deviation for reset-time target noise."""

    pos_action_bounds: tuple[float, float, float] = (0.05, 0.05, 0.05)
    """Position scale applied to normalized policy actions."""

    pos_action_threshold: tuple[float, float, float] = (0.02, 0.02, 0.02)
    """Maximum per-step position delta sent to the OSC controller."""

    rot_action_threshold: tuple[float, float, float] = (0.097, 0.097, 0.097)
    """Maximum per-step orientation delta sent to the OSC controller."""

    pos_threshold_noise_level: tuple[float, float, float] = (0.25, 0.25, 0.25)
    """Reset-time multiplicative noise for position thresholds."""

    rot_threshold_noise_level: tuple[float, float, float] = (0.29, 0.29, 0.29)
    """Reset-time multiplicative noise for orientation thresholds."""

    ema_factor_range: tuple[float, float] = (0.05, 0.2)
    """Reset-time range for the action EMA factor."""

    contact_threshold_range: tuple[float, float] = (5.0, 10.0)
    """Reset-time wrist-force threshold range."""

    pos_dead_zone: tuple[float, float, float] = (0.0005, 0.0005, 0.0005)
    """Position command dead zone."""

    rot_dead_zone: float = 0.001
    """Orientation command dead zone."""

    force_body_name: str = "force_sensor"
    """Body that exposes the wrist force-sensor joint wrench."""

    force_smoothing_factor: float = 0.25
    """EMA factor used to smooth wrist-force readings."""
