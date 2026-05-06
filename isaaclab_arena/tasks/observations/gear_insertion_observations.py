# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the NIST gear insertion task."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import axis_angle_from_quat, quat_unique

from isaaclab_arena_environments.mdp.nist_gear_insertion_osc_action import get_nist_gear_insertion_arm_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


def _offset_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    asset: RigidObject,
    offset: tuple[float, ...] | torch.Tensor,
) -> torch.Tensor:
    """Return an asset-local offset position in each environment frame."""
    root_pos = wp.to_torch(asset.data.root_pos_w) - env.scene.env_origins
    root_quat = wp.to_torch(asset.data.root_quat_w)
    if isinstance(offset, torch.Tensor):
        offset_tensor = offset.to(device=env.device, dtype=torch.float32)
    else:
        offset_tensor = torch.tensor(offset, device=env.device, dtype=torch.float32)
    offset_tensor = offset_tensor.unsqueeze(0).expand(env.num_envs, 3)
    return root_pos + math_utils.quat_apply(root_quat, offset_tensor)


def peg_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
    peg_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return the target peg position in each environment frame."""
    board: RigidObject = env.scene[board_cfg.name]
    return _offset_pos_in_env_frame(env, board, peg_offset)


def held_gear_base_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    held_gear_base_offset: tuple[float, ...] = (2.025e-2, 0.0, 0.0),
) -> torch.Tensor:
    """Return the held gear insertion point in each environment frame."""
    gear: RigidObject = env.scene[gear_cfg.name]
    return _offset_pos_in_env_frame(env, gear, held_gear_base_offset)


def body_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_fingertip_centered",
) -> torch.Tensor:
    """Return a robot body position in each environment frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_id = robot.body_names.index(body_name)
    return wp.to_torch(robot.data.body_pos_w)[:, body_id] - env.scene.env_origins


def body_quat_canonical(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_fingertip_centered",
) -> torch.Tensor:
    """Return a robot body quaternion with a unique sign convention."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_id = robot.body_names.index(body_name)
    quat = wp.to_torch(robot.data.body_quat_w)[:, body_id]
    return quat_unique(quat)


def check_gear_insertion_geometry(
    held_base_pos: torch.Tensor,
    peg_pos: torch.Tensor,
    gear_peg_height: float,
    z_fraction: float,
    xy_threshold: float,
) -> torch.Tensor:
    """Return whether the gear is centered and inserted on the peg.

    Args:
        held_base_pos: Position of the held gear insertion base.
        peg_pos: Position of the target peg.
        gear_peg_height: Physical height of the peg.
        z_fraction: Fraction of peg height used for the insertion threshold.
        xy_threshold: Maximum XY distance from the peg center.

    Returns:
        Boolean tensor indicating insertion success.
    """
    xy_dist = torch.norm(held_base_pos[:, :2] - peg_pos[:, :2], dim=-1)
    z_diff = held_base_pos[:, 2] - peg_pos[:, 2]
    return (xy_dist < xy_threshold) & (z_diff < gear_peg_height * z_fraction)


def peg_delta_from_held_gear_base(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
    peg_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
    held_gear_base_offset: tuple[float, ...] = (2.025e-2, 0.0, 0.0),
) -> torch.Tensor:
    """Return the vector from the held gear insertion point to the peg."""
    held_base = held_gear_base_pos_in_env_frame(env, gear_cfg, held_gear_base_offset)
    peg_pos = peg_pos_in_env_frame(env, board_cfg, peg_offset)
    return peg_pos - held_base


class NistGearInsertionPolicyObservations(ManagerTermBase):
    """Policy observation term for OSC-based NIST gear insertion.

    Output layout (per env)::

        fingertip_pos_rel_fixed  (3)
        fingertip_quat           (4)
        ee_linvel                (3)
        ee_angvel                (3)
        ft_force                 (3)
        force_threshold          (1)
        prev_actions             (7)
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        p = cfg.params
        self._robot_name: str = p.get("robot_name", "robot")
        self._board_name: str = p.get("board_name", "gears_and_base")
        self._peg_offset_values = tuple(p.get("peg_offset", [0.0, 0.0, 0.0]))
        self._peg_offset = torch.tensor(self._peg_offset_values, device=env.device)
        self._fingertip_body: str = p.get("fingertip_body_name", "panda_fingertip_centered")
        self._force_body: str = p.get("force_body_name", "force_sensor")

        self._pos_noise: float = p.get("pos_noise_level", 0.00025)
        self._rot_noise_deg: float = p.get("rot_noise_level_deg", 0.1)
        self._force_noise: float = p.get("force_noise_level", 1.0)

        n = env.num_envs
        dev = env.device

        self._fingertip_idx: int | None = None
        self._body_key = (self._robot_name, self._fingertip_body, self._force_body)

        self._flip_quats = torch.ones(n, device=dev)
        self._prev_noisy_pos = torch.zeros(n, 3, device=dev)
        self._prev_noisy_quat = torch.zeros(n, 4, device=dev)
        self._prev_noisy_quat[:, 3] = 1.0

    def _resolve_fingertip_idx(
        self,
        robot_name: str,
        fingertip_body_name: str,
        force_body_name: str,
    ) -> int:
        """Resolve the fingertip body index used by the observation term."""
        body_key = (robot_name, fingertip_body_name, force_body_name)
        if self._fingertip_idx is not None and body_key == self._body_key:
            return self._fingertip_idx

        robot: Articulation = self._env.scene[robot_name]
        for body_name in (fingertip_body_name, force_body_name):
            if body_name not in robot.body_names:
                raise ValueError(
                    f"Body '{body_name}' is missing from robot '{robot_name}'. Use a USD that defines this "
                    "body or override the corresponding observation parameter."
                )
        fingertip_idx = robot.body_names.index(fingertip_body_name)
        if body_key == self._body_key:
            self._fingertip_idx = fingertip_idx
        return fingertip_idx

    def _resolve_peg_offset(self, peg_offset: list[float] | None, device: torch.device) -> torch.Tensor:
        """Return the cached peg offset unless the manager supplies different values."""
        if peg_offset is None or tuple(peg_offset) == self._peg_offset_values:
            return self._peg_offset
        return torch.tensor(peg_offset, device=device, dtype=torch.float32)

    def _sample_noisy_pose(
        self,
        env: ManagerBasedRLEnv,
        ft_pos: torch.Tensor,
        ft_quat: torch.Tensor,
        pos_noise_level: float,
        rot_noise_level_deg: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fingertip pose with configured sensor noise."""
        pos_noise = torch.randn(env.num_envs, 3, device=env.device) * pos_noise_level
        noisy_pos = ft_pos + pos_noise

        rot_noise_axis = torch.randn(env.num_envs, 3, device=env.device)
        rot_noise_axis = F.normalize(rot_noise_axis, dim=1, eps=1e-8)
        rot_noise_angle = torch.randn(env.num_envs, device=env.device) * math.radians(rot_noise_level_deg)
        noisy_quat = math_utils.quat_mul(
            ft_quat,
            math_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis),
        )
        return noisy_pos, quat_unique(noisy_quat)

    def _compute_fingertip_target_delta(
        self,
        env: ManagerBasedRLEnv,
        noisy_pos: torch.Tensor,
        board_name: str,
        peg_offset: torch.Tensor,
    ) -> torch.Tensor:
        """Return fingertip position relative to the noisy peg target."""
        board: RigidObject = env.scene[board_name]
        arm_osc_action = get_nist_gear_insertion_arm_action(env)
        peg_pos = _offset_pos_in_env_frame(env, board, peg_offset)
        noisy_fixed_pos = peg_pos + arm_osc_action.fixed_pos_noise
        return noisy_pos - noisy_fixed_pos

    def _compute_velocities(
        self,
        noisy_pos: torch.Tensor,
        noisy_quat: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate fingertip linear and angular velocity from noisy poses."""
        safe_dt = max(dt, 1e-6)
        ee_linvel = (noisy_pos - self._prev_noisy_pos) / safe_dt

        rot_diff = math_utils.quat_mul(noisy_quat, math_utils.quat_conjugate(self._prev_noisy_quat))
        rot_diff = quat_unique(rot_diff)
        ee_angvel = axis_angle_from_quat(rot_diff) / safe_dt
        ee_angvel[:, 0:2] = 0.0

        self._prev_noisy_pos[:] = noisy_pos
        self._prev_noisy_quat[:] = noisy_quat
        return ee_linvel, ee_angvel

    def _read_smoothed_force(
        self,
        env: ManagerBasedRLEnv,
        arm_osc_action,
        force_noise_level: float,
    ) -> torch.Tensor:
        """Return the smoothed wrist force with configured observation noise."""
        return arm_osc_action.force_smooth + torch.randn(env.num_envs, 3, device=env.device) * force_noise_level

    def reset(self, env_ids: list[int] | None = None):
        """Reset cached noisy pose state for the selected environments."""
        if env_ids is None or len(env_ids) == 0:
            return

        n = len(env_ids)
        dev = self._env.device

        flip = torch.ones(n, device=dev)
        flip[torch.rand(n, device=dev) > 0.5] = -1.0
        self._flip_quats[env_ids] = flip

        fingertip_idx = self._resolve_fingertip_idx(self._robot_name, self._fingertip_body, self._force_body)
        robot: Articulation = self._env.scene[self._robot_name]
        origins = self._env.scene.env_origins
        self._prev_noisy_pos[env_ids] = wp.to_torch(robot.data.body_pos_w)[env_ids, fingertip_idx] - origins[env_ids]
        reset_quat = wp.to_torch(robot.data.body_quat_w)[env_ids, fingertip_idx]
        self._prev_noisy_quat[env_ids] = quat_unique(reset_quat)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_name: str | None = None,
        board_name: str | None = None,
        peg_offset: list[float] | None = None,
        fingertip_body_name: str | None = None,
        force_body_name: str | None = None,
        pos_noise_level: float | None = None,
        rot_noise_level_deg: float | None = None,
        force_noise_level: float | None = None,
    ) -> torch.Tensor:
        """Return the 24-D policy observation tensor."""
        robot_name = self._robot_name if robot_name is None else robot_name
        board_name = self._board_name if board_name is None else board_name
        fingertip_body_name = self._fingertip_body if fingertip_body_name is None else fingertip_body_name
        force_body_name = self._force_body if force_body_name is None else force_body_name
        pos_noise_level = self._pos_noise if pos_noise_level is None else pos_noise_level
        rot_noise_level_deg = self._rot_noise_deg if rot_noise_level_deg is None else rot_noise_level_deg
        force_noise_level = self._force_noise if force_noise_level is None else force_noise_level

        peg_offset_tensor = self._resolve_peg_offset(peg_offset, env.device)
        fingertip_idx = self._resolve_fingertip_idx(robot_name, fingertip_body_name, force_body_name)

        dt = env.step_dt

        robot: Articulation = env.scene[robot_name]
        origins = env.scene.env_origins

        ft_pos = wp.to_torch(robot.data.body_pos_w)[:, fingertip_idx] - origins
        ft_quat = wp.to_torch(robot.data.body_quat_w)[:, fingertip_idx]

        noisy_pos, noisy_quat_full = self._sample_noisy_pose(env, ft_pos, ft_quat, pos_noise_level, rot_noise_level_deg)
        fingertip_pos_rel = self._compute_fingertip_target_delta(env, noisy_pos, board_name, peg_offset_tensor)
        ee_linvel, ee_angvel = self._compute_velocities(noisy_pos, noisy_quat_full, dt)

        obs_quat = noisy_quat_full.clone()
        # The gear is symmetric about the peg axis, so the policy observes the tilt quaternion up to sign.
        obs_quat[:, [2, 3]] = 0.0
        obs_quat = obs_quat * self._flip_quats.unsqueeze(-1)

        arm_osc_action = get_nist_gear_insertion_arm_action(env)
        noisy_force = self._read_smoothed_force(env, arm_osc_action, force_noise_level)

        force_threshold = arm_osc_action.contact_thresholds.unsqueeze(-1)

        prev_actions = arm_osc_action.smoothed_actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs = torch.cat(
            [
                fingertip_pos_rel,
                obs_quat,
                ee_linvel,
                ee_angvel,
                noisy_force,
                force_threshold,
                prev_actions,
            ],
            dim=-1,
        )
        return torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
