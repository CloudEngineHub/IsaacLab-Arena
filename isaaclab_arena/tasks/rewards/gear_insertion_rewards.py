# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for the NIST gear insertion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply
from isaaclab_tasks.direct.factory.factory_utils import get_keypoint_offsets, squashing_fn

from isaaclab_arena.tasks.observations.gear_insertion_observations import check_gear_insertion_geometry
from isaaclab_arena_environments.mdp.nist_gear_insertion_osc_action import get_nist_gear_insertion_arm_action

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _KeypointDistanceComputer:
    """Keypoint distance calculator with reusable buffers."""

    def __init__(self, num_envs: int, device: torch.device, num_keypoints: int = 4):
        self.offsets_base = get_keypoint_offsets(num_keypoints, device)
        self.n_kp = self.offsets_base.shape[0]
        self.identity_quat = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
            .repeat(num_envs * self.n_kp, 1)
            .contiguous()
        )
        self.offsets_buf = torch.zeros(num_envs, self.n_kp, 3, device=device, dtype=torch.float32)

    def compute(
        self,
        pos_a: torch.Tensor,
        quat_a: torch.Tensor,
        pos_b: torch.Tensor,
        quat_b: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Return mean keypoint L2 distance."""
        n = pos_a.shape[0]
        offsets = self.offsets_base * scale
        self.offsets_buf[:n] = offsets.unsqueeze(0)
        off_flat = self.offsets_buf[:n].reshape(-1, 3)
        iq = self.identity_quat[: n * self.n_kp]

        def _expand(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(1).expand(-1, self.n_kp, -1).reshape(-1, t.shape[-1])

        kp_a, _ = combine_frame_transforms(_expand(pos_a), _expand(quat_a), off_flat, iq)
        kp_b, _ = combine_frame_transforms(_expand(pos_b), _expand(quat_b), off_flat, iq)
        per_kp_dist = torch.norm(kp_b.reshape(n, self.n_kp, 3) - kp_a.reshape(n, self.n_kp, 3), p=2, dim=-1)
        return per_kp_dist.mean(-1)


def _resolve_offset_tensor(
    values: list[float] | None,
    cached_values: tuple[float, ...],
    cached_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return the cached offset tensor unless the manager supplies different values."""
    if values is None or tuple(values) == cached_values:
        return cached_tensor
    return torch.tensor(values, device=device, dtype=torch.float32)


def _offset_pose_in_env_frame(
    env: ManagerBasedRLEnv,
    asset: RigidObject,
    offset: torch.Tensor,
    num_envs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an asset-local offset pose in each environment frame."""
    root_pos = wp.to_torch(asset.data.root_pos_w)[:num_envs] - env.scene.env_origins[:num_envs]
    root_quat = wp.to_torch(asset.data.root_quat_w)[:num_envs]
    offset = offset.unsqueeze(0).expand(num_envs, 3)
    return root_pos + quat_apply(root_quat, offset), root_quat


class gear_peg_keypoint_squashing(ManagerTermBase):
    """Squashing-function keypoint reward for gear-to-peg alignment."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._peg_offset_values = tuple(cfg.params.get("peg_offset", [0.0, 0.0, 0.0]))
        self.peg_offset = torch.tensor(self._peg_offset_values, device=env.device, dtype=torch.float32)
        self._held_gear_base_offset_values = tuple(cfg.params.get("held_gear_base_offset", [2.025e-2, 0.0, 0.0]))
        self.held_gear_base_offset = torch.tensor(
            self._held_gear_base_offset_values, device=env.device, dtype=torch.float32
        )
        self._xy_noise_range = cfg.params.get("peg_offset_xy_noise", 0.0)
        self._num_keypoints: int = cfg.params.get("num_keypoints", 4)
        self.kp = _KeypointDistanceComputer(env.num_envs, env.device, num_keypoints=self._num_keypoints)
        self._offset_noise = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._xy_noise_range <= 0.0:
            return
        if env_ids is None:
            env_ids = torch.arange(self._offset_noise.shape[0], device=self._offset_noise.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self._offset_noise.shape[0], device=self._offset_noise.device)[env_ids]
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        noise_dev = self._offset_noise.device
        self._offset_noise[env_ids, 0] = (torch.rand(n, device=noise_dev) * 2 - 1) * self._xy_noise_range
        self._offset_noise[env_ids, 1] = (torch.rand(n, device=noise_dev) * 2 - 1) * self._xy_noise_range

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
        board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
        peg_offset: list[float] | None = None,
        held_gear_base_offset: list[float] | None = None,
        keypoint_scale: float = 0.15,
        num_keypoints: int = 4,
        peg_offset_xy_noise: float = 0.0,
        squash_a: float = 50.0,
        squash_b: float = 2.0,
    ) -> torch.Tensor:
        if num_keypoints != self._num_keypoints:
            raise ValueError(
                f"num_keypoints is fixed at term initialization. Expected {self._num_keypoints}, got {num_keypoints}."
            )

        n = env.num_envs
        gear: RigidObject = env.scene[gear_cfg.name]
        held_gear_base_offset_tensor = _resolve_offset_tensor(
            held_gear_base_offset,
            self._held_gear_base_offset_values,
            self.held_gear_base_offset,
            env.device,
        )
        held_base_pos, gear_quat = _offset_pose_in_env_frame(env, gear, held_gear_base_offset_tensor, n)

        board: RigidObject = env.scene[board_cfg.name]
        peg_offset_tensor = _resolve_offset_tensor(peg_offset, self._peg_offset_values, self.peg_offset, env.device)
        target_pos, target_quat = _offset_pose_in_env_frame(env, board, peg_offset_tensor, n)
        offset_noise = self._offset_noise[:n] if peg_offset_xy_noise > 0.0 else 0.0
        target_pos = target_pos + offset_noise
        kp_dist = self.kp.compute(target_pos, target_quat, held_base_pos, gear_quat, scale=keypoint_scale)
        return squashing_fn(kp_dist, squash_a, squash_b)


def _compute_gear_position_success(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg,
    board_cfg: SceneEntityCfg,
    peg_offset: torch.Tensor,
    held_gear_base_offset: torch.Tensor,
    gear_peg_height: float,
    z_fraction: float,
    xy_threshold: float,
) -> torch.Tensor:
    """Return whether the gear meets the XY-centering and Z-depth thresholds."""
    gear: RigidObject = env.scene[gear_cfg.name]
    held_base_pos, _ = _offset_pose_in_env_frame(env, gear, held_gear_base_offset, env.num_envs)

    board: RigidObject = env.scene[board_cfg.name]
    peg_pos, _ = _offset_pose_in_env_frame(env, board, peg_offset, env.num_envs)

    return check_gear_insertion_geometry(held_base_pos, peg_pos, gear_peg_height, z_fraction, xy_threshold)


class gear_insertion_geometry_bonus(ManagerTermBase):
    """Bonus when the gear satisfies the configured insertion geometry."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._peg_offset_values = tuple(cfg.params.get("peg_offset", [0.0, 0.0, 0.0]))
        self._peg_offset = torch.tensor(self._peg_offset_values, device=env.device, dtype=torch.float32)
        self._held_gear_base_offset_values = tuple(cfg.params.get("held_gear_base_offset", [2.025e-2, 0.0, 0.0]))
        self._held_gear_base_offset = torch.tensor(
            self._held_gear_base_offset_values, device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
        board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
        peg_offset: list[float] | None = None,
        held_gear_base_offset: list[float] | None = None,
        gear_peg_height: float = 0.02,
        z_fraction: float = 0.05,
        xy_threshold: float = 0.015,
    ) -> torch.Tensor:
        return _compute_gear_position_success(
            env,
            gear_cfg,
            board_cfg,
            _resolve_offset_tensor(peg_offset, self._peg_offset_values, self._peg_offset, env.device),
            _resolve_offset_tensor(
                held_gear_base_offset,
                self._held_gear_base_offset_values,
                self._held_gear_base_offset,
                env.device,
            ),
            gear_peg_height,
            z_fraction,
            xy_threshold,
        ).float()


class osc_action_magnitude_penalty(ManagerTermBase):
    """Penalize large asset-relative position and yaw commands."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action_term = get_nist_gear_insertion_arm_action(env)
        pos_scale = action_term.position_thresholds.clamp_min(1.0e-6)
        rot_scale = action_term.rotation_thresholds[:, 2].clamp_min(1.0e-6)
        pos_error = torch.norm(action_term.delta_pos / pos_scale, p=2, dim=-1)
        rot_error = torch.abs(action_term.delta_yaw) / rot_scale
        return pos_error + rot_error


class osc_action_delta_penalty(ManagerTermBase):
    """Penalize jerky actions using smoothed action deltas."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action_term = get_nist_gear_insertion_arm_action(env)
        return torch.norm(
            action_term.smoothed_actions - action_term.previous_smoothed_actions,
            p=2,
            dim=-1,
        )


class wrist_contact_force_penalty(ManagerTermBase):
    """Penalize wrist/contact force magnitude above per-episode threshold."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action_term = get_nist_gear_insertion_arm_action(env)
        force_mag = torch.norm(action_term.force_smooth, p=2, dim=-1)
        return torch.nn.functional.relu(force_mag - action_term.contact_thresholds)


class success_prediction_error(ManagerTermBase):
    """Penalize incorrect success predictions from the seventh action dimension."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._pred_scale = 0.0
        self._held_gear_base_offset_values = tuple(cfg.params.get("held_gear_base_offset", [2.025e-2, 0.0, 0.0]))
        self._held_gear_base_offset = torch.tensor(
            self._held_gear_base_offset_values, device=env.device, dtype=torch.float32
        )
        self._peg_offset_values = tuple(cfg.params.get("peg_offset", [0.0, 0.0, 0.0]))
        self._peg_offset = torch.tensor(self._peg_offset_values, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
        board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
        peg_offset: list[float] | None = None,
        held_gear_base_offset: list[float] | None = None,
        gear_peg_height: float = 0.02,
        success_z_fraction: float = 0.05,
        xy_threshold: float = 0.0025,
        delay_until_ratio: float = 0.25,
    ) -> torch.Tensor:
        true_success = _compute_gear_position_success(
            env,
            gear_cfg,
            board_cfg,
            _resolve_offset_tensor(peg_offset, self._peg_offset_values, self._peg_offset, env.device),
            _resolve_offset_tensor(
                held_gear_base_offset,
                self._held_gear_base_offset_values,
                self._held_gear_base_offset,
                env.device,
            ),
            gear_peg_height,
            success_z_fraction,
            xy_threshold,
        )

        # Once enough environments have reached the success geometry, keep the auxiliary loss enabled.
        if true_success.float().mean() >= delay_until_ratio:
            self._pred_scale = 1.0

        arm_osc_action = get_nist_gear_insertion_arm_action(env)
        pred = (arm_osc_action.success_pred + 1.0) / 2.0
        error = torch.abs(true_success.float() - pred)
        return error * self._pred_scale
