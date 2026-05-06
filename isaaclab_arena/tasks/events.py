# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task-specific event terms."""

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab_tasks.direct.automate import factory_control as fc
from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import sample_object_poses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_poses_and_align_auxiliary_assets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
    fixed_asset_cfg: SceneEntityCfg | None = None,
    auxiliary_asset_cfgs: list[SceneEntityCfg] | None = None,
    randomization_mode: Literal["held_and_fixed_only", "held_fixed_and_auxiliary"] = "held_and_fixed_only",
):
    """
    Randomize object poses and update the poses of related assets accordingly.

    Args:
        randomization_mode:
            - "held_and_fixed_only": Randomize only the fixed and held assets independently.
            - "held_fixed_and_auxiliary": Randomize fixed, held, and auxiliary assets, with auxiliary
              assets positioned relative to the fixed asset.
    """
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])

            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )

            if (
                randomization_mode == "held_fixed_and_auxiliary"
                and auxiliary_asset_cfgs is not None
                and fixed_asset_cfg is not None
                and asset_cfg.name == fixed_asset_cfg.name
            ):
                # Place auxiliary assets at exactly the same pose as the fixed asset (zero offset).
                # NOTE: This assumes the asset USD files have base frames defined such that zero offset creates a valid scene.
                # Currently designed for gear mesh task where all gears share the same center point.
                # For other assets, this may cause geometry intersections. Customers need to adjust it accordingly.
                for j in range(len(auxiliary_asset_cfgs)):
                    rel_asset_cfg = auxiliary_asset_cfgs[j]
                    rel_asset = env.scene[rel_asset_cfg.name]
                    rel_asset.write_root_pose_to_sim(
                        torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
                    )
                    rel_asset.write_root_velocity_to_sim(
                        torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
                    )


class place_gear_in_gripper(ManagerTermBase):
    """Place the held gear in the gripper during reset."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self._robot_name = robot_cfg.name
        self.robot: Articulation = env.scene[robot_cfg.name]

        gear_cfg: SceneEntityCfg = cfg.params["gear_cfg"]
        self._gear_name = gear_cfg.name
        self.gear = env.scene[gear_cfg.name]

        self.hand_grasp_width: float = cast(float, cfg.params["hand_grasp_width"])
        self.hand_close_width: float = cast(float, cfg.params["hand_close_width"])
        self.gripper_joint_setter_func: Callable[..., None] = cast(
            Callable[..., None],
            cfg.params["gripper_joint_setter_func"],
        )

        grasp_rot_offset = cfg.params["grasp_rot_offset"]
        self._grasp_rot_offset_values = tuple(grasp_rot_offset)
        self.grasp_rot_offset_tensor = (
            torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        )

        grasp_offset = cfg.params["grasp_offset"]
        self._grasp_offset_values = tuple(grasp_offset)
        self.grasp_offset_tensor = torch.tensor(grasp_offset, device=env.device, dtype=torch.float32)

        end_effector_body_name: str = cast(str, cfg.params.get("end_effector_body_name", "panda_hand"))
        finger_joint_names: str = cast(str, cfg.params.get("finger_joint_names", "panda_finger_joint[1-2]"))
        self._eef_name = end_effector_body_name
        self._finger_joint_names = finger_joint_names
        self._resolve_robot_indices(end_effector_body_name, finger_joint_names)

        self._max_ik_iterations: int = cfg.params.get("max_ik_iterations", 10)
        self._pos_threshold: float = cfg.params.get("pos_threshold", 1e-6)
        self._rot_threshold: float = cfg.params.get("rot_threshold", 1e-6)

    def _resolve_robot_indices(self, end_effector_body_name: str, finger_joint_names: str) -> None:
        eef_indices, _ = self.robot.find_bodies([end_effector_body_name])
        if not eef_indices:
            raise ValueError(f"End-effector body '{end_effector_body_name}' not found in robot")
        self.eef_idx: int = eef_indices[0]
        self.jacobi_body_idx: int = self.eef_idx - 1

        self.all_joints, _ = self.robot.find_joints([".*"])
        self.finger_joints, _ = self.robot.find_joints([finger_joint_names])
        if not self.finger_joints:
            raise ValueError(f"Finger joints '{finger_joint_names}' not found in robot")

    def _set_gripper_width(
        self,
        joint_pos: torch.Tensor,
        width: float,
        gripper_joint_setter_func: Callable[..., None],
    ) -> None:
        row_indices = torch.arange(joint_pos.shape[0], device=joint_pos.device)
        gripper_joint_setter_func(joint_pos, row_indices, self.finger_joints, width)

    def _sync_sim_state(self, env: ManagerBasedEnv) -> None:
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)

    def _update_cached_assets(
        self,
        env: ManagerBasedEnv,
        robot_cfg: SceneEntityCfg,
        gear_cfg: SceneEntityCfg | None,
        end_effector_body_name: str,
        finger_joint_names: str,
    ) -> None:
        """Refresh cached scene handles and indices when call-time cfgs change."""
        resolve_robot_indices = (
            end_effector_body_name != self._eef_name or finger_joint_names != self._finger_joint_names
        )
        if robot_cfg.name != self._robot_name:
            self._robot_name = robot_cfg.name
            self.robot = env.scene[robot_cfg.name]
            resolve_robot_indices = True

        if resolve_robot_indices:
            self._resolve_robot_indices(end_effector_body_name, finger_joint_names)
            self._eef_name = end_effector_body_name
            self._finger_joint_names = finger_joint_names

        if gear_cfg is not None and gear_cfg.name != self._gear_name:
            self._gear_name = gear_cfg.name
            self.gear = env.scene[gear_cfg.name]

    def _get_grasp_offsets(
        self,
        env_ids: torch.Tensor,
        num_envs: int,
        device: torch.device,
        grasp_rot_offset: list[float] | None,
        grasp_offset: list[float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return batched grasp offsets for the selected environments."""
        if grasp_rot_offset is None or tuple(grasp_rot_offset) == self._grasp_rot_offset_values:
            grasp_rot_offset_tensor = self.grasp_rot_offset_tensor[env_ids]
        else:
            grasp_rot_offset_tensor = torch.tensor(grasp_rot_offset, device=device, dtype=torch.float32).repeat(
                num_envs, 1
            )

        if grasp_offset is None or tuple(grasp_offset) == self._grasp_offset_values:
            grasp_offset_batch = self.grasp_offset_tensor.unsqueeze(0).expand(num_envs, -1)
        else:
            grasp_offset_batch = (
                torch.tensor(grasp_offset, device=device, dtype=torch.float32).unsqueeze(0).expand(num_envs, -1)
            )

        return grasp_rot_offset_tensor, grasp_offset_batch

    def _run_grasp_ik(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        grasp_rot_offset: torch.Tensor,
        grasp_offset: torch.Tensor,
        max_ik_iterations: int,
        pos_threshold: float,
        rot_threshold: float,
    ) -> torch.Tensor:
        """Move the end-effector to the gear grasp pose using iterative DLS IK.

        Each iteration computes the world-frame grasp pose from the current gear pose,
        applies one damped-least-squares joint update, and flushes the simulation state
        so the next iteration sees updated body transforms.
        """
        joint_vel = torch.zeros(len(env_ids), len(self.all_joints), device=env.device)
        for _ in range(max_ik_iterations):
            # Get the current robot state for this IK iteration.
            joint_pos = wp.to_torch(self.robot.data.joint_pos)[env_ids].clone()
            joint_vel = wp.to_torch(self.robot.data.joint_vel)[env_ids].clone()

            # Build the target grasp pose from the current gear pose.
            gear_pos_w = wp.to_torch(self.gear.data.root_link_pos_w)[env_ids].clone()
            gear_quat_w = wp.to_torch(self.gear.data.root_link_quat_w)[env_ids].clone()
            target_quat = math_utils.quat_mul(gear_quat_w, grasp_rot_offset)
            target_pos = gear_pos_w + math_utils.quat_apply(target_quat, grasp_offset)

            # Compute the end-effector pose error.
            eef_pos = wp.to_torch(self.robot.data.body_pos_w)[env_ids, self.eef_idx]
            eef_quat = wp.to_torch(self.robot.data.body_quat_w)[env_ids, self.eef_idx]
            pos_error, aa_error = fc.get_pose_error(
                fingertip_midpoint_pos=eef_pos,
                fingertip_midpoint_quat=eef_quat,
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            delta_hand_pose = torch.cat((pos_error, aa_error), dim=-1)

            if (
                torch.norm(pos_error, dim=-1).max() < pos_threshold
                and torch.norm(aa_error, dim=-1).max() < rot_threshold
            ):
                break

            # Solve one DLS IK step.
            jacobians = wp.to_torch(self.robot.root_view.get_jacobians()).clone()
            jacobian = jacobians[env_ids, self.jacobi_body_idx, :, :]
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=jacobian,
                device=env.device,
            )

            # Write the updated joint state and refresh sim tensors before the next iteration.
            joint_pos = joint_pos + delta_dof_pos
            joint_vel = torch.zeros_like(joint_pos)
            self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
            self.robot.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
            self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
            self._sync_sim_state(env)

        return joint_vel

    def _write_gripper_width(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        width: float,
        gripper_joint_setter_func: Callable[..., None],
    ) -> None:
        """Write one gripper width and flush the reset state to the simulator."""
        self._set_gripper_width(joint_pos, width, gripper_joint_setter_func)
        self.robot.set_joint_position_target_index(
            target=joint_pos,
            joint_ids=self.all_joints,
            env_ids=env_ids,
        )
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._sync_sim_state(env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        gear_cfg: SceneEntityCfg | None = None,
        hand_grasp_width: float | None = None,
        hand_close_width: float | None = None,
        gripper_joint_setter_func: Callable[..., None] | None = None,
        end_effector_body_name: str = "panda_hand",
        finger_joint_names: str = "panda_finger_joint[1-2]",
        grasp_rot_offset: list[float] | None = None,
        grasp_offset: list[float] | None = None,
        max_ik_iterations: int | None = None,
        pos_threshold: float | None = None,
        rot_threshold: float | None = None,
    ) -> None:
        self._update_cached_assets(env, robot_cfg, gear_cfg, end_effector_body_name, finger_joint_names)
        n = len(env_ids)
        grasp_rot_offset_tensor, grasp_offset_batch = self._get_grasp_offsets(
            env_ids, n, env.device, grasp_rot_offset, grasp_offset
        )
        joint_vel = self._run_grasp_ik(
            env=env,
            env_ids=env_ids,
            grasp_rot_offset=grasp_rot_offset_tensor,
            grasp_offset=grasp_offset_batch,
            max_ik_iterations=self._max_ik_iterations if max_ik_iterations is None else max_ik_iterations,
            pos_threshold=self._pos_threshold if pos_threshold is None else pos_threshold,
            rot_threshold=self._rot_threshold if rot_threshold is None else rot_threshold,
        )

        joint_pos = wp.to_torch(self.robot.data.joint_pos)[env_ids].clone()
        gripper_joint_setter = gripper_joint_setter_func or self.gripper_joint_setter_func
        for width in (
            self.hand_grasp_width if hand_grasp_width is None else hand_grasp_width,
            self.hand_close_width if hand_close_width is None else hand_close_width,
        ):
            self._write_gripper_width(env, env_ids, joint_pos, joint_vel, width, gripper_joint_setter)
