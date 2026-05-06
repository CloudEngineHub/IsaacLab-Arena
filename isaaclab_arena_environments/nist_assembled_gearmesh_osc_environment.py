# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NIST assembled board gear-mesh environment with operational-space torque control."""

from __future__ import annotations

import argparse

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


@register_environment
class NISTAssembledGearMeshOSCEnvironment(ExampleEnvironmentBase):
    """NIST gear insertion using OSC torque control and assembly-style observations."""

    name: str = "nist_assembled_gear_mesh_osc"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab.sim as sim_utils

        import isaaclab_arena_environments.mdp as mdp
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.nist_gear_insertion_task import (
            GearInsertionGeometryCfg,
            GraspCfg,
            NistGearInsertionTask,
        )
        from isaaclab_arena.utils.pose import Pose

        peg_tip_offset = (0.02025, 0.0, 0.025)
        peg_base_offset = (0.02025, 0.0, 0.0)
        success_z_fraction = 0.20
        xy_threshold = 0.0025
        episode_length_s = 15.0

        table = self.asset_registry.get_asset_by_name("table")()
        assembled_board = self.asset_registry.get_asset_by_name("nist_board_assembled")()
        gears_and_base = self.asset_registry.get_asset_by_name("gears_and_base")()
        medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            concatenate_observation_terms=True,
            fixed_asset_name=gears_and_base.name,
            peg_offset=peg_tip_offset,
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, -0.009), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
        assembled_board.set_initial_pose(
            Pose(position_xyz=(0.88, 0.15, -0.009), rotation_xyzw=(0.0, 0.0, -0.7071, 0.7071))
        )
        medium_gear.set_initial_pose(Pose(position_xyz=(0.5462, -0.02386, 0.12858), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        gears_and_base.set_initial_pose(
            Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827))
        )
        scene = Scene(assets=[table, assembled_board, medium_gear, gears_and_base, light])

        grasp_cfg = GraspCfg(**embodiment.get_gear_insertion_grasp_config())
        geometry_cfg = GearInsertionGeometryCfg(
            peg_offset_from_board=list(peg_base_offset),
            peg_offset_for_obs=list(peg_tip_offset),
            success_z_fraction=success_z_fraction,
            xy_threshold=xy_threshold,
        )

        task = NistGearInsertionTask(
            assembled_board=assembled_board,
            held_gear=medium_gear,
            background_scene=table,
            gear_base_asset=gears_and_base,
            geometry_cfg=geometry_cfg,
            episode_length_s=episode_length_s,
            grasp_cfg=grasp_cfg,
            enable_randomization=True,
            rl_training_mode=args_cli.rl_training_mode,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.assembly_env_cfg_callback,
            rl_framework_entry_point="rl_games_cfg_entry_point",
            rl_policy_cfg="isaaclab_arena_examples.policy:nist_gear_insertion_osc_rl_games.yaml",
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="franka_nist_gear_osc", help="Robot embodiment")
        parser.add_argument(
            "--teleop_device", type=str, default=None, help="Teleoperation device (e.g., keyboard, spacemouse)"
        )
        parser.add_argument(
            "--rl_training_mode",
            action="store_true",
            help="Disable success termination (use when training with RL-Games).",
        )
