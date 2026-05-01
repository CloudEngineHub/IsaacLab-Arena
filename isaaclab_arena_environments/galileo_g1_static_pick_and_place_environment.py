# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Static-base G1 pick-and-place environment (WBC stands the robot in place; no nav).

This is a same-shelf-only variant of ``galileo_g1_locomanip_pick_and_place``: same
``galileo_locomanip`` background, same ``g1_wbc_pink`` embodiment, same OpenXR retargeter,
same 23-D action layout. The only differences are:

1. The destination plate sits on the *same* shelf as the apple (within arm's reach), so
   the robot never needs to drive its base anywhere -- WBC just holds the standing pose.
2. The Mimic config (``StaticPickAndPlaceMimicEnvCfg``) collapses the locomanip body
   subtask sequence (``navigate_to_table -> ... -> navigate_to_bin -> final``) into a
   single no-op subtask, since the body channel never moves and there are no nav term
   signals to segment on.

Note: an earlier iteration of this env shipped a fixed-base PinkIK variant
(``fix_root_link=True`` + custom 28-D upper-body action layout). It was visually awkward
(welded pelvis, unactuated legs) and required parallel embodiment / scene / action /
observation / event / MimicEnv classes. Switching to WBC + same-shelf placement removes
all of that custom code while still giving us a no-locomotion data-gen surface.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class GalileoG1StaticPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """G1 (WBC-balanced, no nav) pick-and-place on the locomanip warehouse shelf.

    Defaults to the apple-to-plate pairing so this env composes cleanly into the existing
    apple-to-plate workflow (record_demos -> replay -> eval) without requiring locomotion.
    """

    name: str = "galileo_g1_static_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.static_pick_and_place_task import StaticPickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        # Reuse the locomanip background USD: it bakes in lighting and provides the same
        # shelf-in-front-of-robot geometry the locomanip env was tuned against.
        background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination = self.asset_registry.get_asset_by_name(args_cli.destination)()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Pose tuning notes (all values empirically validated -- see
        # ``scripts/measure_static_shelf.py`` style smoke tests):
        #
        #   - Robot pose mirrors the locomanip env exactly so the WBC controller stands
        #     the robot up in the same shelf-relative spot. The controller dynamically
        #     lifts the pelvis to ~z=0.74 at runtime; init_state.pos.z=0 is correct here
        #     (do NOT change to 0.78 -- that's only required for fix_root_link variants).
        #
        #   - SHELF_SURFACE_Z is measured: spawned a plate at z=0.0707 (locomanip's apple
        #     z) and let it gravity-settle; final z was -0.030 in the env-local frame.
        #     Since the plate's USD origin sits at the plate's bottom (BBox min_z = 0),
        #     -0.030 is the actual shelf-top z in this scene.
        #
        #   - APPLE_USD_ORIGIN_ABOVE_BOTTOM is measured from apple_01.usd (BBox min_z =
        #     -0.019, max_z = 0.049). Because the apple's USD origin is 1.9 cm above its
        #     bottom, we add this offset to apple_z so its bottom -- not its USD origin --
        #     lands on the shelf. Otherwise the apple would sit 1.9 cm lower than the plate.
        #
        #   - Apple Y mirrors the locomanip env (Y=0.18) -- the only on-shelf XY point we
        #     have ground-truth data for via the brown_box flow. The plate is offset 24 cm
        #     in -Y so its 30-cm-wide footprint clears the apple without collision.
        #     Earlier we tried Y=0.30 for the apple, but the smoke test showed it rolls off
        #     the shelf edge from there (settled at z=-0.135, not on the shelf surface).
        #
        #   - SHELF_AIRGAP keeps PhysX from spawning objects in collider penetration with
        #     the shelf on the first sim tick (which would otherwise launch them upward).
        SHELF_SURFACE_Z = -0.030
        APPLE_USD_ORIGIN_ABOVE_BOTTOM = 0.019
        SHELF_AIRGAP = 0.005
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.18, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.5785, 0.18, SHELF_SURFACE_Z + APPLE_USD_ORIGIN_ABOVE_BOTTOM + SHELF_AIRGAP),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        destination.set_initial_pose(
            Pose(
                position_xyz=(0.5785, -0.06, SHELF_SURFACE_Z + SHELF_AIRGAP),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        if args_cli.task_description is not None:
            task_description = args_cli.task_description
        else:
            object_label = args_cli.object.replace("_", " ")
            destination_label = args_cli.destination.replace("_", " ")
            task_description = (
                f"Pick up the {object_label} from the shelf and place it onto the "
                f"{destination_label} on the same shelf next to it."
            )

        # NOTE: unlike galileo_g1_locomanip_pick_and_place, we never inject
        # ``navigation_subgoals`` into the embodiment's action cfg, even when --mimic is
        # passed: WBC is here only to hold the standing pose, so the nav P-controller
        # would just fight the user's intent. The locomotion command stays at zero (no
        # thumbstick input from the user during teleop, no body action emission from
        # Mimic since StaticPickAndPlaceMimicEnvCfg uses a single no-op body subtask).

        scene = Scene(assets=[background, pick_up_object, destination])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=StaticPickAndPlaceTask(
                pick_up_object=pick_up_object,
                destination_location=destination,
                background_scene=background,
                episode_length_s=30.0,
                task_description=task_description,
                # Mirror the locomanip env's success thresholds so metrics are comparable.
                force_threshold=0.5,
                velocity_threshold=0.1,
            ),
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="apple_01_objaverse_robolab")
        parser.add_argument("--destination", type=str, default="clay_plates_hot3d_robolab")
        # Default embodiment is g1_wbc_pink: WBC controller balances the robot in place,
        # PinkIK drives the upper body. Identical action layout to the locomanip env (23-D)
        # so the same OpenXR retargeter / Mimic env / converters apply unchanged.
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--task_description",
            type=str,
            default=None,
            help=(
                "Override the natural-language task description. Defaults to a template "
                "derived from --object and --destination."
            ),
        )
