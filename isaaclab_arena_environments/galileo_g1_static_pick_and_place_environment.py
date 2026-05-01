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
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# Pose tuning constants (all values empirically validated -- see commit history for the
# manual procedure: spawn the destination plate at the locomanip apple z (0.0707) and
# read the final z after gravity-settle; spawn the apple and read its USD AABB):
#
# - SHELF_SURFACE_Z: measured by gravity-settling the plate (whose USD origin sits at
#   its bottom, i.e. BBox min_z = 0); the settled z = -0.030 in env-local frame is the
#   actual shelf-top z in the galileo_locomanip scene.
# - APPLE_USD_ORIGIN_ABOVE_BOTTOM: measured from apple_01.usd (BBox min_z = -0.019,
#   max_z = 0.049). Add this offset to apple_z so the apple's bottom -- not its USD
#   origin -- lands on the shelf surface.
# - SHELF_AIRGAP: keeps PhysX from spawning objects in collider penetration with the
#   shelf on the first sim tick (which would otherwise launch them upward).
SHELF_SURFACE_Z = -0.030
APPLE_USD_ORIGIN_ABOVE_BOTTOM = 0.019
SHELF_AIRGAP = 0.005

# Object XY spawn pose (env-local frame, shelf-relative). X mirrors the locomanip env
# (the only on-shelf X we have ground-truth data for via the brown_box flow). The pickup
# Y also mirrors locomanip (Y=0.18); the destination is offset -0.24 m in Y so the
# plate's 30 cm footprint clears the apple without collision. Earlier we tried Y=0.30
# for the apple but a smoke test showed it rolls off the shelf edge from there.
PICK_UP_OBJECT_SPAWN_XY = (0.5785, 0.18)
DESTINATION_SPAWN_XY = (0.5785, -0.06)


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

        # Robot pose mirrors the locomanip env exactly so the WBC controller stands the
        # robot up in the same shelf-relative spot. The controller dynamically lifts the
        # pelvis to ~z=0.74 at runtime; init_state.pos.z=0 is correct.
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.18, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        pick_up_object_x, pick_up_object_y = PICK_UP_OBJECT_SPAWN_XY
        destination_x, destination_y = DESTINATION_SPAWN_XY
        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(
                    pick_up_object_x,
                    pick_up_object_y,
                    SHELF_SURFACE_Z + APPLE_USD_ORIGIN_ABOVE_BOTTOM + SHELF_AIRGAP,
                ),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        destination.set_initial_pose(
            Pose(
                position_xyz=(destination_x, destination_y, SHELF_SURFACE_Z + SHELF_AIRGAP),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        # We deliberately skip ``patch_g1_locomanip_mimic()`` (which wraps both nav-aware
        # ``DataGenerator.generate`` and recorder patching): WBC is here only to hold the
        # standing pose, so the locomanip-specific generate/navigation P-controller would
        # just fight the user's intent. We *do* need ``patch_recorders()`` though -- it
        # registers ``PostStepFlatPolicyObservationsRecorder``, which writes
        # ``obs_buf["action"]`` into every Mimic-generated dataset. Without it, datasets
        # produced from this env would silently lack the ``"action"`` key and break the
        # shared converter / training pipeline.
        if (
            args_cli.embodiment == "g1_wbc_pink"
            and hasattr(args_cli, "mimic")
            and args_cli.mimic
            and not hasattr(args_cli, "auto")
        ):
            from isaaclab_arena.utils.locomanip_mimic_patch import patch_recorders

            patch_recorders()

        if args_cli.task_description is not None:
            task_description = args_cli.task_description
        else:
            object_label = args_cli.object.replace("_", " ")
            destination_label = args_cli.destination.replace("_", " ")
            task_description = (
                f"Pick up the {object_label} from the shelf and place it onto the "
                f"{destination_label} on the same shelf next to it."
            )

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
