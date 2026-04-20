# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluation task for the Isaac Lab Arena OSMO workflow.

This is the programmatic equivalent of the former ``arena_base.yaml`` template.
"""

from typing import Any

from arena_osmo.base_task import BaseTask

DEFAULT_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
DEFAULT_COMMAND = (
    "/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py"
    " --policy_type zero_action"
    " --num_steps 100"
    " --headless"
    " kitchen_pick_and_place"
    " --object cracker_box"
    " --embodiment franka_ik"
)


class EvaluationTask(BaseTask):
    """OSMO task that runs an Isaac Lab Arena evaluation command."""

    def __init__(self, command: str = DEFAULT_COMMAND, image: str = DEFAULT_IMAGE) -> None:
        self.command = command
        self.image = image

    @staticmethod
    def get_task_name() -> str:
        return "master"

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        # LFS-tracked test data uploaded from the local machine.
        return [{"dataset": {"name": "arena-lfs-data"}}]

    def _get_run_script(self) -> str:
        return (
            "set -euxo pipefail\n"
            "\n"
            "# Run ldconfig to ensure shared libraries are found (mirrors entrypoint.sh)\n"
            "ldconfig\n"
            "\n"
            "# Ensure required directories exist\n"
            "mkdir -p /datasets /models /eval\n"
            "\n"
            "# Ensure the Isaac Sim symlink exists\n"
            "[ -e /workspaces/isaaclab_arena/submodules/IsaacLab/_isaac_sim ] || \\\n"
            "  ln -s /isaac-sim/ /workspaces/isaaclab_arena/submodules/IsaacLab/_isaac_sim\n"
            "\n"
            "# Display system info\n"
            "nvidia-smi\n"
            "cd /workspaces/isaaclab_arena\n"
            "\n"
            "# Overwrite LFS pointer stubs with real data uploaded from local machine.\n"
            "# OSMO nests under: {{input:0}}/arena-lfs-data/test_data/\n"
            'if [ -d "{{input:0}}/arena-lfs-data/test_data" ]; then\n'
            '  cp -r "{{input:0}}/arena-lfs-data/test_data/"* \\\n'
            "    /workspaces/isaaclab_arena/isaaclab_arena/tests/test_data/\n"
            "fi\n"
            "\n"
            f"{self.command}\n"
        )
