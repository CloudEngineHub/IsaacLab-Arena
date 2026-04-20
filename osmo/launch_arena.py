# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and launch an Isaac Lab Arena OSMO workflow.

Usage examples:

    # Default evaluation (zero_action on kitchen_pick_and_place)
    python osmo/launch_arena.py --pool isaac-dev-l40-03

    # Custom command
    python osmo/launch_arena.py \\
        --pool isaac-dev-l40-03 \\
        --command '/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \\
            --policy_type zero_action --num_steps 500 --headless \\
            kitchen_pick_and_place --object cracker_box --embodiment franka_ik'

    # Run tests instead
    python osmo/launch_arena.py \\
        --pool isaac-dev-l40-03 \\
        --command 'ISAACLAB_ARENA_SUBPROCESS_TIMEOUT=900 \\
            /isaac-sim/python.sh -m pytest -sv --durations=0 -m with_subprocess \\
            isaaclab_arena/tests/'

    # Override resources
    python osmo/launch_arena.py --gpus 2 --platform ovx-l40 --memory 128Gi \\
        --pool isaac-dev-l40-03

    # Dry run (print rendered YAML without submitting)
    python osmo/launch_arena.py --pool isaac-dev-l40-03 --dry-run
"""

from __future__ import annotations

import argparse


from arena_osmo.evaluation_task import DEFAULT_COMMAND, EvaluationTask  # noqa: E402
from arena_osmo.workflow import Workflow  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure and submit an Isaac Lab Arena OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    task = parser.add_argument_group("task")
    task.add_argument(
        "--command",
        default=None,
        help="Shell command to run inside the container (defaults to zero_action evaluation)",
    )

    resources = parser.add_argument_group("resources")
    resources.add_argument("--cpus", type=int, default=15)
    resources.add_argument("--gpus", type=int, default=1)
    resources.add_argument("--memory", default="64Gi")
    resources.add_argument("--storage", default="200Gi")
    resources.add_argument("--platform", default="ovx-l40")

    timeouts = parser.add_argument_group("timeouts")
    timeouts.add_argument("--exec_timeout", default="1d")
    timeouts.add_argument("--queue_timeout", default="2d")

    workflow = parser.add_argument_group("workflow")
    workflow.add_argument("--workflow_name", default="arena-evaluation", help="OSMO workflow name")
    workflow.add_argument("--pool", default=None, help="Target a specific OSMO compute pool")
    workflow.add_argument("--priority", default="NORMAL", choices=["HIGH", "NORMAL", "LOW"])

    parser.add_argument("--dry-run", action="store_true", help="Render without submitting")

    return parser


def main() -> int:
    args = build_parser().parse_args()

    command = args.command
    if command is not None:
        command = " ".join(command.replace("\\\n", " ").split())
    else:
        command = DEFAULT_COMMAND

    task = EvaluationTask(command=command)
    workflow = Workflow(workflow_args=args, tasks=[task])

    if args.dry_run:
        print("[dry-run] Rendered workflow YAML:\n")
        print(workflow.render_yaml())
        return 0

    return workflow.submit(pool=args.pool, priority=args.priority)


if __name__ == "__main__":
    raise SystemExit(main())
