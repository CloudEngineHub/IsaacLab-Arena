# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse


def add_policy_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add policy runner specific arguments to the parser."""
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help="Type of policy to use. This is either a registered policy name or a path to a policy class.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps to run the policy (if num_episodes is not provided)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to run the policy (if num_steps is not provided)",
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default=None,
        help="Language instruction for the policy. Takes precedence over the task's own description.",
    )
    parser.add_argument(
        "--scene_graph_yaml",
        type=str,
        default=None,
        help=(
            "Path to a scene-graph YAML file. When set, policy_runner builds the Arena environment directly from "
            "the YAML and does not require an example-environment subcommand."
        ),
    )
    parser.add_argument(
        "--scene_graph_env_name",
        type=str,
        default=None,
        help="Optional Gym registration name override for --scene_graph_yaml environments.",
    )
