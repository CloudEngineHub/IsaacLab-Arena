# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.scripts.reinforcement_learning import train_rl_games


def test_copy_forwarded_args_preserves_isaac_lab_options():
    argv = [
        "--task",
        "nist_assembled_gear_mesh_osc",
        "--num_envs=64",
        "--headless",
        "--agent_cfg_path",
        "legacy.yaml",
        "--embodiment",
        "franka_nist_gear_osc",
    ]

    assert train_rl_games._copy_forwarded_args(argv) == [
        "--task",
        "nist_assembled_gear_mesh_osc",
        "--num_envs=64",
        "--headless",
    ]


def test_drop_forwarded_args_keeps_environment_specific_options():
    argv = [
        "--task",
        "nist_assembled_gear_mesh_osc",
        "--num_envs",
        "64",
        "--headless",
        "--embodiment",
        "franka_nist_gear_osc",
        "--rl_training_mode",
    ]

    assert train_rl_games._drop_forwarded_args(argv) == [
        "--embodiment",
        "franka_nist_gear_osc",
        "--rl_training_mode",
    ]


def test_remove_deprecated_args_returns_experiment_name():
    remaining, experiment_name = train_rl_games._remove_deprecated_args([
        "--agent_cfg_path",
        "legacy.yaml",
        "--experiment_name",
        "test_run",
        "--rl_training_mode",
    ])

    assert remaining == ["--rl_training_mode"]
    assert experiment_name == "test_run"


def test_normalize_legacy_positional_task():
    assert train_rl_games._normalize_legacy_positional_task(["nist_assembled_gear_mesh_osc", "--num_envs", "64"]) == [
        "--task",
        "nist_assembled_gear_mesh_osc",
        "--num_envs",
        "64",
    ]


def test_normalize_legacy_positional_task_keeps_explicit_task():
    argv = ["--task", "nist_assembled_gear_mesh_osc", "--num_envs", "64"]

    assert train_rl_games._normalize_legacy_positional_task(argv) == argv
