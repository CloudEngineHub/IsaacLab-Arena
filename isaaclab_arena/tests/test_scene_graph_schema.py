# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import pathlib

import pytest

from isaaclab_arena.scene_graph import SceneGraphEnvSpec


def test_scene_graph_yaml_loads() -> None:
    repo_root = pathlib.Path(__file__).parents[2]
    path = repo_root / "isaaclab_arena_environments/scene_graphs/Franka_PnP_Sweet_Potato_To_Bowl_scene_graph_0.yaml"

    spec = SceneGraphEnvSpec.from_yaml(path)

    assert spec.name == "Franka_PnP_Sweet_Potato_To_Bowl_scene_graph_0"
    assert spec.node("Sweet_Potato_1").name == "sweet_potato"
    assert spec.node("Bowl").name == "bowl_ycb_robolab"
    assert spec.edge("edge_02").kind == "on"
    assert spec.edge("edge_04").kind == "reach"
    assert spec.edge("edge_05").child == "Bowl"
    assert spec.tasks[0].kind == "pick_and_place"
    assert spec.tasks[0].get_arg("object") == "Sweet_Potato_1"
    assert spec.tasks[0].get_arg("destination") == "Bowl"


def test_scene_graph_parser_rejects_nonstandard_keys() -> None:
    with pytest.raises(AssertionError, match="missing nodes"):
        SceneGraphEnvSpec.from_dict(
            {
                "name": "nonstandard_scene_graph",
                "Nodes": [],
                "Relations": [],
            }
        )


def test_scene_graph_parser_rejects_nonstandard_type_aliases() -> None:
    with pytest.raises(AssertionError, match="unsupported type"):
        SceneGraphEnvSpec.from_dict(
            {
                "name": "nonstandard_scene_graph",
                "nodes": [
                    {"id": "Table", "name": "maple_table_robolab", "type": "Background"},
                ],
                "edges": [],
            }
        )


def test_scene_graph_parser_rejects_nonstandard_task_arg_aliases() -> None:
    with pytest.raises(AssertionError, match="unsupported task_args"):
        SceneGraphEnvSpec.from_dict(
            {
                "name": "nonstandard_scene_graph",
                "nodes": [
                    {"id": "Table", "name": "maple_table_robolab", "type": "background"},
                    {"id": "Sweet_Potato_1", "name": "sweet_potato", "type": "rigid_object"},
                ],
                "edges": [],
                "tasks": [
                    {
                        "id": "Subtask 1",
                        "name": "pick_and_place",
                        "type": "pick_and_place",
                        "task_args": {"Object": "Sweet_Potato_1"},
                    }
                ],
            }
        )


def test_scene_graph_node_asset_overrides_keep_graph_ids_stable() -> None:
    spec = SceneGraphEnvSpec.from_dict(
        {
            "name": "override_scene_graph",
            "nodes": [
                {"id": "Table", "name": "maple_table_robolab", "type": "background"},
                {"id": "Sweet_Potato_1", "name": "sweet_potato", "type": "rigid_object"},
                {"id": "Bowl", "name": "bowl_ycb_robolab", "type": "rigid_object"},
            ],
            "edges": [],
            "tasks": [
                {
                    "id": "Subtask 1",
                    "name": "pick_and_place",
                    "type": "pick_and_place",
                    "task_args": {"object": "Sweet_Potato_1", "destination": "Bowl"},
                }
            ],
        }
    )

    overridden = spec.with_node_asset_overrides({"Sweet_Potato_1": "lemon_01_fruits_veggies_robolab"})

    assert spec.node("Sweet_Potato_1").name == "sweet_potato"
    assert overridden.node("Sweet_Potato_1").name == "lemon_01_fruits_veggies_robolab"
    assert overridden.tasks[0].get_arg("object") == "Sweet_Potato_1"


def test_scene_graph_get_pick_object_node_id_uses_task_object_arg() -> None:
    spec = SceneGraphEnvSpec.from_dict(
        {
            "name": "override_scene_graph",
            "nodes": [
                {"id": "Table", "name": "maple_table_robolab", "type": "background"},
                {"id": "Sweet_Potato_1", "name": "sweet_potato", "type": "rigid_object"},
                {"id": "Bowl", "name": "bowl_ycb_robolab", "type": "rigid_object"},
            ],
            "edges": [],
            "tasks": [
                {
                    "id": "Subtask 1",
                    "name": "pick_and_place",
                    "type": "pick_and_place",
                    "task_args": {"object": "Sweet_Potato_1", "destination": "Bowl"},
                }
            ],
        }
    )

    assert spec.get_pick_object_node_id() == "Sweet_Potato_1"


def test_scene_graph_node_asset_overrides_reject_unknown_nodes() -> None:
    spec = SceneGraphEnvSpec.from_dict(
        {
            "name": "override_scene_graph",
            "nodes": [{"id": "Table", "name": "maple_table_robolab", "type": "background"}],
            "edges": [],
        }
    )

    with pytest.raises(AssertionError, match="unknown nodes"):
        spec.with_node_asset_overrides({"Sweet_Potato_1": "lemon_01_fruits_veggies_robolab"})


def test_scene_graph_task_arg_overrides_update_single_task() -> None:
    spec = SceneGraphEnvSpec.from_dict(
        {
            "name": "override_scene_graph",
            "nodes": [
                {"id": "Table", "name": "maple_table_robolab", "type": "background"},
                {"id": "Sweet_Potato_1", "name": "sweet_potato", "type": "rigid_object"},
                {"id": "Ranch_Dressing_1", "name": "ranch_dressing_hope_robolab", "type": "rigid_object"},
                {"id": "Bowl", "name": "bowl_ycb_robolab", "type": "rigid_object"},
            ],
            "edges": [],
            "tasks": [
                {
                    "id": "Subtask 1",
                    "name": "pick_and_place",
                    "type": "pick_and_place",
                    "task_args": {"object": "Sweet_Potato_1", "destination": "Bowl"},
                }
            ],
        }
    )

    overridden = spec.with_task_arg_overrides({"object": "Ranch_Dressing_1"})

    assert spec.tasks[0].get_arg("object") == "Sweet_Potato_1"
    assert overridden.tasks[0].get_arg("object") == "Ranch_Dressing_1"


def test_scene_graph_cli_object_override_updates_pick_object_asset() -> None:
    spec = SceneGraphEnvSpec.from_dict(
        {
            "name": "override_scene_graph",
            "nodes": [
                {"id": "Table", "name": "maple_table_robolab", "type": "background"},
                {"id": "Sweet_Potato_1", "name": "sweet_potato", "type": "rigid_object"},
                {"id": "Bowl", "name": "bowl_ycb_robolab", "type": "rigid_object"},
            ],
            "edges": [],
            "tasks": [
                {
                    "id": "Subtask 1",
                    "name": "pick_and_place",
                    "type": "pick_and_place",
                    "task_args": {"object": "Sweet_Potato_1", "destination": "Bowl"},
                }
            ],
        }
    )

    overridden = spec.with_cli_overrides(
        argparse.Namespace(
            object="lemon_01_fruits_veggies_robolab",
            scene_graph_node_asset=[],
            scene_graph_task_arg=[],
        )
    )

    assert overridden.node("Sweet_Potato_1").name == "lemon_01_fruits_veggies_robolab"
    assert overridden.tasks[0].get_arg("object") == "Sweet_Potato_1"


def test_scene_graph_cli_overrides_reject_invalid_entries() -> None:
    spec = SceneGraphEnvSpec.from_dict(
        {
            "name": "override_scene_graph",
            "nodes": [{"id": "Sweet_Potato_1", "name": "sweet_potato", "type": "rigid_object"}],
            "edges": [],
            "tasks": [
                {
                    "id": "Subtask 1",
                    "name": "pick_and_place",
                    "type": "pick_and_place",
                    "task_args": {"object": "Sweet_Potato_1", "destination": "Sweet_Potato_1"},
                }
            ],
        }
    )

    with pytest.raises(AssertionError, match="KEY=VALUE"):
        spec.with_cli_overrides(
            argparse.Namespace(object=None, scene_graph_node_asset=["Sweet_Potato_1"], scene_graph_task_arg=[])
        )

    with pytest.raises(AssertionError, match="duplicate"):
        spec.with_cli_overrides(
            argparse.Namespace(
                object=None,
                scene_graph_node_asset=["Sweet_Potato_1=a", "Sweet_Potato_1=b"],
                scene_graph_task_arg=[],
            )
        )
