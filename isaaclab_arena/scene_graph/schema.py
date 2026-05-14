# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any


NODE_TYPES = {"background", "embodiment", "object_reference", "rigid_object"}
EDGE_TYPES = {"is_anchor", "on", "reach"}
TASK_TYPES = {"pick_and_place"}
TASK_ARGS = {
    "background",
    "destination",
    "episode_length_s",
    "object",
    "success_proximity_max_distance",
}


def _coerce_scale(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    assert isinstance(value, (list, tuple)), f"scale must be a list/tuple of 3 floats, got {value!r}"
    assert len(value) == 3, f"scale must contain exactly 3 values, got {value!r}"
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class SceneGraphNode:
    """One logical node in a scene graph.

    ``id`` is the graph identifier used by edges and tasks. ``name`` is the
    registered Arena asset name, except for object references where it is the
    referenced sub-prim name by convention.
    """

    id: str
    name: str
    type: str
    parent: str | None = None
    prim_path: str | None = None
    object_type: str | None = None
    scale: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.type

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> SceneGraphNode:
        node_id = raw.get("id")
        assert node_id is not None, f"node is missing an id: {raw!r}"
        name = raw.get("name")
        assert name is not None, f"node {node_id!r} is missing a name"
        node_type = raw.get("type")
        assert node_type is not None, f"node {node_id!r} is missing a type"
        assert node_type in NODE_TYPES, f"node {node_id!r} has unsupported type {node_type!r}"

        known_keys = {"id", "name", "type", "parent", "prim_path", "object_type", "scale"}
        metadata = {key: value for key, value in raw.items() if key not in known_keys}
        return cls(
            id=str(node_id),
            name=str(name),
            type=str(node_type),
            parent=str(raw["parent"]) if raw.get("parent") is not None else None,
            prim_path=str(raw["prim_path"]) if raw.get("prim_path") is not None else None,
            object_type=str(raw["object_type"]) if raw.get("object_type") is not None else None,
            scale=_coerce_scale(raw.get("scale")),
            metadata=metadata,
        )


@dataclass(frozen=True)
class SceneGraphEdge:
    """One relation edge between graph nodes."""

    id: str
    parent: str
    type: str
    child: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.type

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> SceneGraphEdge:
        edge_id = raw.get("id")
        assert edge_id is not None, f"edge is missing an id: {raw!r}"
        parent = raw.get("parent")
        assert parent is not None, f"edge {edge_id!r} is missing a parent"
        edge_type = raw.get("type")
        assert edge_type is not None, f"edge {edge_id!r} is missing a type"
        assert edge_type in EDGE_TYPES, f"edge {edge_id!r} has unsupported type {edge_type!r}"

        known_keys = {"id", "parent", "child", "type", "params"}
        params = dict(raw.get("params") or {})
        params.update({key: value for key, value in raw.items() if key not in known_keys})
        return cls(
            id=str(edge_id),
            parent=str(parent),
            child=str(raw["child"]) if raw.get("child") is not None else None,
            type=str(edge_type),
            params=params,
        )


@dataclass(frozen=True)
class SceneGraphTask:
    """One task/subtask described by graph node ids."""

    id: str
    name: str
    type: str
    task_args: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.type

    def get_arg(self, name: str, default: Any = None) -> Any:
        return self.task_args.get(name, default)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> SceneGraphTask:
        task_id = raw.get("id")
        assert task_id is not None, f"task is missing an id: {raw!r}"
        name = raw.get("name", task_id)
        task_type = raw.get("type")
        assert task_type is not None, f"task {task_id!r} is missing a type"
        assert task_type in TASK_TYPES, f"task {task_id!r} has unsupported type {task_type!r}"
        raw_args = raw.get("task_args") or {}
        assert isinstance(raw_args, Mapping), f"task_args for task {task_id!r} must be a mapping"
        task_args = dict(raw_args)
        unsupported_args = set(task_args) - set(TASK_ARGS)
        assert not unsupported_args, f"task {task_id!r} has unsupported task_args: {sorted(unsupported_args)}"
        return cls(id=str(task_id), name=str(name), type=str(task_type), task_args=task_args)


@dataclass(frozen=True)
class SceneGraphEnvSpec:
    """A complete scene graph that can materialize an Arena environment."""

    name: str
    nodes: list[SceneGraphNode]
    edges: list[SceneGraphEdge]
    tasks: list[SceneGraphTask] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def add_args_to_parser(parser: Any) -> Any:
        """Add CLI args owned by scene-graph specs."""
        parser.add_argument("--object", type=str, default=None, help="Override the pick object asset.")
        parser.add_argument("--embodiment", type=str, default=None, help="Override the embodiment node asset.")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--scene_graph_node_asset",
            action="append",
            default=[],
            metavar="NODE_ID=ASSET_NAME",
            help=(
                "Override a scene-graph node's registered Arena asset name before building the env. "
                "Repeat for multiple swaps."
            ),
        )
        parser.add_argument(
            "--scene_graph_task_arg",
            action="append",
            default=[],
            metavar="[TASK_ID.]ARG=VALUE",
            help=(
                "Override a scene-graph task arg before building the env. Use ARG=VALUE for a single-task graph "
                "or TASK_ID.ARG=VALUE for multi-task graphs."
            ),
        )
        return parser

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> SceneGraphEnvSpec:
        import yaml

        path = pathlib.Path(path)
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        assert isinstance(raw, Mapping), f"scene graph YAML must contain a mapping at the top level: {path}"
        return cls.from_dict(raw, default_name=path.stem)

    @classmethod
    def from_cli(cls, args_cli: Any) -> SceneGraphEnvSpec:
        """Load a scene graph from CLI args and apply scene-graph-owned overrides."""
        scene_graph_yaml = getattr(args_cli, "scene_graph_yaml", None)
        assert scene_graph_yaml is not None, "--scene_graph_yaml must be provided"
        return cls.from_yaml(scene_graph_yaml).with_cli_overrides(args_cli)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], default_name: str | None = None) -> SceneGraphEnvSpec:
        name = raw.get("name") or default_name
        assert name is not None, "scene graph spec is missing a name"
        assert "nodes" in raw, "scene graph spec is missing nodes"
        assert "edges" in raw, "scene graph spec is missing edges"

        nodes = _parse_nodes(raw["nodes"])
        edges = _parse_edges(raw["edges"])
        tasks = _parse_tasks(raw.get("tasks") or [])

        known_keys = {"name", "nodes", "edges", "tasks"}
        metadata = {key: value for key, value in raw.items() if key not in known_keys}
        return cls(name=str(name), nodes=nodes, edges=edges, tasks=tasks, metadata=metadata)

    def node(self, node_id: str) -> SceneGraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"scene graph node {node_id!r} not found")

    def edge(self, edge_id: str) -> SceneGraphEdge:
        for edge in self.edges:
            if edge.id == edge_id:
                return edge
        raise KeyError(f"scene graph edge {edge_id!r} not found")

    def with_node_asset_overrides(self, overrides: Mapping[str, str] | None) -> SceneGraphEnvSpec:
        """Return a copy with selected node registered asset names replaced.

        The graph ``id`` stays stable so edges and tasks do not need to change.
        Overrides target concrete Arena assets, so object-reference nodes are
        excluded because their ``name`` is a referenced sub-prim convention.
        """
        if not overrides:
            return self

        nodes_by_id = {node.id: node for node in self.nodes}
        unknown_node_ids = set(overrides) - set(nodes_by_id)
        assert not unknown_node_ids, (
            f"scene graph node asset overrides reference unknown nodes: {sorted(unknown_node_ids)}"
        )

        for node_id in overrides:
            assert nodes_by_id[node_id].kind != "object_reference", (
                f"scene graph node asset override {node_id!r} targets an object_reference node"
            )

        nodes = [
            replace(node, name=str(overrides[node.id])) if node.id in overrides else node for node in self.nodes
        ]
        return replace(self, nodes=nodes)

    def with_cli_overrides(self, args_cli: Any) -> SceneGraphEnvSpec:
        """Return a copy with scene-graph CLI overrides applied."""
        node_asset_overrides = _parse_key_value_overrides(
            getattr(args_cli, "scene_graph_node_asset", None), "--scene_graph_node_asset"
        )
        task_arg_overrides = _parse_key_value_overrides(
            getattr(args_cli, "scene_graph_task_arg", None), "--scene_graph_task_arg"
        )

        spec = self.with_task_arg_overrides(task_arg_overrides)
        object_asset = getattr(args_cli, "object", None)
        if object_asset is not None:
            object_node_id = spec.get_pick_object_node_id()
            assert object_node_id not in node_asset_overrides, (
                f"--object and --scene_graph_node_asset both override pick object node {object_node_id!r}"
            )
            node_asset_overrides[object_node_id] = object_asset
        return spec.with_node_asset_overrides(node_asset_overrides)

    def get_pick_object_node_id(self) -> str:
        """Return the graph node id used as the pick object for a single-task pick-and-place graph."""
        assert self.tasks, "scene graph has no tasks, so --object cannot be resolved"
        assert len(self.tasks) == 1, (
            f"scene graph has {len(self.tasks)} tasks, so --object is ambiguous; use --scene_graph_node_asset"
        )
        task = self.tasks[0]
        assert task.kind == "pick_and_place", f"--object is only supported for pick_and_place tasks, got {task.kind!r}"
        object_id = task.get_arg("object")
        assert object_id is not None, f"pick-and-place task {task.id!r} is missing object"
        return str(object_id)

    def with_task_arg_overrides(self, overrides: Mapping[str, Any] | None) -> SceneGraphEnvSpec:
        """Return a copy with selected task args replaced.

        Keys may be ``arg`` when the scene graph has one task, or
        ``task_id.arg`` for multi-task graphs.
        """
        if not overrides:
            return self

        override_by_task_id: dict[str, dict[str, Any]] = {}
        for key, value in overrides.items():
            task_id, arg_name = _resolve_task_arg_override_key(str(key), self.tasks)
            assert arg_name in TASK_ARGS, f"scene graph task arg override {key!r} uses unsupported arg {arg_name!r}"
            override_by_task_id.setdefault(task_id, {})[arg_name] = value

        tasks = []
        for task in self.tasks:
            task_overrides = override_by_task_id.get(task.id)
            if task_overrides is None:
                tasks.append(task)
                continue
            task_args = dict(task.task_args)
            task_args.update(task_overrides)
            tasks.append(replace(task, task_args=task_args))
        return replace(self, tasks=tasks)

    def to_arena_env(self, args_cli: Any | None = None, name: str | None = None):
        """Materialize this scene graph as an ``IsaacLabArenaEnvironment``.

        The returned environment is ready to pass into ``ArenaEnvBuilder``.
        Only Arena runtime imports happen here so the dataclasses remain cheap to
        import in tools and tests.
        """

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        assets_by_id, embodiment, background_ids = self._materialize_nodes(args_cli)
        self._apply_edges(assets_by_id)
        scene = Scene(assets=list(assets_by_id.values()))
        task = self._materialize_task(assets_by_id, background_ids)
        teleop_device = self._materialize_teleop_device(args_cli)
        return IsaacLabArenaEnvironment(
            name=name or self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )

    def _materialize_nodes(self, args_cli: Any | None = None) -> tuple[dict[str, Any], Any | None, list[str]]:
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.registries import AssetRegistry

        asset_registry = AssetRegistry()
        assets_by_id: dict[str, Any] = {}
        embodiment = None
        background_ids: list[str] = []
        reference_nodes: list[SceneGraphNode] = []

        for node in self.nodes:
            if node.kind == "object_reference":
                reference_nodes.append(node)
                continue
            if node.kind == "embodiment":
                embodiment_name = getattr(args_cli, "embodiment", None) or node.name
                embodiment_cls = _get_asset_cls(asset_registry, str(embodiment_name), node)
                embodiment = embodiment_cls(enable_cameras=getattr(args_cli, "enable_cameras", False))
                continue
            if node.kind not in {"background", "rigid_object"}:
                raise ValueError(f"Unsupported scene graph node type {node.type!r} on node {node.id!r}")

            asset_cls = _get_asset_cls(asset_registry, node.name, node)
            kwargs: dict[str, Any] = {}
            if node.scale is not None:
                kwargs["scale"] = node.scale
            asset = _instantiate_asset(asset_cls, kwargs)
            assets_by_id[node.id] = asset
            if node.kind == "background":
                background_ids.append(node.id)

        for node in reference_nodes:
            parent_id = node.parent or _infer_reference_parent(node.id, assets_by_id)
            assert parent_id is not None, f"ObjectReference node {node.id!r} is missing parent"
            assert parent_id in assets_by_id, f"ObjectReference node {node.id!r} has unknown parent {parent_id!r}"
            parent_asset = assets_by_id[parent_id]
            prim_path = node.prim_path or f"{{ENV_REGEX_NS}}/{parent_asset.name}/{node.name}"
            assets_by_id[node.id] = ObjectReference(
                name=node.id,
                prim_path=prim_path,
                parent_asset=parent_asset,
                object_type=_parse_object_type(node.object_type, ObjectType),
            )

        return assets_by_id, embodiment, background_ids

    def _apply_edges(self, assets_by_id: dict[str, Any]) -> None:
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.utils.pose import Pose

        for edge in self.edges:
            parent = assets_by_id.get(edge.parent)
            if edge.kind == "is_anchor":
                assert parent is not None, f"edge {edge.id!r} has unknown parent {edge.parent!r}"
                if parent.get_initial_pose() is None:
                    parent.set_initial_pose(Pose.identity())
                parent.add_relation(IsAnchor())
            elif edge.kind == "on":
                assert parent is not None, f"edge {edge.id!r} has unknown parent {edge.parent!r}"
                assert edge.child is not None, f"ON edge {edge.id!r} is missing child"
                child = assets_by_id.get(edge.child)
                assert child is not None, f"edge {edge.id!r} has unknown child {edge.child!r}"
                child.add_relation(On(parent))
            elif edge.kind == "reach":
                # REACH is a semantic task relation. Arena's placement solver has
                # no Reach relation object today, so the edge stays represented in
                # the dataclass and is intentionally not materialized as a spatial constraint.
                continue
            else:
                raise ValueError(f"Unsupported scene graph edge type {edge.type!r} on edge {edge.id!r}")

    def _materialize_task(self, assets_by_id: dict[str, Any], background_ids: list[str]):
        if not self.tasks:
            return None

        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        task = self.tasks[0]
        if task.kind != "pick_and_place":
            raise ValueError(f"Unsupported scene graph task type {task.type!r} on task {task.id!r}")
        object_id = task.get_arg("object")
        destination_id = task.get_arg("destination")
        assert object_id is not None, f"pick-and-place task {task.id!r} is missing Object"
        assert destination_id is not None, f"pick-and-place task {task.id!r} is missing Destination"

        background_id = task.get_arg("background")
        if background_id is None:
            assert background_ids, f"pick-and-place task {task.id!r} needs a background node"
            background_id = background_ids[0]

        return PickAndPlaceTask(
            pick_up_object=assets_by_id[str(object_id)],
            destination_location=assets_by_id[str(destination_id)],
            background_scene=assets_by_id[str(background_id)],
            episode_length_s=_optional_float(task.get_arg("episode_length_s")),
            success_proximity_max_distance=_optional_float(task.get_arg("success_proximity_max_distance")) or 0.0,
        )

    def _materialize_teleop_device(self, args_cli: Any | None) -> Any | None:
        if args_cli is None or getattr(args_cli, "teleop_device", None) is None:
            return None

        from isaaclab_arena.assets.registries import DeviceRegistry

        return DeviceRegistry().get_device_by_name(args_cli.teleop_device)()


def _parse_nodes(raw: Any) -> list[SceneGraphNode]:
    assert isinstance(raw, list), f"nodes must be a list, got {type(raw).__name__}"
    nodes: list[SceneGraphNode] = []
    for entry in raw:
        assert isinstance(entry, Mapping), f"node entries must be mappings, got {entry!r}"
        nodes.append(SceneGraphNode.from_raw(entry))
    return nodes


def _parse_edges(raw: Any) -> list[SceneGraphEdge]:
    assert isinstance(raw, list), f"edges must be a list, got {type(raw).__name__}"
    edges: list[SceneGraphEdge] = []
    for entry in raw:
        assert isinstance(entry, Mapping), f"edge entries must be mappings, got {entry!r}"
        edges.append(SceneGraphEdge.from_raw(entry))
    return edges


def _parse_tasks(raw: Any) -> list[SceneGraphTask]:
    assert isinstance(raw, list), f"tasks must be a list, got {type(raw).__name__}"
    tasks: list[SceneGraphTask] = []
    for entry in raw:
        assert isinstance(entry, Mapping), f"task entries must be mappings, got {entry!r}"
        tasks.append(SceneGraphTask.from_raw(entry))
    return tasks


def _parse_key_value_overrides(raw_overrides: Sequence[str] | None, flag_name: str) -> dict[str, str]:
    """Parse repeated KEY=VALUE CLI overrides."""
    overrides: dict[str, str] = {}
    for raw_override in raw_overrides or []:
        key, separator, value = raw_override.partition("=")
        key = key.strip()
        value = value.strip()
        assert separator, f"{flag_name} entries must use KEY=VALUE, got {raw_override!r}"
        assert key, f"{flag_name} entries must include a non-empty key, got {raw_override!r}"
        assert value, f"{flag_name} entries must include a non-empty value, got {raw_override!r}"
        assert key not in overrides, f"{flag_name} has duplicate override for {key!r}"
        overrides[key] = value
    return overrides


def _resolve_task_arg_override_key(key: str, tasks: list[SceneGraphTask]) -> tuple[str, str]:
    assert tasks, f"scene graph task arg override {key!r} was provided but the scene graph has no tasks"
    if "." not in key:
        assert len(tasks) == 1, (
            f"scene graph task arg override {key!r} is ambiguous for {len(tasks)} tasks; use task_id.arg"
        )
        return tasks[0].id, key

    task_id, arg_name = key.rsplit(".", 1)
    task_ids = {task.id for task in tasks}
    assert task_id in task_ids, f"scene graph task arg override {key!r} references unknown task {task_id!r}"
    assert arg_name, f"scene graph task arg override {key!r} is missing an arg name"
    return task_id, arg_name


def _instantiate_asset(asset_cls: type, kwargs: dict[str, Any]) -> Any:
    try:
        return asset_cls(**kwargs)
    except TypeError as original_error:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("scale", None)
        if fallback_kwargs == kwargs:
            raise
        try:
            return asset_cls(**fallback_kwargs)
        except TypeError:
            raise original_error


def _get_asset_cls(asset_registry: Any, asset_name: str, node: SceneGraphNode) -> type:
    try:
        return asset_registry.get_asset_by_name(asset_name)
    except AssertionError as exc:
        raise AssertionError(
            f"Scene graph node {node.id!r} references unregistered Arena asset {asset_name!r}."
        ) from exc


def _infer_reference_parent(node_id: str, assets_by_id: Mapping[str, Any]) -> str | None:
    parts = node_id.split("_")
    for end in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:end])
        if candidate in assets_by_id:
            return candidate
    return None


def _parse_object_type(value: str | None, object_type_cls: type) -> Any:
    if value is None:
        return object_type_cls.BASE
    object_types = {
        "articulation": object_type_cls.ARTICULATION,
        "base": object_type_cls.BASE,
        "rigid": object_type_cls.RIGID,
    }
    assert value in object_types, f"Unsupported ObjectReference object_type {value!r}"
    return object_types[value]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
