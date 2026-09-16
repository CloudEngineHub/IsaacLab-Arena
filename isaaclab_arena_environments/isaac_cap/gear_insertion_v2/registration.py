# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Register the v2 gear components under names distinct from the legacy port."""

from __future__ import annotations

from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, TaskRegistry

_registered = False
_registering = False


def _register(registry, component, name: str) -> None:
    """Register one v2 component idempotently."""
    if registry.is_registered(name, ensure_loaded=False):
        existing = registry.get_component_by_name(name)
        assert existing is component, f"Conflicting Isaac CAP v2 registration for {name!r}."
        return
    registry.register(component, name)


def register_components() -> None:
    """Register the isolated v2 assets, embodiment, task, and environments."""
    global _registered, _registering
    if _registered or _registering:
        return

    _registering = True
    try:
        from .asset_factories import GEAR_ASSET_ENTRY_POINTS
        from .embodiment import IndustrialFr3Robotiq2f85DifferentialIKEmbodiment, IndustrialFr3Robotiq2f85Embodiment
        from .gear_mesh_environment import (
            GearInsertionEasyNewtonEnvironment,
            GearInsertionEasyNewtonEnvironmentCfg,
            GearMeshPairNewtonEnvironment,
            GearMeshPairNewtonEnvironmentCfg,
            GearMeshTrainNewtonEnvironment,
            GearMeshTrainNewtonEnvironmentCfg,
        )
        from .task import GearMeshTask

        asset_registry = AssetRegistry()
        _register(asset_registry, IndustrialFr3Robotiq2f85Embodiment, "industrial_fr3_robotiq_2f85_v2")
        _register(
            asset_registry,
            IndustrialFr3Robotiq2f85DifferentialIKEmbodiment,
            "industrial_fr3_robotiq_2f85_differential_ik_v2",
        )
        for name, factory in GEAR_ASSET_ENTRY_POINTS.items():
            _register(asset_registry, factory, name)

        _register(TaskRegistry(), GearMeshTask, "GearMeshTaskV2")

        environment_registry = EnvironmentRegistry()
        for factory, cfg_type in (
            (GearInsertionEasyNewtonEnvironment, GearInsertionEasyNewtonEnvironmentCfg),
            (GearMeshPairNewtonEnvironment, GearMeshPairNewtonEnvironmentCfg),
            (GearMeshTrainNewtonEnvironment, GearMeshTrainNewtonEnvironmentCfg),
        ):
            if environment_registry.is_registered(factory.name, ensure_loaded=False):
                existing = environment_registry.get_component_by_name(factory.name)
                assert existing is factory, f"Conflicting Isaac CAP v2 environment {factory.name!r}."
                continue
            environment_registry.register_environment(factory, cfg_type)
        _registered = True
    finally:
        _registering = False


__all__ = ["register_components"]
