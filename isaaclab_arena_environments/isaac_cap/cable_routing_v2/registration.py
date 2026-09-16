# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Register the v2 cable task and environments without replacing legacy entries."""

from __future__ import annotations

from isaaclab_arena.assets.registries import EnvironmentRegistry, TaskRegistry

_registered = False
_registering = False


def register_components() -> None:
    """Register the isolated v2 cable task and environment factories."""
    global _registered, _registering
    if _registered or _registering:
        return

    _registering = True
    try:
        from .environment import (
            CableRoutingEasyEnvironment,
            CableRoutingEasyEnvironmentCfg,
            CableRoutingMediumEnvironment,
            CableRoutingMediumEnvironmentCfg,
        )
        from .task import CableRoutingTask

        task_registry = TaskRegistry()
        task_name = "CableRoutingTaskV2"
        if task_registry.is_registered(task_name, ensure_loaded=False):
            existing = task_registry.get_component_by_name(task_name)
            assert existing is CableRoutingTask, f"Conflicting Isaac CAP v2 task {task_name!r}."
        else:
            task_registry.register(CableRoutingTask, task_name)

        environment_registry = EnvironmentRegistry()
        for factory, cfg_type in (
            (CableRoutingMediumEnvironment, CableRoutingMediumEnvironmentCfg),
            (CableRoutingEasyEnvironment, CableRoutingEasyEnvironmentCfg),
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
