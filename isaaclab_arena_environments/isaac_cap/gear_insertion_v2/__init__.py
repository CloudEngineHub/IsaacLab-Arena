# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Current Isaac CAP gear-mesh environments, isolated from the legacy port."""

from .. import register_components as register_legacy_components
from .registration import register_components

register_legacy_components()
register_components()

from .gear_mesh_environment import (  # noqa: E402
    GearInsertionEasyNewtonEnvironment,
    GearInsertionEasyNewtonEnvironmentCfg,
    GearMeshPairNewtonEnvironment,
    GearMeshPairNewtonEnvironmentCfg,
    GearMeshTrainNewtonEnvironment,
    GearMeshTrainNewtonEnvironmentCfg,
)

__all__ = [
    "GearInsertionEasyNewtonEnvironment",
    "GearInsertionEasyNewtonEnvironmentCfg",
    "GearMeshPairNewtonEnvironment",
    "GearMeshPairNewtonEnvironmentCfg",
    "GearMeshTrainNewtonEnvironment",
    "GearMeshTrainNewtonEnvironmentCfg",
]
