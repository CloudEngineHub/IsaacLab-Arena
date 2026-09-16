# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Current Isaac CAP cable-routing environments, isolated from the legacy port."""

from .registration import register_components

register_components()

from .environment import (  # noqa: E402
    CableRoutingEasyEnvironment,
    CableRoutingEasyEnvironmentCfg,
    CableRoutingMediumEnvironment,
    CableRoutingMediumEnvironmentCfg,
)

__all__ = [
    "CableRoutingEasyEnvironment",
    "CableRoutingEasyEnvironmentCfg",
    "CableRoutingMediumEnvironment",
    "CableRoutingMediumEnvironmentCfg",
    "register_components",
]
