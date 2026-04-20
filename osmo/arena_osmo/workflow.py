# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow class for Isaac Lab Arena OSMO workflows.

Modeled after ``mindmap_osmo.workflow_utils.workflow.Workflow``. Wraps a list
of ``BaseTask`` into an OSMO workflow dict using Arena's ``version: 2`` schema
(``workflow.groups[*].tasks`` with workflow-level ``resources.default`` and
``timeout`` blocks).
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from arena_osmo.base_task import BaseTask
from arena_osmo.yaml_utils import block_literal_str  # noqa: F401  (registers representer)


class Workflow:
    """Builds, renders, and submits an Arena OSMO workflow."""

    def __init__(
        self,
        workflow_args: argparse.Namespace,
        tasks: list[BaseTask],
        group_name: str = "arena",
    ) -> None:
        assert len(tasks) > 0, "Workflow requires at least one task"
        self.workflow_args = workflow_args
        self.tasks = tasks
        self.group_name = group_name

    def create_workflow_dict(self) -> dict[str, Any]:
        """Build the full OSMO workflow dict."""
        return {
            "version": 2,
            "workflow": {
                "name": self.workflow_args.workflow_name,
                "groups": [
                    {
                        "name": self.group_name,
                        "tasks": [t.create_task_dict() for t in self.tasks],
                    }
                ],
                "resources": {"default": self._create_resource_dict()},
                "timeout": {
                    "exec_timeout": self.workflow_args.exec_timeout,
                    "queue_timeout": self.workflow_args.queue_timeout,
                },
            },
        }

    def render_yaml(self) -> str:
        """Render the workflow dict to YAML text."""
        return yaml.dump(
            self.create_workflow_dict(),
            default_flow_style=False,
            sort_keys=False,
            default_style="",
        )

    def submit(self, pool: str | None = None, priority: str | None = None) -> int:
        """Write the rendered YAML to a temp file and invoke ``osmo workflow submit``."""
        rendered = self.render_yaml()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="arena_", delete=False
        ) as f:
            f.write(rendered)
            rendered_path = f.name

        cmd = ["osmo", "workflow", "submit", rendered_path]
        if pool:
            cmd.extend(["--pool", pool])
        if priority:
            cmd.extend(["--priority", priority])

        print(f"Submitting workflow '{self.workflow_args.workflow_name}':")
        print(f"  {' '.join(cmd)}\n")

        try:
            result = subprocess.run(cmd)
            return result.returncode
        finally:
            Path(rendered_path).unlink(missing_ok=True)

    def _create_resource_dict(self) -> dict[str, Any]:
        return {
            "cpu": self.workflow_args.cpus,
            "gpu": self.workflow_args.gpus,
            "memory": self.workflow_args.memory,
            "platform": self.workflow_args.platform,
            "storage": self.workflow_args.storage,
        }
