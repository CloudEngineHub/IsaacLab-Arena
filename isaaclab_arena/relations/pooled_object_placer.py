# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer import ObjectPlacer
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_result import MultiEnvPlacementResult, PlacementResult

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class PooledObjectPlacer:
    """Object placer that maintains a pool of valid placement layouts.

    Wraps :class:`ObjectPlacer` and solves layouts in batches of ``pool_size``,
    keeping only those that pass validation.  The pool is refilled automatically
    when consumed layouts run out.

    * :meth:`sample_without_replacement` — returns the next *count* layouts
      sequentially.  Auto-refills when exhausted.
    * :meth:`sample_with_replacement` — picks *count* layouts at random
      (non-consuming).  Used for static initial positions.

    Reproducibility is controlled by ``placer_params.placement_seed``: when set,
    refills advance a deterministic seed stream (via ``seed_offset``) and
    ``sample_with_replacement`` draws from a seeded ``random.Random``. When
    ``placement_seed`` is None, both solver init and sampling fall back to
    global RNG state and are non-deterministic.

    Args:
        objects: All objects (including anchors) participating in relation solving.
        placer_params: Parameters forwarded to ``ObjectPlacer`` for the batched solve.
        pool_size: Number of layouts to solve per batch.
    """

    def __init__(
        self,
        objects: list[ObjectBase],
        placer_params: ObjectPlacerParams,
        pool_size: int = 100,
    ) -> None:
        assert pool_size >= 1, f"pool_size must be >= 1, got {pool_size}"

        self._objects = objects
        self._placer = ObjectPlacer(params=placer_params)
        self._pool_size = pool_size
        self._layouts: list[PlacementResult] = []
        self._next_idx: int = 0
        self._rng = random.Random(placer_params.placement_seed)
        self._next_seed_offset: int = 0

        # Pre-solve the initial batch (runs the gradient solver, no simulation is needed).
        self._solve_and_store(pool_size)
        if not self._layouts:
            raise RuntimeError(
                f"Placement pool failed to produce any valid layouts from {pool_size} attempts. "
                "Check object relations and constraints."
            )

    def _compact(self) -> None:
        """Drop consumed layouts and reset the read index to free memory."""
        self._layouts = self._layouts[self._next_idx :]
        self._next_idx = 0

    def _solve_and_store(self, num_layouts: int) -> None:
        """Solve *num_layouts* placements and append valid ones to the pool.

        When no candidates pass strict validation, the best-loss candidates are
        accepted with a warning (matching pre-pool behaviour where validation
        failures were non-fatal).
        """
        self._compact()

        # place() runs: random init → gradient solve → validate → rank.
        # It returns up to num_layouts results; some may fail validation.
        # Advancing seed_offset by the candidate count keeps successive batches on
        # a fresh, non-overlapping seed range so refills don't replay the initial pool.
        seed_offset = self._next_seed_offset
        num_candidates = self._placer.params.max_placement_attempts * num_layouts
        with torch.inference_mode(False):
            result = self._placer.place(
                self._objects, num_envs=num_layouts, result_per_env=True, seed_offset=seed_offset
            )
        # Keep the offset bookkeeping in step with reality: when placement_seed is None,
        # place() ignores seed_offset entirely, so advancing it would suggest a seed stream
        # that does not exist.
        if self._placer.params.placement_seed is not None:
            self._next_seed_offset += num_candidates

        # TODO(@zhx06): Simplify once ObjectPlacer.place() always returns MultiEnvPlacementResult.
        all_results = result.results if isinstance(result, MultiEnvPlacementResult) else [result]
        valid_results = [r for r in all_results if r.success]

        if len(valid_results) < num_layouts:
            print(
                f"Placement pool: solved {num_layouts} candidates,"
                f" {len(valid_results)} valid, {num_layouts - len(valid_results)} failed validation"
            )

        if valid_results:
            self._layouts.extend(valid_results)
        else:
            print("Warning: No candidates passed strict validation. Accepting best-loss layouts as fallback.")
            self._layouts.extend(all_results)

    def sample_without_replacement(self, count: int) -> list[PlacementResult]:
        """Return the next *count* layouts sequentially (without replacement).

        Auto-refills the pool when there are not enough layouts ahead of the read index.

        Raises:
            RuntimeError: If the pool cannot provide *count* layouts after refilling.
        """
        remaining = len(self._layouts) - self._next_idx
        if remaining < count:
            self._solve_and_store(max(self._pool_size, count))  # solve a fresh batch

        remaining = len(self._layouts) - self._next_idx
        if remaining < count:  # still not enough after refill (solver producing too few valid layouts)
            raise RuntimeError(
                f"Placement pool has {remaining} valid layouts but {count} were requested. "
                "The solver is not producing enough valid placements."
            )

        start = self._next_idx
        self._next_idx += count
        return self._layouts[start : self._next_idx]

    def sample_with_replacement(self, count: int) -> list[PlacementResult]:
        """Pick *count* layouts at random with replacement (non-consuming).

        Used by ``resolve_on_reset=False`` to assign initial positions
        that persist across resets.
        """
        return self._rng.choices(self._layouts, k=count)

    @property
    def remaining(self) -> int:
        """Number of layouts not yet consumed by :meth:`sample_without_replacement`."""
        return len(self._layouts) - self._next_idx
