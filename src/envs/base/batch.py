"""
Batch handling mixin for BuildEnv.

Mirrors WSmart-Route/logic/src/envs/base/batch.py — manages TorchRL batch-size
synchronization and spec shape updates.
"""

import contextlib

import torch


class BatchMixin:
    """Mixin to handle batch size management and TorchRL compatibility."""

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        if not isinstance(value, torch.Size):
            value = torch.Size([value]) if isinstance(value, int) else torch.Size(value)
        try:
            super(BatchMixin, self.__class__).batch_size.fset(self, value)  # type: ignore[misc]
        except (ValueError, RuntimeError, AttributeError):
            self._batch_size = value
            self._sync_spec_shapes(value)

    def _sync_spec_shapes(self, value: torch.Size) -> None:
        if hasattr(self, "reward_spec"):
            self._safe_set_shape(self.reward_spec, (*value, 1))
        if hasattr(self, "done_spec"):
            self._safe_set_shape(self.done_spec, (*value, 1))
        for spec_name in ["observation_spec", "action_spec", "input_spec", "output_spec"]:
            with contextlib.suppress(KeyError, AttributeError):
                spec = getattr(self, spec_name, None)
                if spec is None:
                    continue
                if hasattr(spec, "shape"):
                    with contextlib.suppress(ValueError, RuntimeError):
                        spec.shape = value
                if hasattr(spec, "items"):
                    for k, v in spec.items():
                        if k in ["done", "terminated", "truncated", "reward"]:
                            self._safe_set_shape(v, (*value, 1))

    @staticmethod
    def _safe_set_shape(s, shp):
        if s is None:
            return
        try:
            s.shape = shp
        except (ValueError, RuntimeError):
            if hasattr(s, "items"):
                for v in s.values():
                    BatchMixin._safe_set_shape(v, shp)
