"""
Utilities for TensorDict datasets.
"""

from typing import Any, Dict, Union, cast

import tensordict
import torch
from logic.interfaces import ITraversable
from packaging import version
from tensordict.tensordict import TensorDict

# Version check for tensordict
td_kwargs: Dict[str, Any] = (
    {"_run_checks": False} if version.parse(str(tensordict.__version__)) <= version.parse("0.4.0") else {}
)


def tensordict_collate_fn(
    batch: list[Union[dict, TensorDict]],
) -> Union[dict, TensorDict]:
    """Collate list of TensorDicts or dicts into batched TensorDict or dict."""
    if isinstance(batch[0], TensorDict):
        # We stack the TensorDicts. Return as is.
        # Note: If pin_memory issues occur in the future, we may need a more
        # specialized approach that doesn't lose nested structure.
        return torch.stack(batch)  # type: ignore[arg-type]

    if isinstance(batch[0], ITraversable):  # type: ignore[unsafe-overlap]
        # We recursively collate the values
        return {key: tensordict_collate_fn([d[key] for d in batch]) for key in batch[0].keys()}  # type: ignore[union-attr]

    return cast(Union[dict, TensorDict], torch.stack(batch))  # type: ignore[arg-type]
