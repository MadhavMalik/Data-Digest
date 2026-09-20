"""JSON sanitization.

NumPy scalars (`np.bool_`, `np.float64`, `np.int64`) are not JSON-serializable,
and they leak into result dictionaries from anywhere a statistic is computed.
Chasing each leak at its source is whack-a-mole; sanitizing once at the
serialization boundary is the fix that stays fixed.

Also handles the other values that break a JSON dump: NaN and infinity (which
are not valid JSON), Paths, sets, and Enums.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Recursively convert `value` into something `json.dumps` accepts.

    NaN and infinity become None rather than the non-standard `NaN`/`Infinity`
    literals: a consumer that receives `null` knows the value is absent, while
    one that receives `NaN` may fail to parse the response at all.
    """
    if _depth > 24:  # cycle guard
        return str(value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # NumPy scalars: `.item()` yields the native Python equivalent.
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), _depth=_depth + 1)

    if isinstance(value, np.ndarray):
        return [to_jsonable(v, _depth=_depth + 1) for v in value.tolist()]

    if isinstance(value, Enum):
        return to_jsonable(value.value, _depth=_depth + 1)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            (k if isinstance(k, str) else str(to_jsonable(k, _depth=_depth + 1))):
                to_jsonable(v, _depth=_depth + 1)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth=_depth + 1) for v in value]

    if hasattr(value, "to_dict"):
        try:
            return to_jsonable(value.to_dict(), _depth=_depth + 1)
        except Exception:  # noqa: BLE001
            pass

    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(mode="json"), _depth=_depth + 1)
        except Exception:  # noqa: BLE001
            pass

    return str(value)
